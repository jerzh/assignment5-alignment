from __future__ import annotations

from collections import Counter
from typing import Callable, Literal

import torch

from torch.optim import Optimizer
from torch.nn.functional import softmax, log_softmax
from torch.nn.utils import clip_grad_norm_
from transformers import PreTrainedTokenizer, PreTrainedModel


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
    device: str,
) -> dict[str, torch.Tensor]:
    text_list: list[tuple[int, list[int]]] = []
    # tokenize first to get max len
    for prompt, output in zip(prompt_strs, output_strs):
        t_prompt = tokenizer.encode(prompt)
        t_output = tokenizer.encode(output)
        text_list.append((len(t_prompt), t_prompt + t_output))
    B = len(text_list)
    max_len = max(len(t) for _, t in text_list)
    input_ids_full = torch.full((B, max_len), tokenizer.pad_token_id, dtype=torch.long)
    response_mask_full = torch.zeros((B, max_len), dtype=torch.bool)
    for i, (len_p, t) in enumerate(text_list):
        input_ids_full[i, :len(t)] = torch.tensor(t)
        response_mask_full[i, len_p:len(t)] = True
    # Move to gpu after building inputs, slightly faster. Hardcoded to train gpu (0)
    input_ids_full = input_ids_full.to(device)
    response_mask_full = response_mask_full.to(device)
    return {
        "input_ids": input_ids_full[:, :-1],
        "labels": input_ids_full[:, 1:],
        "response_mask": response_mask_full[:, 1:],
    }


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    # outputs = (batch, seq_len, vocab_size)
    log_probs = log_softmax(logits, dim=-1)
    return_dict = {
        "log_probs": torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1),
    }
    if return_token_entropy:
        with torch.no_grad():
            probs = softmax(logits, dim=2)
            return_dict["token_entropy"] = -torch.sum(probs * log_probs, dim=-1)
    return return_dict


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    raw_rewards = []
    total_rewards = Counter()
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, ground_truth)
        raw_rewards.append(reward_dict["reward"])
        total_rewards.update(reward_dict)
    return torch.tensor(raw_rewards), {
        "mean_reward": total_rewards["reward"] / len(rollout_responses),
        "mean_format_reward": total_rewards["format_reward"] / len(rollout_responses),
    }


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
):
    rewards = raw_rewards.reshape(-1, group_size)
    group_mean = rewards.mean(dim=1, keepdim=True)
    # subtract
    if baseline == "mean":
        advantages = rewards - group_mean
    else:
        advantages = rewards
    # divide
    if advantage_normalizer == "std":
        advantages /= torch.std(rewards, dim=1, keepdim=True) + advantage_eps
    elif advantage_normalizer == "mean":
        advantages /= group_mean + advantage_eps
    return advantages.flatten(), {}


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_token_policy_gradient_loss = -raw_rewards_or_advantages.reshape(-1, 1) * policy_log_probs
    return per_token_policy_gradient_loss, {}


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    if loss_normalization == "sequence":
        per_seq_loss = torch.sum(per_token_policy_gradient_loss * mask, dim=1) / mask.sum(dim=1)
        return per_seq_loss.mean()
    else:
        return per_token_policy_gradient_loss.mean() / normalization_constant


def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    microbatch_size = len(repeated_prompts) // gradient_accumulation_steps
    batch_loss = 0
    metadatas = []
    # precompute advantages
    raw_rewards, rewards_metadata = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    advantages, group_rewards_metadata = compute_group_normalized_rewards(raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer)
    for i in range(0, len(repeated_prompts), microbatch_size):
        _prompts = repeated_prompts[i:i+microbatch_size]
        _responses = rollout_responses[i:i+microbatch_size]
        _advantages = advantages[i:i+microbatch_size]
        tokenized = tokenize_prompt_and_output(_prompts, _responses, tokenizer, model.device)
        log_probs_dict = get_response_log_probs(model, tokenized["input_ids"], tokenized["labels"], return_token_entropy=True)
        per_token_loss, loss_metadata = compute_policy_gradient_loss(_advantages.to(model.device), log_probs_dict["log_probs"], importance_reweighting_method, old_log_probs, cliprange, tokenized["response_mask"])
        loss = aggregate_loss_across_microbatch(per_token_loss, tokenized["response_mask"], loss_normalization, normalization_constant) * len(_prompts) / len(repeated_prompts)
        loss.backward()
        # logging
        batch_loss += loss.detach()
        metadatas.append(rewards_metadata | group_rewards_metadata | loss_metadata | {
            # Only consider token entropy over response tokens
            "mean_token_entropy": (log_probs_dict["token_entropy"] * tokenized["response_mask"]).sum().item() / tokenized["response_mask"].sum().item()
        })
    grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss, {
        "sample_prompt": repeated_prompts[0],
        "sample_rollout": rollout_responses[0],
        "grad_norm": grad_norm.item(),
        "mean_token_entropy": sum(m["mean_token_entropy"] for m in metadatas) / len(metadatas),
        "mean_reward": sum(m["mean_reward"] for m in metadatas) / len(metadatas),
        "mean_format_reward": sum(m["mean_format_reward"] for m in metadatas) / len(metadatas),
    }
