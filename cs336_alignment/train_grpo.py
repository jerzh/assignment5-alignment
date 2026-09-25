from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import time
import random
from itertools import chain, repeat
from pathlib import Path

import torch
import wandb

from torch.optim import AdamW
from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.rl_utils import grpo_train_step
from cs336_alignment.vllm_utils import VLLMServer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Transformer LM.")

    # ---- data ----
    p.add_argument("--train-data", type=str, required=True)
    p.add_argument("--val-data", type=str, required=True)
    p.add_argument("--n-train-examples", type=int, default=6400)
    p.add_argument("--n-val-examples", type=int, default=1024)
    p.add_argument("--prompt-path", type=str, default="cs336_alignment/prompts/r1_zero.prompt")

    # ---- model / tokenizer ----
    p.add_argument("--model-name", type=str, default="allenai/OLMo-2-0425-1B")

    # ---- optimizer (AdamW) ----
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    # ---- rollouts / sampling ----
    p.add_argument("--num-rollout-steps", type=int, default=200)
    p.add_argument("--rollout-batch-size", type=int, default=256)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--sampling-temperature", type=float, default=1.0)
    p.add_argument("--sampling-max-tokens", type=int, default=512)

    # ---- GRPO loss / advantages ----
    p.add_argument("--baseline", type=str, choices=["mean", "none"], default="mean")
    p.add_argument("--advantage-eps", type=float, default=1e-6)
    p.add_argument("--advantage-normalizer", type=str,
                   choices=["std", "none", "mean"], default="std")
    p.add_argument("--importance-reweighting-method", type=str,
                   choices=["none", "noclip", "grpo", "gspo"], default="none")
    p.add_argument("--cliprange", type=float, default=None)
    p.add_argument("--loss-normalization", type=str,
                   choices=["sequence", "constant"], default="sequence")
    p.add_argument("--normalization-constant", type=int, default=None)

    # ---- training loop ----
    p.add_argument("--train-batch-size", type=int, default=256)
    p.add_argument("--gradient-accumulation-steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", type=str, default="float32")

    # ---- eval / logging / checkpointing ----
    p.add_argument("--eval-interval", type=int, default=10)
    p.add_argument("--log-interval", type=int, default=40)
    p.add_argument("--checkpoint-interval", type=int, default=1600)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Path to checkpoint to resume training from.")
    p.add_argument("--wandb-project", type=str, default="cs336-assignment-5")
    p.add_argument("--run-name", type=str, default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError("Only f32 supported")
    if args.rollout_batch_size % args.group_size != 0:
        raise ValueError("Group size does not divide rollout batch size")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        run_name = wandb.run.name
    elif args.run_name is None:
        run_name = f"{time.strftime('%Y%m%d-%H%M%S')}"
    else:
        run_name = args.run_name
    ckpt_save_dir = Path(args.checkpoint_dir) / run_name

    with open(args.train_data, "r") as f_train, open(args.val_data, "r") as f_val:
        train_data = random.sample([json.loads(line) for line in f_train.readlines()], args.n_train_examples)
        val_data = random.sample([json.loads(line) for line in f_val.readlines()], args.n_val_examples)

    model, tokenizer = get_model_and_tokenizer(
        args.resume_from or args.model_name,
        device="cuda",
    )
    optimizer = AdamW(
        params=model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    server = VLLMServer(
        model_id="allenai/OLMo-2-0425-1B",
        seed=args.seed,
        gpu=1,
    )
    server.start()
    server.init_weight_sync("cuda:0")
    start_iter = 0
    if args.resume_from is not None:
        obj = torch.load(Path(args.resume_from) / "other_state.pt", weights_only=True)
        optimizer.load_state_dict(obj["optim"])
        start_iter = obj["iter"]

    sampling_params = {
        "temperature": args.sampling_temperature,
        "max_tokens": args.sampling_max_tokens,
        "n": args.group_size,
        "seed": args.seed,
        "stop": "</answer>",
        "include_stop_str_in_output": True,
    }
    val_sampling_params = {
        "temperature": args.sampling_temperature,
        "max_tokens": args.sampling_max_tokens,
        "n": 1,
        "seed": args.seed,
        "stop": "</answer>",
        "include_stop_str_in_output": True,
    }

    for i in range(start_iter + 1, args.num_rollout_steps + 1):
        server.sync_policy_weights(model)

        _train = random.sample(train_data, args.rollout_batch_size // args.group_size)
        with open(args.prompt_path, "r") as f:
            prompt = f.read()
        prompts = [prompt.format(question=obj["question"]) for obj in _train]
        ground_truths = [obj["answer"].split("####")[1].strip() for obj in _train]
        rollout_responses = [c.text for c in server.generate_completions(
            prompts=prompts,
            sampling_params=sampling_params,
        )]
        repeated_prompts = list(chain.from_iterable(repeat(x, args.group_size) for x in prompts))
        repeated_ground_truths = list(chain.from_iterable(repeat(x, args.group_size) for x in ground_truths))
        # note: args.train_batch_size not used currently

        loss, metadata = grpo_train_step(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            reward_fn=r1_zero_reward_fn,
            repeated_prompts=repeated_prompts,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=args.group_size,
            baseline=args.baseline,
            advantage_eps=args.advantage_eps,
            advantage_normalizer=args.advantage_normalizer,
            importance_reweighting_method=args.importance_reweighting_method,
            old_log_probs=None,  # fix later
            cliprange=args.cliprange,
            loss_normalization=args.loss_normalization,
            normalization_constant=args.normalization_constant,
        )

        if i % args.log_interval == 0:
            log_data = {
                "train/loss": loss,
                "train/reward": metadata["mean_reward"],
                "train/format_reward": metadata["mean_format_reward"],
            }
            logging.info(f"iter: {i}  " + "  ".join(f"{k}: {v}" for k, v in log_data.items()))
            if args.wandb_project is not None:
                wandb.log(log_data, step=i)

        if i % args.eval_interval == 0:
            model.eval()
            losses = []
            with open(args.prompt_path, "r") as f:
                prompt = f.read()
            prompts = [prompt.format(question=obj["question"]) for obj in val_data]
            rollout_responses = [c.text for c in server.generate_completions(
                prompts=prompts,
                sampling_params=val_sampling_params,
            )]
            reward_total = collections.Counter()
            for qa_pair, completion in zip(val_data, rollout_responses):
                answer = qa_pair["answer"].split("####")[1].strip()
                logging.info(f"completion: {completion}")
                rewards = r1_zero_reward_fn(completion, answer)
                logging.info(f"format_reward: {rewards['format_reward']}  answer_reward: {rewards['answer_reward']}")
                reward_total.update(rewards)
            mean_reward = reward_total['reward'] / len(val_data)
            mean_format_reward = reward_total['format_reward'] / len(val_data)

            log_data = {
                "val/reward": mean_reward,
                "val/format_reward": mean_format_reward
            }
            logging.info(f"iter: {i}  " + "  ".join(f"{k}: {v}" for k, v in log_data.items()))
            if args.wandb_project is not None:
                wandb.log(log_data, step=i)
            model.train()

        # Also save the final model!
        if i % args.checkpoint_interval == 0 or i == args.num_rollout_steps:
            os.makedirs(ckpt_save_dir, exist_ok=True)
            obj = {
                "optim": optimizer.state_dict(),
                "iter":  i,
            }
            torch.save(obj, ckpt_save_dir / "other_state.pt")

    if args.wandb_project is not None:
        wandb.finish()
