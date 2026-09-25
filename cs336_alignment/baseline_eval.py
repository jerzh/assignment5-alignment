from __future__ import annotations

import argparse
import collections
import json
import logging
import random

import torch

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.vllm_utils import VLLMServer, VLLMCompletion


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Transformer LM.")

    # ---- benchmark parameters ----
    p.add_argument("--n-examples", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)

    # ---- logging parameters ----
    p.add_argument("--completion-display-len", type=int, default=200)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open("data/gsm8k/train.jsonl", "r") as f_train, open("data/gsm8k/test.jsonl", "r") as f_test:
        test_data = random.sample([json.loads(line) for line in f_test.readlines()], args.n_examples)

    server = VLLMServer(
        model_id="allenai/OLMo-2-0425-1B",
        seed=args.seed,
        gpu=1,
    )
    server.start()
    server.init_weight_sync("cuda:0")

    for mode in ["question_only", "r1_zero", "r1_zero_three_shot"]:
        logging.info(f"begin eval for mode: {mode}")
        sampling_params = {
            "temperature": 1,
            "max_tokens": 512,
            "n": 1,
            "seed": args.seed,
        }
        if mode == "question_only":
            prompt_file = "question_only"
            reward_fn = question_only_reward_fn
        elif mode == "r1_zero":
            prompt_file = "r1_zero"
            reward_fn = r1_zero_reward_fn
            sampling_params['stop'] = ["</answer>"]
            sampling_params['include_stop_str_in_output'] = True
        else:
            prompt_file = "r1_zero_three_shot_gsm8k"
            reward_fn = r1_zero_reward_fn
            sampling_params['stop'] = ["</answer>"]
            sampling_params['include_stop_str_in_output'] = True

        with open(f"cs336_alignment/prompts/{prompt_file}.prompt", "r") as f:
            prompt = f.read()
        prompts = [prompt.format(question=obj["question"]) for obj in test_data]
        completions = [c.text for c in server.generate_completions(
            prompts=prompts,
            sampling_params=sampling_params,
        )]
        reward_total = collections.Counter()
        for qa_pair, completion in zip(test_data, completions):
            answer = qa_pair["answer"].split("####")[1].strip()
            if len(completion) <= args.completion_display_len:
                logging.info(f"completion: {completion}")
            else:
                logging.info(f"completion: {completion[:args.completion_display_len//2]}")
                logging.info(f"...")
                logging.info(completion[-args.completion_display_len//2:])
            rewards = reward_fn(completion, answer)
            logging.info(f"answer: {answer}")
            logging.info(f"format_reward: {rewards['format_reward']}  answer_reward: {rewards['answer_reward']}\n")
            reward_total.update(rewards)

        logging.info("======================== FINAL RESULT ========================")
        logging.info(f"format_reward: {reward_total['format_reward'] / len(test_data)}  answer_reward: {reward_total['answer_reward'] / len(test_data)}")
        logging.info("==============================================================\n")
