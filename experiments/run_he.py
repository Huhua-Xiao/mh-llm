"""
Power sampling on HumanEval using mh-llm (vLLM backend).

Output: CSV with columns:
  task_id, question, mcmc_completion
"""
import argparse
import json
import os
import random
import re

import pandas as pd

from mh_llm import MHLLM
from mh_llm.vllm import SamplingParams

MODEL_MAP = {
    "qwen_math": "Qwen/Qwen2.5-Math-7B",
    "qwen":      "Qwen/Qwen2.5-7B",
    "phi":       "microsoft/Phi-3.5-mini-instruct",
    "tulu":      "allenai/Llama-3.1-Tulu-3-8B-DPO",
}


def make_phi_prompt(data: dict) -> str:
    signature = re.search(
        rf"def\s+({data['entry_point']}.*?):\s*\n", data["prompt"]
    ).group(1)
    description = "\n".join(
        line.strip()
        for line in re.search(
            r'(?:"""|\'\'\'')(.*?)(?:"""|\'\'\')', data["prompt"], re.DOTALL
        ).group(1).split("\n")
    )
    return (
        f"Write a Python function `{signature}` to solve the following problem:\n"
        f"{description}\n"
        f"{data['prompt']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="qwen_math",
                        choices=list(MODEL_MAP))
    parser.add_argument("--temperature", default=0.25, type=float)
    parser.add_argument("--alpha",       default=4.0,  type=float,
                        help="Power exponent. MUST equal 1/temperature to reproduce paper (0.25 -> 4.0)")
    parser.add_argument("--mcmc_steps",  default=10,   type=int)
    parser.add_argument("--block_size",  default=192,  type=int)
    parser.add_argument("--max_new_tokens", default=3072, type=int)
    parser.add_argument("--batch_idx",   default=0,    type=int,
                        help="Shard index (41 problems per shard)")
    parser.add_argument("--seed",        default=0,    type=int)
    parser.add_argument("--data",        default="data/HumanEval.jsonl")
    parser.add_argument("--save_dir",    default="results/he")
    args = parser.parse_args()

    random.seed(args.seed)

    model_str = MODEL_MAP[args.model]
    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.data, encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    start = 41 * args.batch_idx
    end   = 41 * (args.batch_idx + 1)
    subset = dataset[start:end]

    if abs(args.alpha - 1.0 / args.temperature) > 1e-6:
        raise ValueError(
            f"alpha ({args.alpha}) must equal 1/temperature ({1/args.temperature:.4f}) "
            "to reproduce paper results."
        )

    mh_llm = MHLLM(model=model_str)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        alpha=args.alpha,
    )

    prompts, task_ids, raw_prompts = [], [], []
    for data in subset:
        if args.model in ("phi", "phi_grpo"):
            prompt = make_phi_prompt(data)
        else:
            prompt = data["prompt"]
        prompts.append(prompt)
        task_ids.append(data["task_id"])
        raw_prompts.append(data["prompt"])

    completions = mh_llm.mh_sample(
        prompts,
        sampling_params=sampling_params,
        block_size=args.block_size,
        max_new_tokens=args.max_new_tokens,
        num_mcmc_steps=args.mcmc_steps,
        use_tqdm=True,
    )

    results = [
        {"task_id": tid, "question": rp, "mcmc_completion": c}
        for tid, rp, c in zip(task_ids, raw_prompts, completions)
    ]

    fname = (
        f"{args.model}_he_mh_"
        f"alpha{args.alpha}_temp{args.temperature}_"
        f"steps{args.mcmc_steps}_"
        f"shard{args.batch_idx}_seed{args.seed}.csv"
    )
    out_path = os.path.join(args.save_dir, fname)
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
