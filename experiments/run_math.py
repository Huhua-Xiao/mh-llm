"""
Power sampling on MATH500 using mh-llm (vLLM backend).

Output: CSV with columns:
  question, correct_answer, mcmc_completion
"""
import argparse
import json
import os
import random

import pandas as pd
from tqdm import tqdm

from mh_llm import MHLLM
from mh_llm.vllm import SamplingParams
from constants import PROMPT, COT, BASE

MODEL_MAP = {
    "qwen_math": "Qwen/Qwen2.5-Math-7B",
    "qwen":      "Qwen/Qwen2.5-7B",
}


def format_prompt(question: str, cot: bool = True) -> str:
    suffix = COT if cot else BASE
    return PROMPT + question + suffix


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
                        help="Shard index (100 problems per shard)")
    parser.add_argument("--seed",        default=0,    type=int)
    parser.add_argument("--data",        default="data/MATH500.json")
    parser.add_argument("--save_dir",    default="results/math")
    parser.add_argument("--cot",         default=True, type=lambda x: x.lower() != "false")
    args = parser.parse_args()

    random.seed(args.seed)

    model_str = MODEL_MAP[args.model]
    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.data) as f:
        dataset = json.load(f)

    start = 100 * args.batch_idx
    end   = 100 * (args.batch_idx + 1)
    subset = dataset[start:end]

    # Paper constraint: alpha must equal 1/temperature.
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

    prompts = [format_prompt(d["prompt"], cot=args.cot) for d in subset]
    answers = [d["answer"] for d in subset]
    questions = [d["prompt"] for d in subset]

    completions = mh_llm.mh_sample(
        prompts,
        sampling_params=sampling_params,
        block_size=args.block_size,
        max_new_tokens=args.max_new_tokens,
        num_mcmc_steps=args.mcmc_steps,
        use_tqdm=True,
    )

    results = [
        {"question": q, "correct_answer": a, "mcmc_completion": c}
        for q, a, c in zip(questions, answers, completions)
    ]

    fname = (
        f"{args.model}_math_mh_"
        f"alpha{args.alpha}_temp{args.temperature}_"
        f"steps{args.mcmc_steps}_"
        f"shard{args.batch_idx}_seed{args.seed}.csv"
    )
    out_path = os.path.join(args.save_dir, fname)
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
