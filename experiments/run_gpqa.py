"""
Power sampling on GPQA Diamond using mh-llm (vLLM backend).

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
from constants import GPQA_QUERY_TEMPLATE

MODEL_MAP = {
    "qwen_math": "Qwen/Qwen2.5-Math-7B",
    "qwen":      "Qwen/Qwen2.5-7B",
    "phi":       "microsoft/Phi-3.5-mini-instruct",
    "tulu":      "allenai/Llama-3.1-Tulu-3-8B-DPO",
}


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
                        help="Shard index (33 problems per shard)")
    parser.add_argument("--seed",        default=0,    type=int)
    parser.add_argument("--data",        default="data/GPQA.jsonl")
    parser.add_argument("--save_dir",    default="results/gpqa")
    args = parser.parse_args()

    random.seed(args.seed)

    model_str = MODEL_MAP[args.model]
    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.data, encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    start = 33 * args.batch_idx
    end   = 33 * (args.batch_idx + 1)
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

    prompts, answers, questions = [], [], []
    for data in subset:
        choices = [
            data["Incorrect Answer 1"],
            data["Incorrect Answer 2"],
            data["Incorrect Answer 3"],
        ]
        random.shuffle(choices)
        gold_index = random.randint(0, 3)
        choices.insert(gold_index, data["Correct Answer"])
        prompt = GPQA_QUERY_TEMPLATE.format(
            Question=data["Question"],
            A=choices[0], B=choices[1], C=choices[2], D=choices[3],
        )
        prompts.append(prompt)
        answers.append("ABCD"[gold_index])
        questions.append(prompt)

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
        f"{args.model}_gpqa_mh_"
        f"alpha{args.alpha}_temp{args.temperature}_"
        f"steps{args.mcmc_steps}_"
        f"shard{args.batch_idx}_seed{args.seed}.csv"
    )
    out_path = os.path.join(args.save_dir, fname)
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
