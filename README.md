# A Metropolis-Hastings sampler for LLMs.
Implements the sampling algorithm described in [Reasoning with Sampling: Your Base Model is Smarter Than You Think](https://arxiv.org/abs/2510.14901) in [vLLM](https://github.com/vllm-project/vllm).
This cuts the evaluation time for [Qwen/Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B) from 30hrs to <1hr on the MATH500 dataset on a B200 GPU compared to the official implementation.

This package patches the `vLLM` `LLMEngine` object and adds the `alpha` parameter to `SamplingParams` in order to sample from the power distribuion.

## Installation

```bash
pip install mh-llm
```

or from the source:
```bash
pip install git+https://github.com/maxzuo/mh-llm.git
```

This was tested with vLLM 0.11.0, it may not work with newer versions.

## Reproducing Paper Experiments

### 1. Environment Setup

Create a new conda environment (requires Python 3.11, vLLM 0.11.0):

```bash
conda create -n mh_llm python=3.11 -y
conda activate mh_llm
pip install vllm==0.11.0
pip install transformers pandas tqdm
pip install -e .
```

**Compatibility fix** — vLLM 0.11.0 uses a `transformers` API removed in `transformers>=5.0`. Patch it:

```bash
F=$(python -c "import vllm; import os; print(os.path.join(os.path.dirname(vllm.__file__), 'transformers_utils/tokenizer.py'))")
sed -i "s/tokenizer\.all_special_tokens_extended/getattr(tokenizer, 'all_special_tokens_extended', tokenizer.all_special_tokens)/g" $F
```

### 2. Data Preparation

MATH500 is included in `experiments/data/`. Download GPQA and HumanEval:

```python
# Run in Python (requires HuggingFace token with GPQA access)
from datasets import load_dataset
import json

# GPQA Diamond — request access at https://huggingface.co/datasets/Idavidrein/gpqa
ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", token="YOUR_HF_TOKEN")
with open("experiments/data/GPQA.jsonl", "w") as f:
    for row in ds["train"]:
        f.write(json.dumps(row) + "\n")

# HumanEval
ds = load_dataset("openai/openai_humaneval", token="YOUR_HF_TOKEN")
with open("experiments/data/HumanEval.jsonl", "w") as f:
    for row in ds["test"]:
        f.write(json.dumps(row) + "\n")
```

### 3. Running Experiments

All experiments use `temperature=0.25, alpha=4.0, mcmc_steps=10` (paper settings).
`alpha` must equal `1/temperature`.

**Single run (for testing):**
```bash
cd experiments
python run_math.py --model=qwen_math --batch_idx=0 --seed=0
python run_gpqa.py --model=qwen_math --batch_idx=0 --seed=0
python run_he.py   --model=qwen_math --batch_idx=0 --seed=0
```

**Full experiments — sequential (single GPU):**
```bash
cd experiments

# MATH500 — 5 shards × 8 seeds (~23 hrs on H100)
nohup bash -c '
cd /path/to/mh-llm/experiments
for batch_idx in 0 1 2 3 4; do
  for seed in 0 1 2 3 4 5 6 7; do
    python run_math.py --model=qwen_math --batch_idx=$batch_idx --seed=$seed
  done
done
' > math_run.log 2>&1 &

# GPQA Diamond — 6 shards × 8 seeds
nohup bash -c '
cd /path/to/mh-llm/experiments
for batch_idx in 0 1 2 3 4 5; do
  for seed in 0 1 2 3 4 5 6 7; do
    python run_gpqa.py --model=qwen_math --batch_idx=$batch_idx --seed=$seed
  done
done
' > gpqa_run.log 2>&1 &

# HumanEval — 4 shards × 8 seeds
nohup bash -c '
cd /path/to/mh-llm/experiments
for batch_idx in 0 1 2 3; do
  for seed in 0 1 2 3 4 5 6 7; do
    python run_he.py --model=qwen_math --batch_idx=$batch_idx --seed=$seed
  done
done
' > he_run.log 2>&1 &
```

**Full experiments — parallel (Slurm cluster):**

Fill in the placeholders in `experiments/scripts/` then:
```bash
sbatch experiments/scripts/run_math.sh   # 40 parallel jobs
sbatch experiments/scripts/run_gpqa.sh  # 48 parallel jobs
sbatch experiments/scripts/run_he.sh    # 32 parallel jobs
```

Results are saved to `experiments/results/{math,gpqa,he}/` as CSV files with columns `question`, `correct_answer`, `mcmc_completion`.

### Parameters

| Dataset | temperature | alpha | problems/shard | shards | seeds |
|---------|-------------|-------|----------------|--------|-------|
| MATH500 | 0.25 | 4.0 | 100 | 5 | 8 |
| GPQA Diamond | 0.25 | 4.0 | 33 | 6 | 8 |
| HumanEval | 0.25 | 4.0 | 41 | 4 | 8 |

## Example Usage
```python
from mh_llm import MHLLM
from mh_llm.vllm import SamplingParams

# Initialize MH LLM with your model
mh_llm = MHLLM(model='Qwen/Qwen2.5-Math-7B')
# Define sampling parameters with alpha
sampling_params = SamplingParams(temperature=0.25, alpha=0.4)

# Generate samples, without metropolis-hastings or power distribution
output = mh_llm.generate("What is 1234 + 5678?", sampling_params=sampling_params)

# Sample with Metropolis-Hastings against the power distribution
mh_output = mh_llm.mh_sample(
    "What is 1234 + 5678?",
    sampling_params=sampling_params,
    block_size=192,
    max_new_tokens=3_072,
    num_mcmc_steps=10,
    use_tqdm=True,
)
```
