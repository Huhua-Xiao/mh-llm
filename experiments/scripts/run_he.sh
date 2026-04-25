#!/bin/bash
#SBATCH --job-name=mh_he
#SBATCH -t 0-23:59
#SBATCH --mem=200000
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --array=0-31          # 4 shards × 8 seeds = 32 tasks

NUM_SHARDS=4
NUM_SEEDS=8
SEED=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))
BATCH_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))

module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01

export HF_HOME={HUGGING_FACE_HOME}
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/models"
export HF_TOKEN={HF_TOKEN}

source activate psamp
cd /path/to/mh-llm/experiments

echo "shard=${BATCH_IDX} seed=${SEED} (task ${SLURM_ARRAY_TASK_ID})"
python run_he.py \
  --model=qwen_math \
  --temperature=0.25 \
  --alpha=4.0 \
  --mcmc_steps=10 \
  --batch_idx="${BATCH_IDX}" \
  --seed="${SEED}" \
  --save_dir=results/he
