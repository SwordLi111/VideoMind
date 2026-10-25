#!/bin/bash
#SBATCH --job-name=vm_ablate
#SBATCH --partition=h100base-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/vm_ablate_%A_%a.out
#SBATCH --error=logs/vm_ablate_%A_%a.err

# Ablate the moment watched by the VideoMind answerer on NExT-GQA causal questions
# (videomind/eval/infer_answer_ablation.py). Outputs go to outputs_semcom/vm_<mode>_min<min_len>.
# Usage: sbatch --array=0-3 scripts/slurm/ablate_videomind.sh <mode: reuse|none|random|anti|oracle|other> <min_len: 32|0>
#   evaluate: python videomind/eval/eval_auto.py outputs_semcom/vm_<mode>_min<min_len> --dataset nextgqa

source ~/miniconda3/bin/activate videomind
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /home/jli038/code/Agent/VideoMind
export PYTHONPATH="./:$PYTHONPATH"
export PYTHONUNBUFFERED=1

python videomind/eval/infer_answer_ablation.py \
    --ground_mode $1 \
    --min_len $2 \
    --pred_path outputs_semcom/vm_$1_min$2 \
    --chunk 4 \
    --index ${SLURM_ARRAY_TASK_ID:-0}
