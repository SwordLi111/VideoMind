#!/bin/bash
#SBATCH --job-name=sc_caption
#SBATCH --partition=h100base-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/sc_caption_%A_%a.out
#SBATCH --error=logs/sc_caption_%A_%a.err

# Pre-compute coarse + 1s-grid fine captions for NExT-QA causal questions (videomind/semcom/prepare_captions.py).
# Usage: sbatch --array=0-<chunk-1> scripts/slurm/semcom_captions.sh <split: train|valid|test> <chunk> [extra args, e.g. --caption_batch 2]
#   then (no GPU): python videomind/semcom/prepare_captions.py --split <split> --build_index

source ~/miniconda3/bin/activate ${VM_ENV:-videomind}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /home/jli038/code/Agent/VideoMind
export PYTHONPATH="./:$PYTHONPATH"
export PYTHONUNBUFFERED=1

python videomind/semcom/prepare_captions.py --split $1 --chunk $2 --index ${SLURM_ARRAY_TASK_ID:-0} "${@:3}"
