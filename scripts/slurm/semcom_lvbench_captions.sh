#!/bin/bash
#SBATCH --job-name=lv_caption
#SBATCH --partition=h100base-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/lv_caption_%A_%a.out
#SBATCH --error=logs/lv_caption_%A_%a.err

# Pre-compute LVBench coarse (60s) + fine (4s grid) captions (videomind/semcom/prepare_lvbench.py).
# Usage: sbatch --array=0-<chunk-1> scripts/slurm/semcom_lvbench_captions.sh <chunk> [extra args, e.g. --caption_batch 2]
#   then (no GPU): python videomind/semcom/prepare_lvbench.py --build_index

source ~/miniconda3/bin/activate ${VM_ENV:-videomind}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/jli038/code/Agent/VideoMind
export PYTHONPATH="./:$PYTHONPATH"
export PYTHONUNBUFFERED=1

python videomind/semcom/prepare_lvbench.py --chunk $1 --index ${SLURM_ARRAY_TASK_ID:-0} "${@:2}"
