#!/bin/bash
#SBATCH --job-name=vm_smoke
#SBATCH --partition=h100base-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/vm_smoke_%j.out
#SBATCH --error=logs/vm_smoke_%j.err

# Smoke test: run VideoMind-2B on ~1% of Charades-STA test (chunk 0 of 100) on one GPU.
# NOTE: torch 2.4 does not support Blackwell (RTX Pro 6000 / B200), use h100base-8 / h100flex-1 / h100flex-2 (h100flex-4/8, h200flex-8 need restricted_gpu QOS).
# Usage: sbatch scripts/slurm/smoke_charades_2b.sh

source ~/miniconda3/bin/activate videomind
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /home/jli038/code/Agent/VideoMind
export PYTHONPATH="./:$PYTHONPATH"

nvidia-smi

pred_path="outputs_2b/smoke_charades_sta_test"
rm -rf $pred_path

python videomind/eval/infer_auto.py \
    --dataset charades_sta \
    --split test \
    --pred_path $pred_path \
    --model_gnd_path model_zoo/VideoMind-2B \
    --model_ver_path model_zoo/VideoMind-2B \
    --model_pla_path model_zoo/VideoMind-2B \
    --chunk 100 \
    --index 0 || exit 1

python videomind/eval/eval_auto.py $pred_path --dataset charades_sta
