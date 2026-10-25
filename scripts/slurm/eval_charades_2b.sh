#!/bin/bash
#SBATCH --job-name=vm_charades
#SBATCH --partition=h100base-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/vm_charades_%j.out
#SBATCH --error=logs/vm_charades_%j.err

# Full evaluation: VideoMind-2B on the full Charades-STA test set (3720 samples) on one GPU.
# NOTE: torch 2.4 does not support Blackwell (RTX Pro 6000 / B200), use h100base-8 / h100flex-1 / h100flex-2 (h100flex-4/8, h200flex-8 need restricted_gpu QOS).
# Usage: sbatch scripts/slurm/eval_charades_2b.sh

source ~/miniconda3/bin/activate videomind
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /home/jli038/code/Agent/VideoMind
export PYTHONPATH="./:$PYTHONPATH"

nvidia-smi

pred_path="outputs_2b/charades_sta_test"
rm -rf $pred_path

python videomind/eval/infer_auto.py \
    --dataset charades_sta \
    --split test \
    --pred_path $pred_path \
    --model_gnd_path model_zoo/VideoMind-2B \
    --model_ver_path model_zoo/VideoMind-2B \
    --model_pla_path model_zoo/VideoMind-2B \
    || exit 1

python videomind/eval/eval_auto.py $pred_path --dataset charades_sta
