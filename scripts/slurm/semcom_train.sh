#!/bin/bash
#SBATCH --job-name=sc_train
#SBATCH --partition=h100base-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/sc_train_%j.out
#SBATCH --error=logs/sc_train_%j.err

# Train the semantic encoder / decoder (videomind/semcom/train.py).
# Usage: sbatch scripts/slurm/semcom_train.sh <stage: recon|task> <work_dir> [extra args...]
#   sbatch scripts/slurm/semcom_train.sh recon work_dirs/semcom/recon
#   sbatch scripts/slurm/semcom_train.sh task work_dirs/semcom/task --init work_dirs/semcom/recon/recon_final.pth

source ~/miniconda3/bin/activate ${VM_ENV:-videomind}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /home/jli038/code/Agent/VideoMind
export PYTHONPATH="./:$PYTHONPATH"
export PYTHONUNBUFFERED=1

python videomind/semcom/train.py --stage $1 --work_dir $2 "${@:3}"
