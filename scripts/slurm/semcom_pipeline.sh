#!/bin/bash
#SBATCH --job-name=sc_pipe
#SBATCH --partition=h100base-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/sc_pipe_%A_%a.out
#SBATCH --error=logs/sc_pipe_%A_%a.err

# Two-round QA over a channel (videomind/semcom/run_pipeline.py). Outputs go to outputs_semcom/<channel>[_TAG]_<split>.
# Usage: [TAG=xxx] sbatch [--array=0-<chunk-1>] scripts/slurm/semcom_pipeline.sh <split> <channel: text|semcom> [<chunk>] [extra args...]
#   noiseless text, all train questions, grounding only:
#       sbatch --array=0-7 scripts/slurm/semcom_pipeline.sh train text 8 --no_answer
#   semantic communication at 5 dB:
#       TAG=snr5 sbatch --array=0-3 scripts/slurm/semcom_pipeline.sh test semcom 4 --ckpt work_dirs/semcom/task/task_final.pth --snr 5
#   evaluate: python videomind/eval/eval_auto.py outputs_semcom/<name> --dataset nextgqa

source ~/miniconda3/bin/activate ${VM_ENV:-videomind}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /home/jli038/code/Agent/VideoMind
export PYTHONPATH="./:$PYTHONPATH"
export PYTHONUNBUFFERED=1

split=$1
channel=$2
chunk=${3:-1}
shift 3 2>/dev/null || shift $#

pred_path="outputs_semcom/${channel}${TAG:+_$TAG}_${split}"

python videomind/semcom/run_pipeline.py \
    --split $split \
    --channel $channel \
    --pred_path $pred_path \
    --chunk $chunk \
    --index ${SLURM_ARRAY_TASK_ID:-0} \
    "$@"
