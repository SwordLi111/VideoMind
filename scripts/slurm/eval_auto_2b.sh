#!/bin/bash
#SBATCH --job-name=vm_eval
#SBATCH --partition=h100base-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/vm_eval_%j.out
#SBATCH --error=logs/vm_eval_%j.err

# Evaluate VideoMind-2B on any supported benchmark with one GPU.
# NOTE: torch 2.4 does not support Blackwell (RTX Pro 6000 / B200), use h100base-8 / h100flex-1 / h100flex-2 (h100flex-4/8, h200flex-8 need restricted_gpu QOS).
# Usage: sbatch scripts/slurm/eval_auto_2b.sh <dataset> [<split>] [<chunk> <index>]
#   full:     sbatch scripts/slurm/eval_auto_2b.sh nextgqa
#   smoke:    sbatch scripts/slurm/eval_auto_2b.sh nextgqa test 100 0   (1% of samples)
#   parallel: sbatch --array=0-3 scripts/slurm/eval_auto_2b.sh nextgqa test 4   (index = array task id)
#             then: sbatch --dependency=afterok:<id> scripts/slurm/eval_merge.sh outputs_2b/nextgqa_test_chunk4 nextgqa

source ~/miniconda3/bin/activate videomind
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /home/jli038/code/Agent/VideoMind
export PYTHONPATH="./:$PYTHONPATH"
export PYTHONUNBUFFERED=1

dataset=$1
split=${2:-"test"}
chunk=${3:-1}
index=${4:-${SLURM_ARRAY_TASK_ID:-0}}

if [ "$chunk" -gt 1 ]; then
    pred_path="outputs_2b/${dataset}_${split}_chunk${chunk}"
    rm -f $pred_path/output_${index}.json
else
    pred_path="outputs_2b/${dataset}_${split}"
    rm -rf $pred_path
fi

nvidia-smi

python videomind/eval/infer_auto.py \
    --dataset $dataset \
    --split $split \
    --pred_path $pred_path \
    --model_gnd_path model_zoo/VideoMind-2B \
    --model_ver_path model_zoo/VideoMind-2B \
    --model_pla_path model_zoo/VideoMind-2B \
    --chunk $chunk \
    --index $index \
    || exit 1

# Chunked runs only cover part of the data, evaluate them with eval_merge.sh instead
if [ "$chunk" -eq 1 ] || [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    python videomind/eval/eval_auto.py $pred_path --dataset $dataset
fi
