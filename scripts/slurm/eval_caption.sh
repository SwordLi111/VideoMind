#!/bin/bash
#SBATCH --job-name=vm_caption
#SBATCH --partition=h100base-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/vm_caption_%j.out
#SBATCH --error=logs/vm_caption_%j.err

# Coarse-to-fine caption-based QA (videomind/eval/infer_caption.py) with one GPU.
# Usage: sbatch scripts/slurm/eval_caption.sh <dataset> <answer_mode: text|visual> [<split>] [<chunk> <index>] [extra args...]
#   smoke:    sbatch scripts/slurm/eval_caption.sh nextgqa text test 100 0
#   ablation: TAG=moment sbatch scripts/slurm/eval_caption.sh nextgqa text test 100 0 --score_mode moment
#   RTX Pro 6000 (needs the newer torch in /scratch/jli038/envs/videomind-new):
#             VM_ENV=/scratch/jli038/envs/videomind-new sbatch --partition=rtxp6000flex-1 scripts/slurm/eval_caption.sh nextgqa text test 100 0
#   parallel: sbatch --array=0-3 scripts/slurm/eval_caption.sh nextgqa text test 4
#             then: sbatch --dependency=afterok:<id> scripts/slurm/eval_merge.sh outputs_2b/caption_text_nextgqa_test_chunk4 nextgqa

source ~/miniconda3/bin/activate ${VM_ENV:-videomind}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /home/jli038/code/Agent/VideoMind
export PYTHONPATH="./:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# positional args stop at the first option (e.g. --score_mode), everything after is passed to infer_caption.py
pos=()
while [ $# -gt 0 ] && [[ "$1" != -* ]]; do
    pos+=("$1")
    shift
done

dataset=${pos[0]}
mode=${pos[1]}
split=${pos[2]:-"test"}
chunk=${pos[3]:-1}
index=${pos[4]:-${SLURM_ARRAY_TASK_ID:-0}}

# optional TAG (env var) distinguishes experiments with different extra args
name="caption_${mode}${TAG:+_$TAG}_${dataset}"
if [ "$chunk" -gt 1 ]; then
    pred_path="outputs_2b/${name}_${split}_chunk${chunk}"
    rm -f $pred_path/output_${index}.json
else
    pred_path="outputs_2b/${name}_${split}"
    rm -rf $pred_path
fi

nvidia-smi

python videomind/eval/infer_caption.py \
    --dataset $dataset \
    --split $split \
    --answer_mode $mode \
    --pred_path $pred_path \
    --chunk $chunk \
    --index $index \
    "$@" \
    || exit 1

if [ "$chunk" -eq 1 ] || [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    python videomind/eval/eval_auto.py $pred_path --dataset $dataset
fi
