#!/bin/bash
#SBATCH --job-name=vm_merge
#SBATCH --partition=cpu-2
#SBATCH --time=00:30:00
#SBATCH --output=logs/vm_merge_%j.out
#SBATCH --error=logs/vm_merge_%j.err

# Evaluate the merged outputs of a parallel (array) run.
# Usage: sbatch --dependency=afterok:<array_job_id> scripts/slurm/eval_merge.sh <pred_dir> <dataset>
#   e.g. sbatch --dependency=afterok:123 scripts/slurm/eval_merge.sh outputs_2b/nextgqa_test_chunk4 nextgqa

source ~/miniconda3/bin/activate videomind
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /home/jli038/code/Agent/VideoMind
export PYTHONPATH="./:$PYTHONPATH"

python videomind/eval/eval_auto.py $1 --dataset $2
