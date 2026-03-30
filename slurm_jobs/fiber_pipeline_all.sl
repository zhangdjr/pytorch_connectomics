#!/bin/bash
#SBATCH --job-name=fiber_pipe
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-12
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhangdjr@bc.edu

# Tile list (13 tiles, indexed 0-12)
TILES=(A1 A2 A3 B4 B3 B2 B1 C1 C2 C3 D2 D1 E1)
TILE=${TILES[$SLURM_ARRAY_TASK_ID]}

echo "============================================"
echo "Fiber Pipeline - ${TILE} tile (A1-2003)"
echo "Array task: $SLURM_ARRAY_TASK_ID / Job: $SLURM_ARRAY_JOB_ID"
echo "============================================"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

source ~/.bashrc
conda activate pytc

export PYTHONPATH=/projects/weilab/weidf/lib/pytorch_connectomics/lib/MedNeXt:/home/zhangdjr/projects/umich-fiber/pytorch_connectomics:$PYTHONPATH

cd /home/zhangdjr/projects/umich-fiber/pytorch_connectomics

python -u tools/fiber_pipeline.py \
    --tile ${TILE} \
    --nd2-name A1-2003 \
    --n-jobs 16

echo ""
echo "End time: $(date)"
echo "============================================"
