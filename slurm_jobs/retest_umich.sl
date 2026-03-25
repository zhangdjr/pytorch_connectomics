#!/bin/bash
#SBATCH --job-name=retest_umich
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --exclude=g008
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhangdjr@bc.edu

echo "Job running on $(hostname)"
nvidia-smi

module purge
source activate pytc

export PYTHONUNBUFFERED=1
export PYTHONPATH=/projects/weilab/weidf/lib/pytorch_connectomics/lib/MedNeXt:/home/zhangdjr/projects/umich-fiber/pytorch_connectomics:$PYTHONPATH

cd /home/zhangdjr/projects/umich-fiber/pytorch_connectomics

FINETUNE_CKPT="outputs/fiber_linghu26_umich_finetune/20260302_212636/checkpoints/last.ckpt"
CONFIG="tutorials/fiber_linghu26_umich_finetune.yaml"

echo "==================================="
echo "Re-test with fixed Optuna param application"
echo "Checkpoint: $FINETUNE_CKPT"
echo "Config: $CONFIG"
echo "Started: $(date)"
echo "==================================="

python -u scripts/main.py \
    --config "$CONFIG" \
    --mode tune-test \
    --checkpoint "$FINETUNE_CKPT"

echo "==================================="
echo "Re-test finished: $(date)"
echo "==================================="
