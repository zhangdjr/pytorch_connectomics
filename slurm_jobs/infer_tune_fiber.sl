#!/bin/bash
#SBATCH --job-name=fiber_tune            # Job name
#SBATCH --output=logs/%x_%j.out          # Output file
#SBATCH --error=logs/%x_%j.err           # Error file
#SBATCH --nodes=1
#SBATCH --ntasks=1                       # Single task (no DDP for inference)
#SBATCH --cpus-per-task=8
#SBATCH --partition=short                # A100 partition, 12h limit
#SBATCH --gres=gpu:a100:1               # 1x A100 for inference
#SBATCH --exclude=g008
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhangdjr@bc.edu

echo "Job running on $(hostname)"
nvidia-smi

module purge
source activate pytc

export PYTHONUNBUFFERED=1
export PYTHONPATH=/projects/weilab/weidf/lib/pytorch_connectomics/lib/MedNeXt:/home/zhangdjr/projects/umich-fiber/pytorch_connectomics:$PYTHONPATH

cd /home/zhangdjr/projects/umich-fiber/pytorch_connectomics

CHECKPOINT="/projects/weilab/zhangdjr/umich-fiber/pytorch_connectomics/outputs/fiber_linghu26/20260220_122809/checkpoints/last.ckpt"

echo "==================================="
echo "Step 1: Inference on PT37_round2 (10 volumes) with Michael's thresholds"
echo "Started: $(date)"
echo "==================================="

python -u scripts/main.py \
    --config tutorials/fiber_linghu26.yaml \
    --mode test \
    --checkpoint "$CHECKPOINT"

echo "==================================="
echo "Inference finished: $(date)"
echo "==================================="

echo "==================================="
echo "Step 2: Optuna parameter sweep (100 trials)"
echo "Started: $(date)"
echo "==================================="

python -u scripts/main.py \
    --config tutorials/fiber_linghu26.yaml \
    --mode tune \
    --checkpoint "$CHECKPOINT"

echo "==================================="
echo "Optuna sweep finished: $(date)"
echo "==================================="
