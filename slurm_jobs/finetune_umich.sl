#!/bin/bash
#SBATCH --job-name=finetune_umich
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
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

ORIGINAL_CHECKPOINT="/projects/weilab/zhangdjr/umich-fiber/pytorch_connectomics/outputs/fiber_linghu26/20260220_122809/checkpoints/last.ckpt"
CONFIG="tutorials/fiber_linghu26_umich_finetune.yaml"

echo "==================================="
echo "Step 1: Fine-tune on UMich CA1-3_2"
echo "Original checkpoint: $ORIGINAL_CHECKPOINT"
echo "Config: $CONFIG"
echo "Started: $(date)"
echo "==================================="

python -u scripts/main.py \
    --config "$CONFIG" \
    --mode train \
    --checkpoint "$ORIGINAL_CHECKPOINT" \
    --external-prefix "model.model."

TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "❌ Fine-tuning failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

echo "==================================="
echo "Fine-tuning finished: $(date)"
echo "==================================="

# Find the latest fine-tuned checkpoint
FINETUNE_CKPT=$(find outputs/fiber_linghu26_umich_finetune/ -name "last.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

if [ -z "$FINETUNE_CKPT" ]; then
    echo "❌ Could not find fine-tuned checkpoint in outputs/fiber_linghu26_umich_finetune/"
    exit 1
fi

# Delete old tuning results so Optuna re-runs for this new model
rm -rf outputs/fiber_linghu26_umich_finetune/*/tuning/

echo "==================================="
echo "Step 2: Optuna tune + test on UMich CA1-4_2"
echo "Fine-tuned checkpoint: $FINETUNE_CKPT"
echo "Started: $(date)"
echo "==================================="

python -u scripts/main.py \
    --config "$CONFIG" \
    --mode tune-test \
    --checkpoint "$FINETUNE_CKPT"

echo "==================================="
echo "Tune + test finished: $(date)"
echo "==================================="
