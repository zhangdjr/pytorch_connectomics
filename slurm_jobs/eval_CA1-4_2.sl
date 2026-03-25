#!/bin/bash
#SBATCH --job-name=eval_CA1-4_2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00

echo "============================================"
echo "Re-evaluate CA1-4_2 with Optuna best params"
echo "Started: $(date)"
echo "============================================"

echo "Job running on $(hostname)"
nvidia-smi

module purge
source activate pytc

export PYTHONUNBUFFERED=1
export PYTHONPATH=/projects/weilab/weidf/lib/pytorch_connectomics/lib/MedNeXt:/home/zhangdjr/projects/umich-fiber/pytorch_connectomics:$PYTHONPATH

cd /home/zhangdjr/projects/umich-fiber/pytorch_connectomics

CKPT="outputs/fiber_retrain_all/20260311_223801/checkpoints/last.ckpt"
CONFIG="tutorials/fiber_retrain_all.yaml"

echo "Checkpoint: $CKPT"
echo "Config: $CONFIG"
echo ""

python -u scripts/main.py \
    --config "$CONFIG" \
    --mode test \
    --checkpoint "$CKPT"

EXIT_CODE=$?

echo ""
echo "============================================"
echo "Evaluation finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation completed successfully"
    echo ""
    echo "Check ARE score above"
else
    echo "❌ Evaluation failed"
fi
