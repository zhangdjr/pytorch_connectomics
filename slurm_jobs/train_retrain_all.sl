#!/bin/bash
#SBATCH --job-name=retrain_all
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --partition=weilab
#SBATCH --gres=gpu:a10:4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhangdjr@bc.edu

echo "============================================"
echo "Retrain from scratch: PT37 (10) + UMich (2)"
echo "Optuna tune + eval on held-out CA1-4_2"
echo "Started: $(date)"
echo "============================================"

echo "Job running on $(hostname)"
nvidia-smi

module purge
source activate pytc

export PYTHONUNBUFFERED=1
export PYTHONPATH=/projects/weilab/weidf/lib/pytorch_connectomics/lib/MedNeXt:/home/zhangdjr/projects/umich-fiber/pytorch_connectomics:$PYTHONPATH

cd /home/zhangdjr/projects/umich-fiber/pytorch_connectomics

CONFIG="tutorials/fiber_retrain_all.yaml"

# ===========================================================
# STEP 1: Train from scratch (DDP, 4 GPUs)
# ===========================================================
echo ""
echo "######################################################################"
echo "# STEP 1/3: Training (800 epochs, 4x A10 DDP)"
echo "######################################################################"
echo "Config: $CONFIG"
echo "Started: $(date)"

srun python -u scripts/main.py --config "$CONFIG"

TRAIN_EXIT=$?
echo "Training finished: $(date)"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "❌ Training FAILED with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi
echo "✅ Training completed successfully"

# Find the latest checkpoint
CKPT=$(find outputs/fiber_retrain_all/ -name "last.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

if [ -z "$CKPT" ]; then
    echo "❌ Could not find checkpoint in outputs/fiber_retrain_all/"
    exit 1
fi
echo "Found checkpoint: $CKPT"

# ===========================================================
# STEP 2: Optuna tune decoding params on CA1-4_2 (single GPU)
# ===========================================================
echo ""
echo "######################################################################"
echo "# STEP 2/3: Optuna tuning on held-out CA1-4_2"
echo "######################################################################"
echo "Checkpoint: $CKPT"
echo "Started: $(date)"

python -u scripts/main.py \
    --config "$CONFIG" \
    --mode tune \
    --checkpoint "$CKPT"

TUNE_EXIT=$?
echo "Optuna tuning finished: $(date)"

if [ $TUNE_EXIT -ne 0 ]; then
    echo "❌ Optuna tuning FAILED with exit code $TUNE_EXIT"
    exit $TUNE_EXIT
fi
echo "✅ Optuna tuning completed successfully"

# ===========================================================
# STEP 3: Evaluate on CA1-4_2 with best params (single GPU)
# ===========================================================
echo ""
echo "######################################################################"
echo "# STEP 3/3: Evaluate on CA1-4_2 with optimized decoding"
echo "######################################################################"
echo "Checkpoint: $CKPT"
echo "Started: $(date)"

python -u scripts/main.py \
    --config "$CONFIG" \
    --mode tune-test \
    --checkpoint "$CKPT"

TEST_EXIT=$?
echo "Evaluation finished: $(date)"

if [ $TEST_EXIT -ne 0 ]; then
    echo "❌ Evaluation FAILED with exit code $TEST_EXIT"
else
    echo "✅ Evaluation completed successfully"
fi

echo ""
echo "######################################################################"
echo "# ALL STEPS COMPLETE: $(date)"
echo "# Check ARE score in the output above"
echo "######################################################################"
