#!/bin/bash
#SBATCH --job-name=finetune_3vol
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
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

ORIGINAL_CKPT="/projects/weilab/zhangdjr/umich-fiber/pytorch_connectomics/outputs/fiber_linghu26/20260220_122809/checkpoints/last.ckpt"

VOLUMES=("CA1-2_2" "CA1-3_2" "CA1-4_2")
CONFIGS=("tutorials/fiber_umich_CA1-2_2.yaml" "tutorials/fiber_umich_CA1-3_2.yaml" "tutorials/fiber_umich_CA1-4_2.yaml")

for i in "${!VOLUMES[@]}"; do
    VOL="${VOLUMES[$i]}"
    CONFIG="${CONFIGS[$i]}"

    echo ""
    echo "######################################################################"
    echo "# VOLUME $((i+1))/3: $VOL"
    echo "######################################################################"

    # --- Step 1: Fine-tune ---
    echo "==================================="
    echo "Step 1: Fine-tune on $VOL"
    echo "Base checkpoint: $ORIGINAL_CKPT"
    echo "Config: $CONFIG"
    echo "Started: $(date)"
    echo "==================================="

    python -u scripts/main.py \
        --config "$CONFIG" \
        --mode train \
        --checkpoint "$ORIGINAL_CKPT" \
        --external-prefix "model.model." \
        --freeze-encoder

    TRAIN_EXIT=$?
    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "❌ Fine-tuning $VOL failed with exit code $TRAIN_EXIT"
        echo "Continuing to next volume..."
        continue
    fi

    echo "✅ Fine-tuning $VOL finished: $(date)"

    # --- Find the latest checkpoint for this volume ---
    FINETUNE_CKPT=$(find "outputs/fiber_umich_${VOL}/" -name "last.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -z "$FINETUNE_CKPT" ]; then
        echo "❌ Could not find fine-tuned checkpoint for $VOL"
        continue
    fi

    echo "Found checkpoint: $FINETUNE_CKPT"

    # --- Step 2: Optuna tune ---
    echo "==================================="
    echo "Step 2: Optuna tune on $VOL"
    echo "Checkpoint: $FINETUNE_CKPT"
    echo "Started: $(date)"
    echo "==================================="

    python -u scripts/main.py \
        --config "$CONFIG" \
        --mode tune \
        --checkpoint "$FINETUNE_CKPT"

    TUNE_EXIT=$?
    if [ $TUNE_EXIT -ne 0 ]; then
        echo "❌ Optuna tuning $VOL failed with exit code $TUNE_EXIT"
    else
        echo "✅ Optuna tuning $VOL finished: $(date)"
    fi

    echo ""
done

echo ""
echo "######################################################################"
echo "# ALL 3 VOLUMES COMPLETE: $(date)"
echo "######################################################################"
