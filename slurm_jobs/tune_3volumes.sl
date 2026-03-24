#!/bin/bash
#SBATCH --job-name=tune_3vol
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00

module purge
source activate pytc

export PYTHONUNBUFFERED=1
export PYTHONPATH=/projects/weilab/weidf/lib/pytorch_connectomics/lib/MedNeXt:/home/zhangdjr/projects/umich-fiber/pytorch_connectomics:$PYTHONPATH

cd /home/zhangdjr/projects/umich-fiber/pytorch_connectomics

echo "============================================"
echo "Optuna Tune-Only for 3 UMich Volumes"
echo "Using existing fine-tuned checkpoints"
echo "Started: $(date)"
echo "============================================"

# Volume names and configs
VOLUMES=("CA1-2_2" "CA1-3_2" "CA1-4_2")
CONFIGS=("tutorials/fiber_umich_CA1-2_2.yaml" "tutorials/fiber_umich_CA1-3_2.yaml" "tutorials/fiber_umich_CA1-4_2.yaml")
CHECKPOINTS=(
    "outputs/fiber_umich_CA1-2_2/20260303_230946/checkpoints/last.ckpt"
    "outputs/fiber_umich_CA1-3_2/20260303_233411/checkpoints/last.ckpt"
    "outputs/fiber_umich_CA1-4_2/20260304_000026/checkpoints/last.ckpt"
)

for i in "${!VOLUMES[@]}"; do
    VOL="${VOLUMES[$i]}"
    CONFIG="${CONFIGS[$i]}"
    CKPT="${CHECKPOINTS[$i]}"

    echo ""
    echo "######################################################################"
    echo "# VOLUME $((i+1))/3: $VOL"
    echo "######################################################################"

    echo "==================================="
    echo "Optuna tune on $VOL"
    echo "Checkpoint: $CKPT"
    echo "Config: $CONFIG"
    echo "Started: $(date)"
    echo "==================================="

    python -u scripts/main.py \
        --config "$CONFIG" \
        --mode tune \
        --checkpoint "$CKPT"

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
