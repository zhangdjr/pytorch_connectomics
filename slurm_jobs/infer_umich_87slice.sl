#!/bin/bash
#SBATCH --job-name=infer_87sl
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhangdjr@bc.edu

echo "============================================"
echo "Inference on full 87-slice UMich volumes"
echo "Using per-volume fine-tuned models (LR=1e-5)"
echo "with Optuna-optimized decoding parameters"
echo "Started: $(date)"
echo "============================================"

module purge
source activate pytc

export PYTHONUNBUFFERED=1
export PYTHONPATH=/projects/weilab/weidf/lib/pytorch_connectomics/lib/MedNeXt:/home/zhangdjr/projects/umich-fiber/pytorch_connectomics:$PYTHONPATH

cd /home/zhangdjr/projects/umich-fiber/pytorch_connectomics

# Per-volume checkpoints (LR=1e-5 fine-tuned)
CONFIGS=(
    "tutorials/fiber_umich_CA1-2_2_infer.yaml"
    "tutorials/fiber_umich_CA1-3_2_infer.yaml"
    "tutorials/fiber_umich_CA1-4_2_infer.yaml"
)
CHECKPOINTS=(
    "outputs/fiber_umich_CA1-2_2/20260303_230946/checkpoints/last.ckpt"
    "outputs/fiber_umich_CA1-3_2/20260303_233411/checkpoints/last.ckpt"
    "outputs/fiber_umich_CA1-4_2/20260304_000026/checkpoints/last.ckpt"
)
NAMES=("CA1-1-1-2" "CA1-1-1-3" "CA1-1-1-4")

for i in 0 1 2; do
    echo ""
    echo "######################################################################"
    echo "# VOLUME $((i+1))/3: ${NAMES[$i]}"
    echo "######################################################################"
    echo "==================================="
    echo "Inference on ${NAMES[$i]} (87 slices)"
    echo "Checkpoint: ${CHECKPOINTS[$i]}"
    echo "Config: ${CONFIGS[$i]}"
    echo "Started: $(date)"
    echo "==================================="

    python -u scripts/main.py \
        --config "${CONFIGS[$i]}" \
        --mode test \
        --checkpoint "${CHECKPOINTS[$i]}"

    echo "✅ ${NAMES[$i]} finished: $(date)"
done

echo ""
echo "######################################################################"
echo "# ALL 3 VOLUMES COMPLETE: $(date)"
echo "######################################################################"
