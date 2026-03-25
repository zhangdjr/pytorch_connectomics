#!/bin/bash
#SBATCH --job-name=infer_multiregion
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=weilab
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhangdjr@bc.edu

echo "============================================"
echo "Inference on 4 multi-region volumes"
echo "  A1-2002, A1-2003 s01/s02, A1-2007"
echo "  All 162.9nm XY, 0.4um Z (same as training)"
echo "  No test_resolution (avoiding z-padding bug)"
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
CONFIG="tutorials/fiber_retrain_all_infer_multiregion.yaml"

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
echo "Inference finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All inference completed successfully"
    echo "Results saved to: outputs/fiber_retrain_all/20260311_223801/results/"
    echo "Run 'python view_multiregion_predictions.py' to visualize."
else
    echo "❌ Inference failed"
fi
