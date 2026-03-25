#!/bin/bash
#SBATCH --job-name=infer_new_vols
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=weilab
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

echo "============================================"
echo "Inference on 5 new volumes (retrained model)"
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
CONFIG="tutorials/fiber_retrain_all_infer_new.yaml"

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
else
    echo "❌ Inference failed"
fi
