#!/bin/bash
#SBATCH --job-name=infer_shock0202
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

echo "Job running on $(hostname)"
nvidia-smi

module purge
source activate pytc

export PYTHONUNBUFFERED=1
export PYTHONPATH=/projects/weilab/weidf/lib/pytorch_connectomics/lib/MedNeXt:/home/zhangdjr/projects/umich-fiber/pytorch_connectomics:$PYTHONPATH

cd /home/zhangdjr/projects/umich-fiber/pytorch_connectomics

# Fine-tuned model (conservative, ARE=0.310 on CA1-4_2)
CKPT="outputs/fiber_linghu26_umich_finetune/20260303_133156/checkpoints/last.ckpt"
CONFIG="tutorials/fiber_linghu26_shock0202_infer.yaml"

echo "==================================="
echo "Inference on shock0202 volumes (no eval)"
echo "Checkpoint: $CKPT"
echo "Config: $CONFIG"
echo "Started: $(date)"
echo "==================================="

python -u scripts/main.py \
    --config "$CONFIG" \
    --mode test \
    --checkpoint "$CKPT"

echo "==================================="
echo "Inference finished: $(date)"
echo "==================================="
