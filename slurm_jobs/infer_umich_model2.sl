#!/bin/bash
#SBATCH --job-name=umich_model2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=g008
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhangdjr@bc.edu

echo "Job running on $(hostname)"
nvidia-smi

module purge
source activate pytc

export PYTHONUNBUFFERED=1
export PYTHONPATH=/projects/weilab/weidf/lib/pytorch_connectomics/lib/MedNeXt:/home/zhangdjr/projects/umich-fiber/pytorch_connectomics:$PYTHONPATH

cd /home/zhangdjr/projects/umich-fiber/pytorch_connectomics

CHECKPOINT="/projects/weilab/public/fiber_model_dw.ckpt"

echo "==================================="
echo "Model 2: fiber_model_dw on UMich data"
echo "Checkpoint: $CHECKPOINT"
echo "Config: tutorials/fiber_linghu26_wei.yaml"
echo "Started: $(date)"
echo "==================================="

python -u scripts/main.py \
    --config tutorials/fiber_linghu26_wei.yaml \
    --mode test \
    --checkpoint "$CHECKPOINT"

echo "==================================="
echo "Inference finished: $(date)"
echo "==================================="
