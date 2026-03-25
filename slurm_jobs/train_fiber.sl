#!/bin/bash
#SBATCH --job-name=fiber_linghu26        # Job name
#SBATCH --output=logs/%x_%j.out              # Output file
#SBATCH --error=logs/%x_%j.err               # Error file
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=4             # 1 task per GPU = 4 DDP processes
#SBATCH --cpus-per-task=8               # 8 CPUs per task (32 total)
#SBATCH --partition=weilab              # dedicated lab partition, no time limit
#SBATCH --gres=gpu:a10:4               # weilab has 4x A10 (24GB each) per node
#SBATCH --mem=128G                      # Memory
#SBATCH --time=2-00:00:00              # 2 days (weilab has no limit, safe ceiling)
#SBATCH --mail-type=BEGIN,END,FAIL      # Mail events
#SBATCH --mail-user=zhangdjr@bc.edu     # Email for notifications

# Print node information
echo "Job running on $(hostname)"
echo "Available GPUs:"
nvidia-smi

# Load modules/environment
module purge

# Activate conda environment
source activate pytc

# Set environment variables
export PYTHONUNBUFFERED=1

# Fix PYTHONPATH to ensure the right connectomics module is found
export PYTHONPATH=/projects/weilab/weidf/lib/pytorch_connectomics/lib/MedNeXt:/home/zhangdjr/projects/umich-fiber/pytorch_connectomics:$PYTHONPATH

# Change to project directory
cd /home/zhangdjr/projects/umich-fiber/pytorch_connectomics

# Print versions
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

echo "==================================="
echo "Starting training: $(date)"
echo "==================================="

# Run training with srun so all 4 tasks launch as DDP workers
srun python -u scripts/main.py --config tutorials/fiber_linghu26.yaml
TRAIN_EXIT=$?

echo "==================================="
echo "Training finished at: $(date)"
echo "==================================="

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training FAILED with exit code $TRAIN_EXIT. Skipping inference."
    exit $TRAIN_EXIT
fi

# After training, find the best checkpoint (lowest loss, saved by ModelCheckpoint)
# Best checkpoint is the most recently modified non-last.ckpt file
BEST_CKPT=$(find outputs/fiber_linghu26 -name "epoch=*.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)

if [ -n "$BEST_CKPT" ]; then
    echo "Found best checkpoint: $BEST_CKPT"
    echo "==================================="
    echo "Starting inference: $(date)"
    echo "==================================="

    # Inference only needs 1 GPU — run plain python (not srun)
    python -u scripts/main.py \
        --config tutorials/fiber_linghu26.yaml \
        --mode test \
        --checkpoint "$BEST_CKPT"

    echo "==================================="
    echo "Inference finished at: $(date)"
    echo "==================================="
else
    echo "No checkpoint found under outputs/fiber_linghu26/. Skipping inference."
fi
