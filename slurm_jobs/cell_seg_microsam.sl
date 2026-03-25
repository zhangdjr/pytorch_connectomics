#!/bin/bash
#SBATCH --job-name=cell_seg_msam
#SBATCH --output=logs/cell_seg_msam_%j.out
#SBATCH --error=logs/cell_seg_msam_%j.err
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00

echo "============================================"
echo "Cell Segmentation (micro-sam)"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Activate microsam env
source activate microsam

# Run on all 13 tiles (skips any that are already cached)
python cell_seg_microsam.py --tile all --model_type vit_b_lm

echo ""
echo "End time: $(date)"
echo "============================================"
