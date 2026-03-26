#!/bin/bash
#SBATCH --job-name=infer_nd2_all
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=11:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liupen@bc.edu

# Full pipeline: Extract all 13 tiles from ND2 → Run inference on all tiles

echo "============================================"
echo "Fiber Segmentation - All 13 ND2 Tiles"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

source ~/.bashrc
conda activate pytc

export PYTHONPATH=/projects/weilab/weidf/lib/pytorch_connectomics/lib/MedNeXt:/home/zhangdjr/projects/umich-fiber/pytorch_connectomics:$PYTHONPATH

WORK_DIR="/projects/weilab/liupeng/projects/umich-fiber/pytorch_connectomics"
CONFIG="tutorials/fiber_nd2_all_tiles.yaml"
CKPT="outputs/fiber_retrain_all/20260311_223801/checkpoints/last.ckpt"

cd $WORK_DIR

# =============================================
# Step 1: Extract all 13 tiles from ND2
# =============================================
echo "Step 1: Extracting all 13 tiles from ND2..."
echo ""

python -u extract_nd2_tile.py

if [ $? -ne 0 ]; then
    echo "ERROR: Extraction failed!"
    exit 1
fi

echo ""
echo "============================================"
echo "Step 1 complete. Starting inference..."
echo "============================================"
echo ""

# =============================================
# Step 2: Run inference on all 13 tiles
# =============================================
echo "Step 2: Running inference on all tiles..."
echo ""
echo "Configuration: $CONFIG"
echo "Checkpoint: $CKPT"
echo ""

python -u scripts/main.py \
    --config "$CONFIG" \
    --mode test \
    --checkpoint "$CKPT"

if [ $? -ne 0 ]; then
    echo "ERROR: Inference failed!"
    exit 1
fi

echo ""
echo "============================================"
echo "Step 2 complete. Starting post-processing..."
echo "============================================"
echo ""

# =============================================
# Step 3: Post-process (fix z-padding, generate CSVs)
# =============================================
echo "Step 3: Generating fiber coordinate CSVs..."
echo ""

python -u generate_fiber_coordinates.py \
    --pred_dir outputs/fiber_retrain_all/20260311_223801/results \
    --meta_dir /projects/weilab/dataset/barcode/2026/umich/nd2_tiles \
    --output_dir fiber_analysis/nd2_all_tiles

if [ $? -ne 0 ]; then
    echo "ERROR: Post-processing failed!"
    exit 1
fi

echo ""
echo "============================================"
echo "ALL 3 STEPS COMPLETE!"
echo "End time: $(date)"
echo "============================================"
echo ""
echo "Outputs:"
echo "  Tiles: /projects/weilab/dataset/barcode/2026/umich/nd2_tiles/"
echo "  Predictions: outputs/fiber_retrain_all/20260311_223801/results/"
echo "  CSVs: fiber_analysis/nd2_all_tiles/"
echo "  Master CSV: fiber_analysis/nd2_all_tiles/all_tiles_fiber_coordinates.csv"
