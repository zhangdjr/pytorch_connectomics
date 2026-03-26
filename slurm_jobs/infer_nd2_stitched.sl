#!/bin/bash
#SBATCH --job-name=infer_nd2_stitch
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=160G         # 57 GB float32 pred + 28 GB float16 buf + decoding overhead
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liupen@bc.edu

# Full pipeline: Stitch 13 tiles → Inference on merged volume → Generate CSVs
# No tile boundary artifacts — fibers tracked continuously across entire sample.
#
# Comparison with per-tile pipeline (infer_nd2_all_tiles.sl):
#   Old: extract tiles → infer 13 separate volumes → assemble CSVs (fiber boundaries at tile edges)
#   New: stitch once   → infer 1 merged volume     → generate CSVs (no fiber boundaries)

echo "============================================"
echo "Fiber Segmentation - Full Stitched Volume"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"
echo ""

source ~/.bashrc
conda activate pytc

export PYTHONPATH=/projects/weilab/weidf/lib/pytorch_connectomics/lib/MedNeXt:/home/zhangdjr/projects/umich-fiber/pytorch_connectomics:$PYTHONPATH

WORK_DIR="/projects/weilab/liupeng/projects/umich-fiber/pytorch_connectomics"
ND2="/projects/weilab/dataset/barcode/2026/umich/A1-2003.nd2"
STITCH_DIR="/projects/weilab/dataset/barcode/2026/umich"
CONFIG="tutorials/fiber_nd2_stitched.yaml"
CKPT="checkpoints/last.ckpt"

cd $WORK_DIR
mkdir -p logs

# =============================================
# Step 1: Stitch all 13 tiles into one H5
# =============================================
# Skip if already stitched (idempotent)
STITCHED_H5="${STITCH_DIR}/nd2_stitched.h5"

if [ -f "$STITCHED_H5" ]; then
    echo "Step 1: Stitched volume already exists, skipping."
    echo "  $STITCHED_H5"
else
    echo "Step 1: Stitching 13 ND2 tiles into single volume..."
    echo ""

    python -u stitch_nd2_to_h5.py \
        --nd2 "$ND2" \
        --output_dir "$STITCH_DIR" \
        --channel 1

    if [ $? -ne 0 ]; then
        echo "ERROR: Stitching failed!"
        exit 1
    fi
fi

echo ""
echo "============================================"
echo "Step 1 complete. Shape: 56 x ~10643 x ~8607"
echo "Starting inference..."
echo "============================================"
echo ""

# =============================================
# Step 2: Strip-based sliding-window inference
#   Splits volume into X-strips of 2000 px to avoid 57 GB GPU OOM.
#   Predictions accumulated on CPU (float16), blended, decoded once.
# =============================================
echo "Step 2: Running strip-based inference on stitched volume..."
echo ""
echo "Config:     $CONFIG"
echo "Checkpoint: $CKPT"
echo ""

RESULTS_DIR="outputs/fiber_nd2_stitched/results"

python -u infer_stitched_strips.py \
    --stitched "${STITCH_DIR}/nd2_stitched.h5" \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --output_dir "$RESULTS_DIR" \
    --strip_width 2000 \
    --overlap 300

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
# Step 3: Generate fiber coordinate CSVs
#   --meta_dir points to the JSON files saved by stitch_nd2_to_h5.py
# =============================================
echo "Step 3: Generating fiber coordinate CSVs..."
echo ""

python -u generate_fiber_coordinates.py \
    --pred_dir "$RESULTS_DIR" \
    --meta_dir "${STITCH_DIR}/tile_metadata" \
    --output_dir fiber_analysis/nd2_stitched

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
echo "  Stitched volume: ${STITCHED_H5}"
echo "  Tile metadata:   ${STITCH_DIR}/tile_metadata/"
echo "  Predictions:     outputs/fiber_nd2_stitched/<timestamp>/results/"
echo "  Fiber CSVs:      fiber_analysis/nd2_stitched/"
echo "  Master CSV:      fiber_analysis/nd2_stitched/all_tiles_fiber_coordinates.csv"
