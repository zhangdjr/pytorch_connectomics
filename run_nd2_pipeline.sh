#!/bin/bash
#SBATCH --job-name=nd2_pipe
#SBATCH --output=logs/nd2_pipe_%j.out
#SBATCH --error=logs/nd2_pipe_%j.err
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhangdjr@bc.edu
# =============================================================================
# End-to-end fiber analysis pipeline: ND2 → segmentation → CSV readouts
#
# Usage:
#   sbatch run_nd2_pipeline.sh /path/to/sample.nd2 [output_base_dir]
#   bash   run_nd2_pipeline.sh /path/to/sample.nd2 [output_base_dir]
#
# This script chains 5 steps, switching conda envs as needed:
#   1. Extract all tiles + channels from ND2 file (pytc env)
#   2. Run fiber segmentation inference (pytc env, GPU)
#   3. Run micro-sam cell segmentation (microsam env, GPU)
#   4. Run fiber analysis pipeline on each tile (pytc env)
#   5. Combine per-tile CSVs into master CSV
#
# Output directory structure:
#   {output_base}/{nd2_basename}/
#     ├── tiles/                  # extracted channel TIFFs
#     ├── fiber_seg/              # fiber segmentation predictions
#     ├── cache/                  # cell seg NPZ, skeleton NPZ
#     ├── {nd2_basename}_{tile}.csv  # per-tile CSVs
#     └── {nd2_basename}_combined.csv # master CSV
# =============================================================================

# NOTE: delay strict mode until after sourcing bashrc (which may have
# unbound vars or commands that return non-zero)
source ~/.bashrc 2>/dev/null || true

set -euo pipefail

# ---- Parse arguments ----
if [ $# -lt 1 ]; then
    echo "Usage: $0 <nd2_file> [output_base_dir]"
    echo ""
    echo "  nd2_file        Path to the ND2 file"
    echo "  output_base_dir Base directory for outputs (default: /projects/weilab/dataset/barcode/2026/broad_dongqing/fiber_results)"
    exit 1
fi

ND2_FILE="$(realpath "$1")"
OUTPUT_BASE="${2:-/projects/weilab/dataset/barcode/2026/broad_dongqing/fiber_results}"

# ---- Derived paths ----
ND2_BASENAME="$(basename "$ND2_FILE" .nd2)"
OUTPUT_BASE="$(realpath -m "${OUTPUT_BASE}")"
OUTPUT_DIR="${OUTPUT_BASE}/${ND2_BASENAME}"
TILE_DIR="${OUTPUT_DIR}/tiles"
FIBER_SEG_DIR="${OUTPUT_DIR}/fiber_seg"
CACHE_DIR="${OUTPUT_DIR}/cache"

# Pipeline code directory — CHANGE THIS to where you cloned the repo
# (hardcoded because SLURM copies scripts to /var/spool/slurmd/)
SCRIPT_DIR="/home/zhangdjr/projects/umich-fiber/pytorch_connectomics"

# Model checkpoint (fixed)
CKPT="${SCRIPT_DIR}/outputs/fiber_retrain_all/20260311_223801/checkpoints/last.ckpt"
INFERENCE_YAML_TEMPLATE="${SCRIPT_DIR}/tutorials/fiber_nd2_all_tiles.yaml"

echo "============================================================"
echo "FIBER ANALYSIS PIPELINE"
echo "============================================================"
echo "ND2 file:    ${ND2_FILE}"
echo "ND2 name:    ${ND2_BASENAME}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Script dir:  ${SCRIPT_DIR}"
echo "Start time:  $(date)"
echo "============================================================"
echo ""

mkdir -p "${TILE_DIR}" "${FIBER_SEG_DIR}" "${CACHE_DIR}" logs

# ============================================================================
# STEP 1: Extract all tiles + all channels from ND2
# ============================================================================
echo "============================================================"
echo "STEP 1/5: Extracting tiles from ND2..."
echo "============================================================"

conda activate pytc
export PYTHONPATH="/projects/weilab/weidf/lib/pytorch_connectomics/lib/MedNeXt:${SCRIPT_DIR}:${PYTHONPATH:-}"

python -u "${SCRIPT_DIR}/extract_nd2_tile.py" \
    --nd2 "${ND2_FILE}" \
    --output "${TILE_DIR}" \
    --all-channels

if [ $? -ne 0 ]; then
    echo "ERROR: Tile extraction failed!"
    exit 1
fi

# ---- Auto-detect tile names from extracted files ----
TILE_NAMES=()
for f in "${TILE_DIR}"/*_ch1.tif; do
    basename_f="$(basename "$f")"
    tile_name="${basename_f%_ch1.tif}"
    TILE_NAMES+=("$tile_name")
done

N_TILES=${#TILE_NAMES[@]}
if [ "$N_TILES" -eq 0 ]; then
    echo "ERROR: No tiles extracted!"
    exit 1
fi
echo ""
echo "Detected ${N_TILES} tiles: ${TILE_NAMES[*]}"
echo ""

# ============================================================================
# STEP 2: Generate inference YAML and run fiber segmentation
# ============================================================================
echo "============================================================"
echo "STEP 2/5: Fiber segmentation inference (${N_TILES} tiles)..."
echo "============================================================"

# Generate a temporary YAML config with the correct tile paths
TEMP_YAML="${OUTPUT_DIR}/inference_config.yaml"

# Build the test_image list
TEST_IMAGE_LIST=""
for tile in "${TILE_NAMES[@]}"; do
    TEST_IMAGE_LIST="${TEST_IMAGE_LIST}    - ${TILE_DIR}/${tile}_ch1.tif
"
done

cat > "${TEMP_YAML}" << YAMLEOF
_base_: ${SCRIPT_DIR}/tutorials/bases/mednext.yaml
experiment_name: fiber_retrain_all
description: "Auto-generated inference config for ${ND2_BASENAME}"

system:
  inference:
    num_cpus: 4
    num_workers: 4
    batch_size: 1
  seed: 42

model:
  out_channels: 3
  input_size: [32, 96, 96]
  output_size: [32, 96, 96]
  mednext_size: S
  mednext_kernel_size: 3
  deep_supervision: false
  loss_functions:
  - WeightedBCEWithLogitsLoss
  - DiceLoss
  - WeightedBCEWithLogitsLoss
  - DiceLoss
  - WeightedMSELoss
  loss_weights: [4.0, 2.0, 1.0, 0.5, 2.0]
  loss_kwargs:
  - {reduction: mean, pos_weight: 20.0}
  - {sigmoid: true, smooth_nr: 1e-5, smooth_dr: 1e-5}
  - {reduction: mean}
  - {sigmoid: true, smooth_nr: 1e-5, smooth_dr: 1e-5}
  - {tanh: true}
  multi_task_config:
  - [0, 1, binary, [0, 1]]
  - [1, 2, instance_boundary, [2, 3]]
  - [2, 3, skeleton_aware_edt, [4]]

data:
  image_transform:
    clip_percentile_low: 0.005
    clip_percentile_high: 0.995

test:
  data:
    test_image:
${TEST_IMAGE_LIST}    output_path: "${FIBER_SEG_DIR}"
    image_transform:
      normalize: "0-1"
      clip_percentile_low: 0.005
      clip_percentile_high: 0.995

inference:
  sliding_window:
    window_size: [32, 256, 256]
    stride: [16, 128, 128]
    blending: gaussian
    sigma_scale: 0.25
    padding_mode: reflect
    pad_size: [16, 32, 32]
  test_time_augmentation:
    enabled: true
    channel_activations:
    - [0, 1, sigmoid]
    - [1, 2, sigmoid]
    - [2, 3, tanh]
  decoding:
  - name: decode_instance_binary_contour_distance
    kwargs:
      binary_threshold: [0.7503891044149624, 0.0046306263327246]
      contour_threshold: [0.48708101851244906, 1.025119772551816]
      distance_threshold: [-0.6695257662389609, -0.07270575159140885]
      min_instance_size: 100
      min_seed_size: 40
  save_prediction:
    output_formats:
    - tiff
  evaluation:
    enabled: false
YAMLEOF

echo "Generated inference config: ${TEMP_YAML}"
echo ""

python -u "${SCRIPT_DIR}/scripts/main.py" \
    --config "${TEMP_YAML}" \
    --mode test \
    --checkpoint "${CKPT}"

if [ $? -ne 0 ]; then
    echo "ERROR: Fiber segmentation inference failed!"
    exit 1
fi

# Clean up TTA intermediate files (large float32 caches, not needed downstream)
echo "Cleaning up inference intermediates..."
rm -f "${FIBER_SEG_DIR}"/*_tta_prediction.tiff "${FIBER_SEG_DIR}"/*_tta_prediction.h5
echo "Step 2 complete."
echo ""

# ============================================================================
# STEP 3: Cell segmentation (micro-sam, needs microsam env)
# ============================================================================
echo "============================================================"
echo "STEP 3/5: Cell segmentation (micro-sam, ${N_TILES} tiles)..."
echo "============================================================"

conda activate microsam

python -u "${SCRIPT_DIR}/cell_seg_microsam.py" \
    --tile all \
    --tile-dir "${TILE_DIR}" \
    --output-dir "${CACHE_DIR}"

if [ $? -ne 0 ]; then
    echo "ERROR: Cell segmentation failed!"
    exit 1
fi

echo "Step 3 complete."
echo ""

# ============================================================================
# STEP 4: Fiber analysis pipeline (per tile)
# ============================================================================
echo "============================================================"
echo "STEP 4/5: Fiber analysis pipeline (${N_TILES} tiles)..."
echo "============================================================"

conda activate pytc

for tile in "${TILE_NAMES[@]}"; do
    echo ""
    echo "--- Processing tile: ${tile} ---"
    python -u "${SCRIPT_DIR}/fiber_pipeline.py" \
        --tile "${tile}" \
        --nd2-name "${ND2_BASENAME}" \
        --tile-dir "${TILE_DIR}" \
        --pred-dir "${FIBER_SEG_DIR}" \
        --output-dir "${OUTPUT_BASE}" \
        --n-jobs 16
done

if [ $? -ne 0 ]; then
    echo "ERROR: Fiber analysis pipeline failed!"
    exit 1
fi

echo "Step 4 complete."
echo ""

# ============================================================================
# STEP 5: Combine per-tile CSVs into master CSV
# ============================================================================
echo "============================================================"
echo "STEP 5/5: Combining CSV files..."
echo "============================================================"

COMBINED_CSV="${OUTPUT_DIR}/${ND2_BASENAME}_combined.csv"
FIRST=true

for csv_file in "${OUTPUT_DIR}/${ND2_BASENAME}"_*.csv; do
    # Skip the combined CSV itself
    if [[ "$csv_file" == *"_combined.csv" ]]; then
        continue
    fi
    if [ "$FIRST" = true ]; then
        # Include header from first file
        cat "$csv_file" > "${COMBINED_CSV}"
        FIRST=false
    else
        # Skip header for subsequent files
        tail -n +2 "$csv_file" >> "${COMBINED_CSV}"
    fi
done

if [ "$FIRST" = true ]; then
    echo "WARNING: No per-tile CSVs found to combine!"
else
    N_ROWS=$(tail -n +2 "${COMBINED_CSV}" | wc -l)
    echo "Combined CSV: ${COMBINED_CSV} (${N_ROWS} fibers)"
fi

echo ""

# Combine per-tile intensity profile NPZs into one combined NPZ
echo "Combining intensity profile NPZs..."
python -u -c "
import numpy as np, glob, sys
files = sorted(glob.glob('${OUTPUT_DIR}/${ND2_BASENAME}_*_profiles.npz'))
if not files:
    print('  WARNING: No per-tile profile NPZs found'); sys.exit(0)
all_fids, all_valid, all_tiles = [], [], []
ch_profiles = {}
for f in files:
    d = np.load(f)
    tile = f.rsplit('_profiles.npz', 1)[0].rsplit('_', 1)[-1]
    n = len(d['fiber_ids'])
    all_fids.append(d['fiber_ids'])
    all_valid.append(d['is_valid'])
    all_tiles.extend([tile] * n)
    for k in d.files:
        if k not in ('fiber_ids', 'is_valid'):
            ch_profiles.setdefault(k, []).append(d[k])
out = '${OUTPUT_DIR}/${ND2_BASENAME}_combined_profiles.npz'
np.savez_compressed(out,
    fiber_ids=np.concatenate(all_fids),
    is_valid=np.concatenate(all_valid),
    tile_names=np.array(all_tiles),
    **{k: np.concatenate(v) for k, v in ch_profiles.items()})
print(f'  Combined profiles: {out} ({sum(len(f) for f in all_fids)} fibers from {len(files)} tiles)')
"

echo ""

# ============================================================================
# Summary
# ============================================================================
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo "ND2 file:     ${ND2_FILE}"
echo "ND2 name:     ${ND2_BASENAME}"
echo "Tiles:        ${N_TILES} (${TILE_NAMES[*]})"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Combined CSV: ${COMBINED_CSV}"
echo "Profiles:     ${OUTPUT_DIR}/${ND2_BASENAME}_combined_profiles.npz"
echo ""
echo "Cleaning up tiles to save disk space..."
rm -rf "${TILE_DIR}"
echo "  Deleted: ${TILE_DIR}"
echo "End time:     $(date)"
echo "============================================================"
