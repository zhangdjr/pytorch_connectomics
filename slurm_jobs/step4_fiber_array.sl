#!/bin/bash
#SBATCH --job-name=nd2_fiber
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=short
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-12
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=liupen@bc.edu

set -euo pipefail

TILE_NAMES_FILE="${TILE_NAMES_FILE:-}"
if [ -n "$TILE_NAMES_FILE" ] && [ -f "$TILE_NAMES_FILE" ]; then
    mapfile -t TILE_NAMES < "$TILE_NAMES_FILE"
else
    TILE_NAMES=(A1 A2 A3 B4 B3 B2 B1 C1 C2 C3 D2 D1 E1)
fi
TOTAL_TASKS="${#TILE_NAMES[@]}"

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL_TASKS}" ]; then
    echo "ERROR: array task id ${SLURM_ARRAY_TASK_ID} out of range for ${TOTAL_TASKS} tiles"
    exit 1
fi

TILE_NAME="${TILE_NAMES[$SLURM_ARRAY_TASK_ID]}"
WORK_DIR="${WORK_DIR:-/projects/weilab/liupeng/projects/umich-fiber/pytorch_connectomics}"
ND2_ID="${ND2_ID:-unknown_nd2}"
TILE_DIR="${TILE_DIR:-/projects/weilab/dataset/barcode/2026/umich/nd2_tiles}"
PRED_DIR="${PRED_DIR:-last/results}"
POSTPROC_BASE="${POSTPROC_BASE:-${WORK_DIR}/fiber_analysis}"
FIBER_N_JOBS="${FIBER_N_JOBS:-16}"

POSTPROC_ND2_DIR="${POSTPROC_BASE}/${ND2_ID}"
CACHE_DIR="${POSTPROC_ND2_DIR}/cache"
CSV_OUT="${POSTPROC_ND2_DIR}/${ND2_ID}_${TILE_NAME}.csv"
PROFILES_OUT="${POSTPROC_ND2_DIR}/${ND2_ID}_${TILE_NAME}_profiles.npz"
PRED_TIFF="${PRED_DIR}/${TILE_NAME}_ch1_prediction.tiff"
CELL_SEG="${CACHE_DIR}/${TILE_NAME}_cell_seg.npz"

mkdir -p "$POSTPROC_ND2_DIR" "$CACHE_DIR"

if [ ! -f "$PRED_TIFF" ]; then
    echo "ERROR: prediction missing for ${TILE_NAME}: ${PRED_TIFF}"
    exit 1
fi

if [ ! -f "$CELL_SEG" ]; then
    echo "ERROR: cell segmentation missing for ${TILE_NAME}: ${CELL_SEG}"
    exit 1
fi

if [ -f "$CSV_OUT" ] && [ -f "$PROFILES_OUT" ]; then
    echo "Fiber outputs already exist for ${TILE_NAME}; skipping."
    exit 0
fi

set +e +u
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/home/liupen/miniconda3/etc/profile.d/conda.sh" ]; then
    . "/home/liupen/miniconda3/etc/profile.d/conda.sh"
else
    source ~/.bashrc >/dev/null 2>&1 || true
fi
set -e -u

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda command not found on node ${SLURM_NODELIST}"
    exit 1
fi

conda activate pytc

cd "$WORK_DIR"

echo "============================================"
echo "Step 4: Fiber analysis - Tile ${TILE_NAME}"
echo "============================================"
echo "Job ID:   $SLURM_JOB_ID  Array: $SLURM_ARRAY_JOB_ID"
echo "Node:     $SLURM_NODELIST"
echo "Start:    $(date)"
echo "Tile:     ${TILE_NAME}"
echo "Pred dir: ${PRED_DIR}"
echo "Out dir:  ${POSTPROC_ND2_DIR}"
echo "n_jobs:   ${FIBER_N_JOBS}"
echo ""

python -u fiber_pipeline.py \
    --tile "${TILE_NAME}" \
    --nd2-name "${ND2_ID}" \
    --tile-dir "${TILE_DIR}" \
    --pred-dir "${PRED_DIR}" \
    --output-dir "${POSTPROC_BASE}" \
    --n-jobs "${FIBER_N_JOBS}"

if [ ! -f "$CSV_OUT" ] || [ ! -f "$PROFILES_OUT" ]; then
    echo "ERROR: missing expected outputs for ${TILE_NAME}"
    echo "  expected: ${CSV_OUT}"
    echo "  expected: ${PROFILES_OUT}"
    exit 1
fi

echo "Done: $(date)"
echo "Fiber analysis complete for ${TILE_NAME}"
