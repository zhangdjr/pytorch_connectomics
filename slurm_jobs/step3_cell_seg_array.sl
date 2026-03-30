#!/bin/bash
#SBATCH --job-name=nd2_cellseg
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-12
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=liupen@bc.edu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

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
WORK_DIR="${WORK_DIR:-${REPO_DIR}}"
TILE_DIR="${TILE_DIR:-/projects/weilab/dataset/barcode/2026/umich/nd2_tiles}"
POSTPROC_BASE="${POSTPROC_BASE:-${WORK_DIR}/fiber_analysis}"
ND2_ID="${ND2_ID:-unknown_nd2}"
CACHE_DIR="${CACHE_DIR:-${POSTPROC_BASE}/${ND2_ID}/cache}"
CELL_SEG_MODEL_TYPE="${CELL_SEG_MODEL_TYPE:-vit_b_lm}"

DAPI_FILE="${TILE_DIR}/${TILE_NAME}_ch0_dapi.tif"
OUTPUT_FILE="${CACHE_DIR}/${TILE_NAME}_cell_seg.npz"

mkdir -p "$CACHE_DIR"

if [ ! -f "$DAPI_FILE" ]; then
    echo "ERROR: DAPI input not found for ${TILE_NAME}: ${DAPI_FILE}"
    exit 1
fi

if [ -f "$OUTPUT_FILE" ]; then
    echo "Cell segmentation already exists for ${TILE_NAME}; skipping."
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

conda activate microsam

cd "$WORK_DIR"

echo "============================================"
echo "Step 3: Cell segmentation - Tile ${TILE_NAME}"
echo "============================================"
echo "Job ID:  $SLURM_JOB_ID  Array: $SLURM_ARRAY_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "Start:   $(date)"
echo "Tile:    ${TILE_NAME}"
echo "DAPI:    ${DAPI_FILE}"
echo "Output:  ${OUTPUT_FILE}"
echo "Model:   ${CELL_SEG_MODEL_TYPE}"
echo ""

python -u tools/cell_seg_microsam.py \
    --tile "${TILE_NAME}" \
    --tile-dir "${TILE_DIR}" \
    --output-dir "${CACHE_DIR}" \
    --model_type "${CELL_SEG_MODEL_TYPE}"

if [ ! -f "$OUTPUT_FILE" ]; then
    echo "ERROR: expected output missing: ${OUTPUT_FILE}"
    exit 1
fi

echo "Done: $(date)"
echo "Cell segmentation complete for ${TILE_NAME}"
