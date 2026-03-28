#!/bin/bash
#SBATCH --job-name=nd2_postproc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=short
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=liupen@bc.edu
# No GPU needed — pure CPU post-processing

set -euo pipefail

echo "============================================"
echo "Step 3: Post-processing — Generate CSVs (CPU only)"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"
echo ""

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

WORK_DIR="${WORK_DIR:-/projects/weilab/liupeng/projects/umich-fiber/pytorch_connectomics}"
RUN_ID="${RUN_ID:-adhoc}"
ND2_ID="${ND2_ID:-unknown_nd2}"
PRED_DIR="${PRED_DIR:-last/results}"
TILE_DIR="${TILE_DIR:-/projects/weilab/dataset/barcode/2026/umich/nd2_tiles}"
POSTPROC_DIR="${POSTPROC_DIR:-fiber_analysis/nd2_all_tiles}"
QC_DIR="${QC_DIR:-${WORK_DIR}/qc}"
META_DIR="${META_DIR:-${WORK_DIR}/meta}"
TILE_NAMES_FILE="${TILE_NAMES_FILE:-}"
CLEANUP_TILES="${CLEANUP_TILES:-true}"

cd "$WORK_DIR"
mkdir -p "$POSTPROC_DIR" "$QC_DIR" "$META_DIR"

if [ -n "$TILE_NAMES_FILE" ] && [ -f "$TILE_NAMES_FILE" ]; then
    mapfile -t TILE_NAMES < "$TILE_NAMES_FILE"
else
    TILE_NAMES=(A1 A2 A3 B4 B3 B2 B1 C1 C2 C3 D2 D1 E1)
fi

echo "run_id:    ${RUN_ID}"
echo "nd2_id:    ${ND2_ID}"
echo "tile_dir:  ${TILE_DIR}"
echo "pred_dir:  ${PRED_DIR}"
echo "postproc:  ${POSTPROC_DIR}"

# Verify prediction files exist for all tiles
echo "Verifying prediction files in: ${PRED_DIR}/"
MISSING=0
for TILE in "${TILE_NAMES[@]}"; do
    if [ -f "${PRED_DIR}/${TILE}_ch1_prediction.tiff" ]; then
        echo "  OK: ${TILE}_ch1_prediction.tiff"
    else
        echo "  WARNING: Missing ${TILE}_ch1_prediction.tiff"
        MISSING=$((MISSING+1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo "ERROR: ${MISSING} tile(s) missing predictions. Aborting."
    exit 1
fi

COMBINED_PRED_DIR="$PRED_DIR"   # all tiles write to the same results dir

echo ""
echo "Running generate_fiber_coordinates.py..."
python -u generate_fiber_coordinates.py \
    --pred_dir "$COMBINED_PRED_DIR" \
    --meta_dir "$TILE_DIR" \
    --output_dir "$POSTPROC_DIR"


if [ $? -ne 0 ]; then
    echo "ERROR: Post-processing failed!"
    exit 1
fi

export RUN_ID ND2_ID PRED_DIR TILE_DIR POSTPROC_DIR QC_DIR META_DIR
python - <<'PY'
import json
import os
from pathlib import Path

run_id = os.environ["RUN_ID"]
nd2_id = os.environ["ND2_ID"]
pred_dir = Path(os.environ["PRED_DIR"])
tile_dir = Path(os.environ["TILE_DIR"])
postproc_dir = Path(os.environ["POSTPROC_DIR"])
qc_dir = Path(os.environ["QC_DIR"])
meta_dir = Path(os.environ["META_DIR"])

qc_dir.mkdir(parents=True, exist_ok=True)
meta_dir.mkdir(parents=True, exist_ok=True)

report = {
    "run_id": run_id,
    "nd2_id": nd2_id,
    "pred_dir": str(pred_dir),
    "tile_dir": str(tile_dir),
    "postproc_dir": str(postproc_dir),
    "prediction_tiff_count": len(list(pred_dir.glob("*_ch1_prediction.tiff"))),
    "prediction_h5_count": len(list(pred_dir.glob("*_ch1_prediction.h5"))),
    "tile_tiff_count": len(list(tile_dir.glob("*_ch1.tif"))),
    "metadata_count": len(list(tile_dir.glob("*_metadata.json"))),
    "tile_csv_count": len(list(postproc_dir.glob("*_fiber_coordinates.csv"))),
    "master_csv_exists": (postproc_dir / "all_tiles_fiber_coordinates.csv").exists(),
}

with open(qc_dir / "check_report.json", "w") as f:
    json.dump(report, f, indent=2)

with open(meta_dir / "step3_postproc_summary.txt", "w") as f:
    for k, v in report.items():
        f.write(f"{k}={v}\n")
PY

if [ "$CLEANUP_TILES" = "true" ]; then
    if [ -n "${TILE_DIR}" ] && [ "${TILE_DIR}" != "/" ]; then
        echo ""
        echo "Cleaning up tiles to save disk space..."
        rm -rf "${TILE_DIR}"
        echo "  Deleted: ${TILE_DIR}"
    else
        echo "WARNING: skip tile cleanup due to unsafe TILE_DIR='${TILE_DIR}'"
    fi
fi

echo ""
echo "============================================"
echo "ALL DONE: $(date)"
echo "============================================"
echo ""
echo "Outputs:"
echo "  Tiles:      ${TILE_DIR}/"
echo "  Predictions:${PRED_DIR}/"
echo "  Fiber CSVs: ${POSTPROC_DIR}/"
echo "  Master CSV: ${POSTPROC_DIR}/all_tiles_fiber_coordinates.csv"
echo "  QC report:  ${QC_DIR}/check_report.json"
