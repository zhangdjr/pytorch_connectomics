#!/bin/bash
#SBATCH --job-name=nd2_infer
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-63              # can be overridden by sbatch --array
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=liupen@bc.edu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# -- Tile index -> name + file ------------------------------------------------
TILE_NAMES_FILE="${TILE_NAMES_FILE:-}"
if [ -n "$TILE_NAMES_FILE" ] && [ -f "$TILE_NAMES_FILE" ]; then
    mapfile -t TILE_NAMES < <(sed '/^[[:space:]]*$/d' "$TILE_NAMES_FILE")
else
    TILE_NAMES=(A1 A2 A3 B4 B3 B2 B1 C1 C2 C3 D2 D1 E1)
fi
TOTAL_TASKS="${#TILE_NAMES[@]}"

if [ "${TOTAL_TASKS}" -le 0 ]; then
    echo "ERROR: No tile names available (TILE_NAMES_FILE='${TILE_NAMES_FILE}')"
    exit 1
fi

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL_TASKS}" ]; then
    echo "No tile mapped for array task ${SLURM_ARRAY_TASK_ID}; total tiles=${TOTAL_TASKS}. Exiting successfully."
    exit 0
fi

TILE_NAME="${TILE_NAMES[$SLURM_ARRAY_TASK_ID]}"
RUN_ID="${RUN_ID:-adhoc}"
ND2_ID="${ND2_ID:-unknown_nd2}"
TILE_DIR="${TILE_DIR:-/projects/weilab/dataset/barcode/2026/umich/nd2_tiles}"
PRED_DIR="${PRED_DIR:-last/results}"
TILE_FILE="${TILE_DIR}/${TILE_NAME}_ch1.tif"
WORK_DIR="${WORK_DIR:-${REPO_DIR}}"
META_DIR="${META_DIR:-${WORK_DIR}/meta}"
SKIP_EMPTY_PATCHES="${SKIP_EMPTY_PATCHES:-true}"
EMPTY_THRESHOLD="${EMPTY_THRESHOLD:-0.02}"

echo "============================================"
echo "Step 2: Inference - Tile ${TILE_NAME}  (task ${SLURM_ARRAY_TASK_ID}/$((TOTAL_TASKS - 1)))"
echo "============================================"
echo "Job ID:  $SLURM_JOB_ID  Array: $SLURM_ARRAY_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "Start:   $(date)"
echo "run_id:  ${RUN_ID}"
echo "nd2_id:  ${ND2_ID}"
echo "Tile:    $TILE_FILE"
echo "Pred dir:${PRED_DIR}"
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

export PYTHONPATH=/projects/weilab/weidf/lib/pytorch_connectomics/lib/MedNeXt:${WORK_DIR}:${PYTHONPATH:-}
# Enable Tensor Core utilization (L40S / A100)
export TORCH_MATMUL_PRECISION=high

CKPT="${CKPT:-checkpoints/last.ckpt}"
TEMPLATE="${TEMPLATE:-tutorials/fiber_nd2_single_tile.yaml}"

cd "$WORK_DIR"
mkdir -p logs
mkdir -p "$PRED_DIR"
mkdir -p "${META_DIR}/tmp"

if [ -f "${PRED_DIR}/${TILE_NAME}_ch1_prediction.h5" ] && [ -f "${PRED_DIR}/${TILE_NAME}_ch1_prediction.tiff" ]; then
    echo "Predictions already exist for ${TILE_NAME}; skipping inference."
    exit 0
fi

# -- GPU preflight -------------------------------------------------------------
echo "Running GPU preflight checks..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found on node ${SLURM_NODELIST}"
    exit 1
fi

if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi failed on node ${SLURM_NODELIST}"
    exit 1
fi

GPU_LIST="$(nvidia-smi -L || true)"
if [ -z "${GPU_LIST}" ]; then
    echo "ERROR: No GPUs reported by nvidia-smi on node ${SLURM_NODELIST}"
    exit 1
fi
echo "${GPU_LIST}"

python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("ERROR: torch.cuda.is_available() == False", file=sys.stderr)
    sys.exit(1)

count = torch.cuda.device_count()
if count < 1:
    print("ERROR: torch.cuda.device_count() < 1", file=sys.stderr)
    sys.exit(1)

print(f"torch.cuda.device_count() = {count}")
for idx in range(count):
    print(f"  cuda:{idx} -> {torch.cuda.get_device_name(idx)}")
PY

if [ ! -f "$TILE_FILE" ]; then
    echo "ERROR: Tile not found: $TILE_FILE"
    exit 1
fi
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi
if [ ! -f "$TEMPLATE" ]; then
    echo "ERROR: Template config not found: $TEMPLATE"
    exit 1
fi

# -- Generate per-tile YAML (swap test_image path) ----------------------------
TILE_YAML="${META_DIR}/tmp/fiber_nd2_tile_${ND2_ID}_${TILE_NAME}_${SLURM_JOB_ID}.yaml"
export TILE_FILE TILE_YAML TILE_NAME ND2_ID PRED_DIR TEMPLATE
python - <<'PY'
import os
from pathlib import Path

import yaml

template = Path(os.environ["TEMPLATE"]).resolve()
tile_file = os.environ["TILE_FILE"]
tile_yaml = os.environ["TILE_YAML"]
tile_name = os.environ["TILE_NAME"]
nd2_id = os.environ["ND2_ID"]
pred_dir = os.environ["PRED_DIR"]

with template.open() as f:
    cfg = yaml.safe_load(f)

# Preserve `_base_` inheritance when temp yaml is outside template dir.
base_field = cfg.get("_base_", None)
if isinstance(base_field, str):
    base_path = Path(base_field)
    if not base_path.is_absolute():
        cfg["_base_"] = str((template.parent / base_path).resolve())
elif isinstance(base_field, list):
    resolved = []
    for item in base_field:
        base_path = Path(str(item))
        if not base_path.is_absolute():
            base_path = (template.parent / base_path).resolve()
        resolved.append(str(base_path))
    cfg["_base_"] = resolved

cfg["test"]["data"]["test_image"] = [tile_file]
cfg["test"]["data"]["output_path"] = pred_dir
cfg["experiment_name"] = f"fiber_nd2_tile_{nd2_id}_{tile_name}"

with open(tile_yaml, "w") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

print(f"Generated: {tile_yaml}")
print(f"Output path: {pred_dir}")
PY

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to generate tile YAML!"
    exit 1
fi

# -- Run inference -------------------------------------------------------------
cmd=(
    python -u scripts/main.py
    --config "$TILE_YAML"
    --mode test
    --checkpoint "$CKPT"
)
if [ "$SKIP_EMPTY_PATCHES" = true ]; then
    cmd+=(--skip-empty-patches --empty-threshold "$EMPTY_THRESHOLD")
fi

"${cmd[@]}"

EXIT_CODE=$?

# Clean up temp YAML
rm -f "$TILE_YAML"

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Inference failed for tile ${TILE_NAME}!"
    exit 1
fi

if [ ! -f "${PRED_DIR}/${TILE_NAME}_ch1_prediction.h5" ] || [ ! -f "${PRED_DIR}/${TILE_NAME}_ch1_prediction.tiff" ]; then
    echo "ERROR: Missing prediction outputs for tile ${TILE_NAME} in ${PRED_DIR}"
    exit 1
fi

echo ""
echo "Done: $(date)"
echo "Tile ${TILE_NAME} inference complete."
