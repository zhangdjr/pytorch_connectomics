#!/bin/bash
#SBATCH --job-name=nd2_merge
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=liupen@bc.edu

set -euo pipefail

WORK_DIR="${WORK_DIR:-/projects/weilab/liupeng/projects/umich-fiber/pytorch_connectomics}"
ND2_ID="${ND2_ID:-unknown_nd2}"
POSTPROC_BASE="${POSTPROC_BASE:-${WORK_DIR}/fiber_analysis}"
POSTPROC_ND2_DIR="${POSTPROC_ND2_DIR:-${POSTPROC_BASE}/${ND2_ID}}"
QC_DIR="${QC_DIR:-${WORK_DIR}/qc}"

mkdir -p "$POSTPROC_ND2_DIR" "$QC_DIR"

set +e +u
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/home/liupen/miniconda3/etc/profile.d/conda.sh" ]; then
    . "/home/liupen/miniconda3/etc/profile.d/conda.sh"
else
    source ~/.bashrc >/dev/null 2>&1 || true
fi
set -e -u

if command -v conda >/dev/null 2>&1; then
    conda activate pytc || true
fi

cd "$WORK_DIR"

echo "============================================"
echo "Step 5: Merge per-tile outputs"
echo "============================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "Start:   $(date)"
echo "ND2:     ${ND2_ID}"
echo "Source:  ${POSTPROC_ND2_DIR}"
echo ""

COMBINED_CSV="${POSTPROC_ND2_DIR}/${ND2_ID}_combined.csv"
FIRST=true
CSV_COUNT=0

for csv_file in "${POSTPROC_ND2_DIR}/${ND2_ID}"_*.csv; do
    if [ ! -f "$csv_file" ]; then
        continue
    fi
    if [[ "$csv_file" == *"_combined.csv" ]]; then
        continue
    fi

    if [ "$FIRST" = true ]; then
        cat "$csv_file" > "$COMBINED_CSV"
        FIRST=false
    else
        tail -n +2 "$csv_file" >> "$COMBINED_CSV"
    fi
    CSV_COUNT=$((CSV_COUNT + 1))
done

if [ "$FIRST" = true ]; then
    echo "WARNING: no per-tile CSV files found in ${POSTPROC_ND2_DIR}"
else
    ROWS=$(tail -n +2 "$COMBINED_CSV" | wc -l)
    echo "Combined CSV: ${COMBINED_CSV}"
    echo "  source tiles: ${CSV_COUNT}"
    echo "  rows:         ${ROWS}"
fi

COMBINED_PROFILES="${POSTPROC_ND2_DIR}/${ND2_ID}_combined_profiles.npz"

python -u - "$POSTPROC_ND2_DIR" "$ND2_ID" "$COMBINED_PROFILES" <<'PY'
import glob
import os
import sys

import numpy as np

postproc_dir, nd2_id, combined_out = sys.argv[1:4]
pattern = os.path.join(postproc_dir, f"{nd2_id}_*_profiles.npz")
files = sorted(f for f in glob.glob(pattern) if not f.endswith("_combined_profiles.npz"))

if not files:
    print("WARNING: no per-tile profile NPZ files found")
    sys.exit(0)

all_fids, all_valid, all_tiles = [], [], []
ch_profiles = {}

for fpath in files:
    data = np.load(fpath)
    tile = os.path.basename(fpath).rsplit("_profiles.npz", 1)[0].rsplit("_", 1)[-1]
    n = len(data["fiber_ids"])
    all_fids.append(data["fiber_ids"])
    all_valid.append(data["is_valid"])
    all_tiles.extend([tile] * n)

    for key in data.files:
        if key in ("fiber_ids", "is_valid"):
            continue
        ch_profiles.setdefault(key, []).append(data[key])

np.savez_compressed(
    combined_out,
    fiber_ids=np.concatenate(all_fids),
    is_valid=np.concatenate(all_valid),
    tile_names=np.array(all_tiles),
    **{k: np.concatenate(v) for k, v in ch_profiles.items()},
)

print(f"Combined profiles: {combined_out}")
print(f"  source tiles: {len(files)}")
print(f"  fibers:       {sum(len(x) for x in all_fids)}")
PY

python -u - "$COMBINED_CSV" "$COMBINED_PROFILES" "$QC_DIR" "$ND2_ID" <<'PY'
import json
import os
import sys

combined_csv, combined_profiles, qc_dir, nd2_id = sys.argv[1:5]
os.makedirs(qc_dir, exist_ok=True)

summary = {
    "nd2_id": nd2_id,
    "combined_csv": combined_csv,
    "combined_csv_exists": os.path.exists(combined_csv),
    "combined_profiles": combined_profiles,
    "combined_profiles_exists": os.path.exists(combined_profiles),
}

qc_path = os.path.join(qc_dir, f"{nd2_id}_merge_summary.json")
with open(qc_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Wrote merge summary: {qc_path}")
PY

echo ""
echo "Done: $(date)"
echo "Merge complete."
