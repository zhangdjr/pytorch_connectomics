#!/bin/bash
#SBATCH --job-name=nd2_extract
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=liupen@bc.edu
# No GPU needed - pure file I/O

set -euo pipefail

echo "============================================"
echo "Step 1: Extract ND2 Tiles (CPU only)"
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
    # Fallback for custom shells; ignore non-critical rc errors.
    source ~/.bashrc >/dev/null 2>&1 || true
fi
set -e -u

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda command not found on node ${SLURM_NODELIST}"
    exit 1
fi
conda activate pytc

WORK_DIR="${WORK_DIR:-/projects/weilab/liupeng/projects/umich-fiber/pytorch_connectomics}"
ND2_PATH="${ND2_PATH:-/projects/weilab/dataset/barcode/2026/umich/A1-2003.nd2}"
TILE_DIR="${TILE_DIR:-/projects/weilab/dataset/barcode/2026/umich/nd2_tiles}"
META_DIR="${META_DIR:-${WORK_DIR}/meta}"
RUN_ID="${RUN_ID:-adhoc}"
ND2_ID="${ND2_ID:-$(basename "${ND2_PATH%.*}")}"
TILE_NAMES_FILE="${TILE_NAMES_FILE:-}"
EXTRACT_ALL_CHANNELS="${EXTRACT_ALL_CHANNELS:-false}"

cd "$WORK_DIR"
mkdir -p "$TILE_DIR" "$META_DIR"

if [ ! -f "$ND2_PATH" ]; then
    echo "ERROR: ND2 file not found: $ND2_PATH"
    exit 1
fi

if [ -n "$TILE_NAMES_FILE" ] && [ -f "$TILE_NAMES_FILE" ]; then
    mapfile -t TILE_NAMES < "$TILE_NAMES_FILE"
else
    TILE_NAMES=(A1 A2 A3 B4 B3 B2 B1 C1 C2 C3 D2 D1 E1)
fi

required_suffixes=("ch1.tif")
if [ "$EXTRACT_ALL_CHANNELS" = true ]; then
    required_suffixes=("ch0_dapi.tif" "ch1.tif" "ch2_cfos.tif" "ch3_timestamp.tif")
fi

complete=true
for tile in "${TILE_NAMES[@]}"; do
    for suffix in "${required_suffixes[@]}"; do
        if [ ! -f "${TILE_DIR}/${tile}_${suffix}" ]; then
            complete=false
            break
        fi
    done
    if [ "$complete" = false ] || [ ! -f "${TILE_DIR}/${tile}_metadata.json" ]; then
        complete=false
        break
    fi
done

if [ "$complete" = true ]; then
    echo "Tiles already extracted for nd2_id=${ND2_ID}, skipping extraction."
    echo "  ${TILE_DIR}/"
    if [ -n "$TILE_NAMES_FILE" ]; then
        find "$TILE_DIR" -maxdepth 1 -type f -name '*_ch1.tif' -printf '%f\n' \
            | sed 's/_ch1\.tif$//' | sort -u > "$TILE_NAMES_FILE"
        echo "Wrote tile name list: ${TILE_NAMES_FILE} ($(wc -l < "$TILE_NAMES_FILE") tiles)"
    fi
    exit 0
fi

extract_cmd=(python -u extract_nd2_tile.py --nd2 "$ND2_PATH" --output "$TILE_DIR")
if [ "$EXTRACT_ALL_CHANNELS" = true ]; then
    extract_cmd+=(--all-channels)
fi
"${extract_cmd[@]}"

if [ $? -ne 0 ]; then
    echo "ERROR: Extraction failed!"
    exit 1
fi

cat > "${META_DIR}/step1_extract_summary.txt" <<EOF
run_id=${RUN_ID}
nd2_id=${ND2_ID}
nd2_path=${ND2_PATH}
tile_dir=${TILE_DIR}
extract_all_channels=${EXTRACT_ALL_CHANNELS}
done_at=$(date -Iseconds)
EOF

if [ -n "$TILE_NAMES_FILE" ]; then
    find "$TILE_DIR" -maxdepth 1 -type f -name '*_ch1.tif' -printf '%f\n' \
        | sed 's/_ch1\.tif$//' | sort -u > "$TILE_NAMES_FILE"
    echo "Wrote tile name list: ${TILE_NAMES_FILE} ($(wc -l < "$TILE_NAMES_FILE") tiles)"
fi

echo ""
echo "Done: $(date)"
echo "Tiles saved to: ${TILE_DIR}/"
