#!/bin/bash
# =============================================================================
# [LEGACY] ND2 full fiber pipeline launcher (managed mode)
#
# Stages:
#   Step 1 (CPU):       Extract ND2 tiles (all channels)
#   Step 2 (GPU array): Fiber segmentation inference per tile
#   Step 3 (GPU array): Cell segmentation (micro-sam) per tile
#   Step 4 (CPU array): Fiber analysis per tile
#   Step 5 (CPU):       Merge per-tile CSV + profile NPZ outputs
#
# Usage:
#   bash run_nd2_pipeline.sh --nd2 /path/to/sample.nd2
#   bash run_nd2_pipeline.sh --nd2 /path/to/sample.nd2 --run-id 20260326_prod_v1
#   bash run_nd2_pipeline.sh --nd2 /path/to/sample.nd2 --skip-step1
#   bash run_nd2_pipeline.sh --nd2 /path/to/sample.nd2 --only-merge
# =============================================================================

set -euo pipefail

cat <<'EOF'
[LEGACY ENTRYPOINT] run_nd2_pipeline.sh
Recommended default ND2 pipeline:
  1) bash pipelines/nd2/run_pipeline.sh --nd2 /path/to/file.nd2
  2) bash pipelines/nd2/run_pipeline_batch.sh --nd2-dir /path/to/nd2_dir
This legacy script is kept for compatibility (5-step flow).
EOF

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNS_ROOT="/projects/weilab/dataset/barcode/2026/umich/fiber_runs_full"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
ND2_PATH=""
ND2_ID=""
CKPT="outputs/fiber_retrain_all/20260311_223801/checkpoints/last.ckpt"
TEMPLATE="tutorials/fiber_nd2_single_tile.yaml"
MAIL_USER="${USER}@bc.edu"
TILE_NAMES_CSV=""
CELL_SEG_MODEL_TYPE="vit_b_lm"
FIBER_N_JOBS=16

SKIP_STEP1=false
SKIP_STEP2=false
SKIP_STEP3=false
SKIP_STEP4=false
ONLY_MERGE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nd2)
            ND2_PATH="$2"
            shift 2
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --nd2-id)
            ND2_ID="$2"
            shift 2
            ;;
        --runs-root)
            RUNS_ROOT="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --checkpoint)
            CKPT="$2"
            shift 2
            ;;
        --template)
            TEMPLATE="$2"
            shift 2
            ;;
        --tile-names)
            TILE_NAMES_CSV="$2"
            shift 2
            ;;
        --cell-seg-model)
            CELL_SEG_MODEL_TYPE="$2"
            shift 2
            ;;
        --fiber-n-jobs)
            FIBER_N_JOBS="$2"
            shift 2
            ;;
        --mail-user)
            MAIL_USER="$2"
            shift 2
            ;;
        --skip-step1)
            SKIP_STEP1=true
            shift
            ;;
        --skip-step2)
            SKIP_STEP2=true
            shift
            ;;
        --skip-step3)
            SKIP_STEP3=true
            shift
            ;;
        --skip-step4)
            SKIP_STEP4=true
            shift
            ;;
        --only-merge)
            ONLY_MERGE=true
            SKIP_STEP1=true
            SKIP_STEP2=true
            SKIP_STEP3=true
            SKIP_STEP4=true
            shift
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$ND2_PATH" ]; then
    echo "ERROR: --nd2 is required"
    exit 1
fi

if [ ! -f "$ND2_PATH" ]; then
    echo "ERROR: ND2 file not found: $ND2_PATH"
    exit 1
fi

if [ ! -d "$WORK_DIR" ]; then
    echo "ERROR: work dir not found: $WORK_DIR"
    exit 1
fi

cd "$WORK_DIR"

ND2_PATH="$(realpath "$ND2_PATH")"
RUNS_ROOT="$(realpath -m "$RUNS_ROOT")"

if [ -z "$ND2_ID" ]; then
    ND2_BASENAME="$(basename "$ND2_PATH")"
    ND2_ID="${ND2_BASENAME%.*}"
fi

SAFE_ND2_ID="$(echo "$ND2_ID" | tr -c 'A-Za-z0-9_-' '_' | cut -c1-40)"

RUN_ROOT="${RUNS_ROOT}/${RUN_ID}"
ND2_ROOT="${RUN_ROOT}/${ND2_ID}"
INPUT_DIR="${ND2_ROOT}/input"
TILE_DIR="${ND2_ROOT}/tiles"
PRED_DIR="${ND2_ROOT}/pred"
POSTPROC_BASE="${ND2_ROOT}/postproc"
POSTPROC_ND2_DIR="${POSTPROC_BASE}/${ND2_ID}"
CACHE_DIR="${POSTPROC_ND2_DIR}/cache"
LOG_DIR="${ND2_ROOT}/logs"
META_DIR="${ND2_ROOT}/meta"
QC_DIR="${ND2_ROOT}/qc"

mkdir -p "$INPUT_DIR" "$TILE_DIR" "$PRED_DIR" "$POSTPROC_BASE" "$CACHE_DIR" "$LOG_DIR" "$META_DIR" "$QC_DIR"
ln -sfn "$ND2_PATH" "${INPUT_DIR}/$(basename "$ND2_PATH")"

TILE_NAMES_FILE="${META_DIR}/tile_names.txt"

write_tile_names_from_csv() {
    local csv="$1"
    : > "$TILE_NAMES_FILE"
    IFS=',' read -r -a names <<< "$csv"
    for t in "${names[@]}"; do
        t="$(echo "$t" | xargs)"
        [ -n "$t" ] && echo "$t" >> "$TILE_NAMES_FILE"
    done
}

detect_tile_names_from_existing_tiles() {
    find "$TILE_DIR" -maxdepth 1 -type f -name '*_ch1.tif' -printf '%f\n' \
        | sed 's/_ch1\.tif$//' | sort -u > "$TILE_NAMES_FILE"
}

detect_tile_names_from_nd2() {
    python - "$ND2_PATH" > "$TILE_NAMES_FILE" <<'PY'
import sys

try:
    import nd2
except Exception as exc:
    raise RuntimeError(f"nd2 import failed: {exc}")

nd2_path = sys.argv[1]

with nd2.ND2File(nd2_path) as f:
    names = []
    for loop in f.experiment:
        if getattr(loop, "type", "") == "XYPosLoop":
            for point in loop.parameters.points:
                names.append(str(point.name))
            break

if not names:
    raise RuntimeError("no XYPosLoop tile names found")

for name in names:
    print(name)
PY
}

if [ -n "$TILE_NAMES_CSV" ]; then
    write_tile_names_from_csv "$TILE_NAMES_CSV"
elif [ "$SKIP_STEP1" = true ] && find "$TILE_DIR" -maxdepth 1 -name '*_ch1.tif' | grep -q .; then
    detect_tile_names_from_existing_tiles
else
    if ! detect_tile_names_from_nd2; then
        echo "INFO: direct python tile discovery failed, retrying with conda env 'pytc'"
        set +u
        if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
            . "$HOME/miniconda3/etc/profile.d/conda.sh"
        elif [ -f "/home/liupen/miniconda3/etc/profile.d/conda.sh" ]; then
            . "/home/liupen/miniconda3/etc/profile.d/conda.sh"
        else
            source ~/.bashrc >/dev/null 2>&1 || true
        fi
        set -u
        conda activate pytc
        detect_tile_names_from_nd2
    fi
fi

if [ ! -s "$TILE_NAMES_FILE" ]; then
    echo "ERROR: failed to resolve tile names. Pass --tile-names explicitly."
    exit 1
fi

mapfile -t TILE_NAMES_ARRAY < "$TILE_NAMES_FILE"
TILE_COUNT="${#TILE_NAMES_ARRAY[@]}"
ARRAY_RANGE="0-$((TILE_COUNT - 1))"

MANIFEST_CSV="${RUN_ROOT}/manifest.csv"
if [ ! -f "$MANIFEST_CSV" ]; then
    echo "run_id,nd2_id,nd2_path,status_step1,status_step2,status_step3,status_step4,status_step5,submitted_at,nd2_root,tile_dir,pred_dir,postproc_dir,log_dir,checkpoint,template,tile_count" > "$MANIFEST_CSV"
fi

cat > "${META_DIR}/run_context.env" <<EOF_RUN
RUN_ID=${RUN_ID}
ND2_ID=${ND2_ID}
ND2_PATH=${ND2_PATH}
RUNS_ROOT=${RUNS_ROOT}
RUN_ROOT=${RUN_ROOT}
ND2_ROOT=${ND2_ROOT}
INPUT_DIR=${INPUT_DIR}
TILE_DIR=${TILE_DIR}
PRED_DIR=${PRED_DIR}
POSTPROC_BASE=${POSTPROC_BASE}
POSTPROC_ND2_DIR=${POSTPROC_ND2_DIR}
CACHE_DIR=${CACHE_DIR}
LOG_DIR=${LOG_DIR}
META_DIR=${META_DIR}
QC_DIR=${QC_DIR}
CKPT=${CKPT}
TEMPLATE=${TEMPLATE}
TILE_NAMES_FILE=${TILE_NAMES_FILE}
TILE_COUNT=${TILE_COUNT}
CELL_SEG_MODEL_TYPE=${CELL_SEG_MODEL_TYPE}
FIBER_N_JOBS=${FIBER_N_JOBS}
MAIL_USER=${MAIL_USER}
EOF_RUN

COMMON_EXPORT="ALL,RUN_ID=${RUN_ID},ND2_ID=${ND2_ID},ND2_PATH=${ND2_PATH},RUNS_ROOT=${RUNS_ROOT},RUN_ROOT=${RUN_ROOT},ND2_ROOT=${ND2_ROOT},INPUT_DIR=${INPUT_DIR},TILE_DIR=${TILE_DIR},PRED_DIR=${PRED_DIR},POSTPROC_BASE=${POSTPROC_BASE},POSTPROC_ND2_DIR=${POSTPROC_ND2_DIR},CACHE_DIR=${CACHE_DIR},LOG_DIR=${LOG_DIR},META_DIR=${META_DIR},QC_DIR=${QC_DIR},WORK_DIR=${WORK_DIR},CKPT=${CKPT},TEMPLATE=${TEMPLATE},TILE_NAMES_FILE=${TILE_NAMES_FILE},CELL_SEG_MODEL_TYPE=${CELL_SEG_MODEL_TYPE},FIBER_N_JOBS=${FIBER_N_JOBS},MAIL_USER=${MAIL_USER}"

STEP1_STATUS="skipped"
STEP2_STATUS="skipped"
STEP3_STATUS="skipped"
STEP4_STATUS="skipped"
STEP5_STATUS="submitted"

echo "================================================"
echo "ND2 Full Fiber Pipeline Launcher"
echo "================================================"
echo "Launched:     $(date)"
echo "run_id:       ${RUN_ID}"
echo "nd2_id:       ${ND2_ID}"
echo "nd2:          ${ND2_PATH}"
echo "tile_count:   ${TILE_COUNT}"
echo "tile_names:   $(paste -sd, "$TILE_NAMES_FILE")"
echo "nd2_root:     ${ND2_ROOT}"
echo "array_range:  ${ARRAY_RANGE}"
echo ""

STEP1_DEP=""
STEP2_JOB=""
STEP3_JOB=""
STEP4_JOB=""
STEP5_JOB=""

if [ "$SKIP_STEP1" = false ]; then
    STEP1_JOB=$(sbatch --parsable \
        --chdir "$WORK_DIR" \
        --export="${COMMON_EXPORT},EXTRACT_ALL_CHANNELS=true" \
        --job-name="nd2_extract_${SAFE_ND2_ID}" \
        --output "${LOG_DIR}/nd2_extract_%j.out" \
        --error "${LOG_DIR}/nd2_extract_%j.err" \
        slurm_jobs/step1_extract_tiles.sl)
    STEP1_STATUS="submitted"
    STEP1_DEP="--dependency=afterok:${STEP1_JOB}"
    echo "Step 1 submitted: ${STEP1_JOB} (extract all channels)"
else
    echo "Step 1 skipped"
fi

if [ "$SKIP_STEP2" = false ]; then
    STEP2_JOB=$(sbatch --parsable $STEP1_DEP \
        --chdir "$WORK_DIR" \
        --export="${COMMON_EXPORT}" \
        --job-name="nd2_infer_${SAFE_ND2_ID}" \
        --array="$ARRAY_RANGE" \
        --output "${LOG_DIR}/nd2_infer_%A_%a.out" \
        --error "${LOG_DIR}/nd2_infer_%A_%a.err" \
        slurm_jobs/step2_infer_tiles_array.sl)
    STEP2_STATUS="submitted"
    echo "Step 2 submitted: ${STEP2_JOB}_[${ARRAY_RANGE}] (fiber inference array)"
else
    echo "Step 2 skipped"
fi

if [ "$SKIP_STEP3" = false ]; then
    STEP3_JOB=$(sbatch --parsable $STEP1_DEP \
        --chdir "$WORK_DIR" \
        --export="${COMMON_EXPORT}" \
        --job-name="nd2_cellseg_${SAFE_ND2_ID}" \
        --array="$ARRAY_RANGE" \
        --output "${LOG_DIR}/nd2_cellseg_%A_%a.out" \
        --error "${LOG_DIR}/nd2_cellseg_%A_%a.err" \
        slurm_jobs/step3_cell_seg_array.sl)
    STEP3_STATUS="submitted"
    echo "Step 3 submitted: ${STEP3_JOB}_[${ARRAY_RANGE}] (cell segmentation array)"
else
    echo "Step 3 skipped"
fi

STEP4_DEP=""
STEP4_DEPS=()
if [ -n "$STEP2_JOB" ]; then
    STEP4_DEPS+=("$STEP2_JOB")
fi
if [ -n "$STEP3_JOB" ]; then
    STEP4_DEPS+=("$STEP3_JOB")
fi
if [ "${#STEP4_DEPS[@]}" -gt 0 ]; then
    STEP4_DEP="--dependency=afterok:$(IFS=:; echo "${STEP4_DEPS[*]}")"
fi

if [ "$SKIP_STEP4" = false ]; then
    STEP4_JOB=$(sbatch --parsable $STEP4_DEP \
        --chdir "$WORK_DIR" \
        --export="${COMMON_EXPORT}" \
        --job-name="nd2_fiber_${SAFE_ND2_ID}" \
        --array="$ARRAY_RANGE" \
        --output "${LOG_DIR}/nd2_fiber_%A_%a.out" \
        --error "${LOG_DIR}/nd2_fiber_%A_%a.err" \
        slurm_jobs/step4_fiber_array.sl)
    STEP4_STATUS="submitted"
    echo "Step 4 submitted: ${STEP4_JOB}_[${ARRAY_RANGE}] (fiber analysis array)"
else
    echo "Step 4 skipped"
fi

STEP5_DEP=""
if [ -n "$STEP4_JOB" ]; then
    STEP5_DEP="--dependency=afterok:${STEP4_JOB}"
fi

STEP5_JOB=$(sbatch --parsable $STEP5_DEP \
    --chdir "$WORK_DIR" \
    --export="${COMMON_EXPORT}" \
    --job-name="nd2_merge_${SAFE_ND2_ID}" \
    --output "${LOG_DIR}/nd2_merge_%j.out" \
    --error "${LOG_DIR}/nd2_merge_%j.err" \
    slurm_jobs/step5_merge_outputs.sl)

if [ "$ONLY_MERGE" = true ]; then
    echo "Only merge mode enabled"
fi

echo "${RUN_ID},${ND2_ID},${ND2_PATH},${STEP1_STATUS},${STEP2_STATUS},${STEP3_STATUS},${STEP4_STATUS},${STEP5_STATUS},$(date -Iseconds),${ND2_ROOT},${TILE_DIR},${PRED_DIR},${POSTPROC_ND2_DIR},${LOG_DIR},${CKPT},${TEMPLATE},${TILE_COUNT}" >> "$MANIFEST_CSV"

echo ""
echo "================================================"
echo "All jobs submitted"
echo "================================================"
echo "Step 1 job: ${STEP1_JOB:-N/A}"
echo "Step 2 job: ${STEP2_JOB:-N/A}_[${ARRAY_RANGE}]"
echo "Step 3 job: ${STEP3_JOB:-N/A}_[${ARRAY_RANGE}]"
echo "Step 4 job: ${STEP4_JOB:-N/A}_[${ARRAY_RANGE}]"
echo "Step 5 job: ${STEP5_JOB}"
echo ""
echo "Monitor:"
echo "  squeue -u $USER"
echo "  tail -f ${LOG_DIR}/nd2_merge_${STEP5_JOB}.out"
echo ""
echo "Paths:"
echo "  run root:    ${RUN_ROOT}"
echo "  nd2 root:    ${ND2_ROOT}"
echo "  tiles:       ${TILE_DIR}"
echo "  predictions: ${PRED_DIR}"
echo "  postproc:    ${POSTPROC_ND2_DIR}"
echo "  qc:          ${QC_DIR}"
echo "  context:     ${META_DIR}/run_context.env"
echo "================================================"
