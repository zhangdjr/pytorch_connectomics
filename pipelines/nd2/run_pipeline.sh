#!/bin/bash
# ============================================================
# Master launcher for the 3-stage ND2 fiber segmentation pipeline.
#
# Usage:
#   bash pipelines/nd2/run_pipeline.sh --nd2 /path/to/file.nd2
#   bash pipelines/nd2/run_pipeline.sh --nd2 /path/to/file.nd2 --run-id 20260325_prod_v1
#   bash pipelines/nd2/run_pipeline.sh --nd2 /path/to/file.nd2 --skip-step1
#   bash pipelines/nd2/run_pipeline.sh --nd2 /path/to/file.nd2 --only-step3
#   bash pipelines/nd2/run_pipeline.sh --nd2 /path/to/file.nd2 --exclude-nodes g007
#
# Dependency chain:
#   Step 1 (CPU, extract) → Step 2 (GPU array, dynamic tile count) → Step 3 (CPU, CSV)
#
# Estimated wall time:
#   Step 1: ~10 min   (CPU)
#   Step 2: ~60 min   (parallel tiles on A100; depends on tile count)
#   Step 3: ~15 min   (CPU)
#   Total:  ~1.5 h typical
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$WORK_DIR"

SKIP_STEP1=false
ONLY_STEP3=false
RUNS_ROOT="/projects/weilab/dataset/barcode/2026/umich/fiber_runs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
ND2_PATH=""
ND2_ID=""
CKPT="checkpoints/last.ckpt"
TEMPLATE="tutorials/fiber_nd2_single_tile.yaml"
TILE_NAMES_CSV="A1,A2,A3,B4,B3,B2,B1,C1,C2,C3,D2,D1,E1"
MAX_ARRAY_TASKS=64
EXCLUDE_NODES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-step1)
            SKIP_STEP1=true
            shift
            ;;
        --only-step3)
            ONLY_STEP3=true
            SKIP_STEP1=true
            shift
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --nd2)
            ND2_PATH="$2"
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
        --max-array-tasks)
            MAX_ARRAY_TASKS="$2"
            shift 2
            ;;
        --exclude-nodes)
            EXCLUDE_NODES="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$ND2_PATH" ]; then
    echo "ERROR: --nd2 is required."
    exit 1
fi

if [ ! -f "$ND2_PATH" ]; then
    echo "ERROR: ND2 file not found: $ND2_PATH"
    exit 1
fi

if [ -z "$ND2_ID" ]; then
    ND2_BASENAME="$(basename "$ND2_PATH")"
    ND2_ID="${ND2_BASENAME%.*}"
fi

if ! [[ "$MAX_ARRAY_TASKS" =~ ^[0-9]+$ ]] || [ "$MAX_ARRAY_TASKS" -lt 1 ]; then
    echo "ERROR: --max-array-tasks must be a positive integer (got: ${MAX_ARRAY_TASKS})"
    exit 1
fi

IFS=',' read -r -a TILE_NAMES_ARRAY <<< "$TILE_NAMES_CSV"
if [ "${#TILE_NAMES_ARRAY[@]}" -le 0 ]; then
    echo "ERROR: --tile-names resolved to empty list."
    exit 1
fi
ARRAY_RANGE="0-$((MAX_ARRAY_TASKS - 1))"

SAFE_ND2_ID="$(echo "$ND2_ID" | tr -c 'A-Za-z0-9_-' '_' | cut -c1-40)"

RUN_ROOT="${RUNS_ROOT}/${RUN_ID}"
ND2_ROOT="${RUN_ROOT}/${ND2_ID}"
INPUT_DIR="${ND2_ROOT}/input"
TILE_DIR="${ND2_ROOT}/tiles"
PRED_DIR="${ND2_ROOT}/pred"
POSTPROC_DIR="${ND2_ROOT}/postproc"
LOG_DIR="${ND2_ROOT}/logs"
META_DIR="${ND2_ROOT}/meta"
QC_DIR="${ND2_ROOT}/qc"

mkdir -p "$INPUT_DIR" "$TILE_DIR" "$PRED_DIR" "$POSTPROC_DIR" "$LOG_DIR" "$META_DIR" "$QC_DIR"
ln -sfn "$ND2_PATH" "${INPUT_DIR}/$(basename "$ND2_PATH")"
TILE_NAMES_FILE="${META_DIR}/tile_names.txt"
printf "%s\n" "${TILE_NAMES_ARRAY[@]}" > "$TILE_NAMES_FILE"

if [ "$SKIP_STEP1" = true ]; then
    mapfile -t EXISTING_TILE_NAMES < <(
        find "$TILE_DIR" -maxdepth 1 -type f -name '*_ch1.tif' -printf '%f\n' 2>/dev/null \
            | sed 's/_ch1\.tif$//' | sort -u
    )
    if [ "${#EXISTING_TILE_NAMES[@]}" -gt 0 ]; then
        printf "%s\n" "${EXISTING_TILE_NAMES[@]}" > "$TILE_NAMES_FILE"
        echo "Step1 skipped: discovered ${#EXISTING_TILE_NAMES[@]} existing tile(s) in ${TILE_DIR}"
    else
        echo "Step1 skipped: no existing tiles found in ${TILE_DIR}; using --tile-names seed list (${#TILE_NAMES_ARRAY[@]} entries)"
    fi
fi

MANIFEST_CSV="${RUN_ROOT}/manifest.csv"
if [ ! -f "$MANIFEST_CSV" ]; then
    echo "run_id,nd2_id,nd2_path,status_extract,status_infer,status_postproc,submitted_at,nd2_root,pred_dir,postproc_dir,log_dir,model_ckpt,config_template" > "$MANIFEST_CSV"
fi

cat > "${META_DIR}/run_context.env" <<EOF
RUN_ID=${RUN_ID}
ND2_ID=${ND2_ID}
ND2_PATH=${ND2_PATH}
RUNS_ROOT=${RUNS_ROOT}
ND2_ROOT=${ND2_ROOT}
INPUT_DIR=${INPUT_DIR}
TILE_DIR=${TILE_DIR}
PRED_DIR=${PRED_DIR}
POSTPROC_DIR=${POSTPROC_DIR}
LOG_DIR=${LOG_DIR}
META_DIR=${META_DIR}
QC_DIR=${QC_DIR}
CKPT=${CKPT}
TEMPLATE=${TEMPLATE}
TILE_NAMES_CSV=${TILE_NAMES_CSV}
TILE_NAMES_FILE=${TILE_NAMES_FILE}
MAX_ARRAY_TASKS=${MAX_ARRAY_TASKS}
EXCLUDE_NODES=${EXCLUDE_NODES}
MANIFEST_CSV=${MANIFEST_CSV}
EOF

COMMON_EXPORT="ALL,RUN_ID=${RUN_ID},ND2_ID=${ND2_ID},ND2_PATH=${ND2_PATH},RUNS_ROOT=${RUNS_ROOT},ND2_ROOT=${ND2_ROOT},INPUT_DIR=${INPUT_DIR},TILE_DIR=${TILE_DIR},PRED_DIR=${PRED_DIR},POSTPROC_DIR=${POSTPROC_DIR},LOG_DIR=${LOG_DIR},META_DIR=${META_DIR},QC_DIR=${QC_DIR},WORK_DIR=${WORK_DIR},CKPT=${CKPT},TEMPLATE=${TEMPLATE},TILE_NAMES_FILE=${TILE_NAMES_FILE},MAX_ARRAY_TASKS=${MAX_ARRAY_TASKS}"
if [ -n "${EXCLUDE_NODES}" ]; then
    COMMON_EXPORT="${COMMON_EXPORT},EXCLUDE_NODES=${EXCLUDE_NODES}"
fi

SBATCH_EXCLUDE_ARGS=()
if [ -n "${EXCLUDE_NODES}" ]; then
    SBATCH_EXCLUDE_ARGS=(--exclude "${EXCLUDE_NODES}")
fi

echo "================================================"
echo "ND2 Fiber Segmentation Pipeline"
echo "================================================"
echo "Launched: $(date)"
echo "run_id:   ${RUN_ID}"
echo "nd2_id:   ${ND2_ID}"
echo "nd2:      ${ND2_PATH}"
echo "nd2_root: ${ND2_ROOT}"
if [ -n "${EXCLUDE_NODES}" ]; then
    echo "exclude:  ${EXCLUDE_NODES}"
fi
echo ""

# ── Step 1 ────────────────────────────────────────
if [ "$SKIP_STEP1" = true ]; then
    echo "Step 1: Skipped (--skip-step1)"
    STEP1_DEP=""
else
    JOB1=$(sbatch --parsable "${SBATCH_EXCLUDE_ARGS[@]}" \
        --chdir "$WORK_DIR" \
        --export="$COMMON_EXPORT" \
        --job-name="nd2_extract_${SAFE_ND2_ID}" \
        --output "${LOG_DIR}/nd2_extract_%j.out" \
        --error "${LOG_DIR}/nd2_extract_%j.err" \
        slurm_jobs/nd2/step1_extract_tiles.sl)
    echo "Step 1: Submitted  job=${JOB1}  (CPU, extract tiles)"
    STEP1_DEP="--dependency=afterok:${JOB1}"
fi

# ── Step 2 ────────────────────────────────────────
if [ "$ONLY_STEP3" = true ]; then
    echo "Step 2: Skipped (--only-step3)"
    STEP2_DEP=""
else
    JOB2=$(sbatch --parsable $STEP1_DEP "${SBATCH_EXCLUDE_ARGS[@]}" \
        --chdir "$WORK_DIR" \
        --export="$COMMON_EXPORT" \
        --job-name="nd2_infer_${SAFE_ND2_ID}" \
        --array="$ARRAY_RANGE" \
        --output "${LOG_DIR}/nd2_infer_%A_%a.out" \
        --error "${LOG_DIR}/nd2_infer_%A_%a.err" \
        slurm_jobs/nd2/step2_infer_tiles_array.sl)
    ARRAY_JOB_ID=$JOB2
    echo "Step 2: Submitted  job=${JOB2}  (GPU array ${ARRAY_RANGE}; actual tile count from ${TILE_NAMES_FILE})"
    # Wait for all array tasks before launching Step 3
    STEP2_DEP="--dependency=afterok:${ARRAY_JOB_ID}"
fi

# ── Step 3 ────────────────────────────────────────
JOB3=$(sbatch --parsable $STEP2_DEP "${SBATCH_EXCLUDE_ARGS[@]}" \
    --chdir "$WORK_DIR" \
    --export="$COMMON_EXPORT" \
    --job-name="nd2_postproc_${SAFE_ND2_ID}" \
    --output "${LOG_DIR}/nd2_postproc_%j.out" \
    --error "${LOG_DIR}/nd2_postproc_%j.err" \
    slurm_jobs/nd2/step3_postprocess.sl)
echo "Step 3: Submitted  job=${JOB3}  (CPU, generate CSVs)"

STEP1_ID="${JOB1:-NA}"
STEP2_ID="${ARRAY_JOB_ID:-NA}"
cat > "${META_DIR}/job_ids.env" <<EOF
step1_job_id=${STEP1_ID}
step2_job_id=${STEP2_ID}
step3_job_id=${JOB3}
EOF

echo "${RUN_ID},${ND2_ID},${ND2_PATH},submitted,submitted,submitted,$(date -Iseconds),${ND2_ROOT},${PRED_DIR},${POSTPROC_DIR},${LOG_DIR},${CKPT},${TEMPLATE}" >> "$MANIFEST_CSV"

echo ""
echo "================================================"
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Job IDs:"
[ "${SKIP_STEP1}" = false ] && echo "  Step 1 (extract): ${JOB1}"
[ "${ONLY_STEP3}"  = false ] && echo "  Step 2 (infer):   ${ARRAY_JOB_ID}_[${ARRAY_RANGE}]"
echo "  Step 3 (postproc): ${JOB3}"
echo ""
echo "Live logs:"
[ "${SKIP_STEP1}" = false ] && echo "  tail -f ${LOG_DIR}/nd2_extract_${JOB1}.out"
[ "${ONLY_STEP3}"  = false ] && echo "  tail -f ${LOG_DIR}/nd2_infer_${ARRAY_JOB_ID}_0.out"
echo "  tail -f ${LOG_DIR}/nd2_postproc_${JOB3}.out"
echo ""
echo "Output locations:"
echo "  Input ND2 link:   ${INPUT_DIR}/"
echo "  Tiles (TIFF):     ${TILE_DIR}/"
echo "  Predictions:      ${PRED_DIR}/"
echo "  Fiber CSVs:       ${POSTPROC_DIR}/"
echo "  Metadata/QC:      ${META_DIR}/, ${QC_DIR}/"
echo "  Run context:      ${META_DIR}/run_context.env"
echo "  Job IDs file:     ${META_DIR}/job_ids.env"
echo "================================================"
echo "PIPELINE_JOB_IDS RUN_ID=${RUN_ID} ND2_ID=${ND2_ID} STEP1=${STEP1_ID} STEP2=${STEP2_ID} STEP3=${JOB3}"
