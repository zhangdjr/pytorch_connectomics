#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PIPELINE_SCRIPT="${WORK_DIR}/pipelines/nd2/run_pipeline.sh"

ND2_DIR=""
GLOB_PATTERN="*.nd2"
RUNS_ROOT="/projects/weilab/dataset/barcode/2026/broad_dongqing/fiber_results"
SUBMIT_TAG="$(date +%Y%m%d_%H%M%S)"
IGNORED_RUN_ID=""
CHECKPOINT=""
TEMPLATE=""
TILE_NAMES=""
MAX_ARRAY_TASKS=""
EXCLUDE_NODES=""
SKIP_STEP1=false
ONLY_STEP3=false

usage() {
    cat <<'EOF'
Batch submit ND2 pipeline jobs.

Usage:
  bash pipelines/nd2/run_pipeline_batch.sh --nd2-dir /path/to/nd2_dir [options]

Required:
  --nd2-dir DIR            Directory containing ND2 files

Optional:
  --glob PATTERN           Filename pattern in nd2-dir (default: *.nd2)
  --run-id ID              Deprecated; ignored (run-id is no longer used)
  --runs-root DIR          Root directory for outputs
  --checkpoint PATH        Override checkpoint passed to run_pipeline.sh
  --template PATH          Override config template passed to run_pipeline.sh
  --tile-names CSV         Override tile names CSV (e.g. A1,A2,...)
  --max-array-tasks N      Max array size for Step 2 (default in run_pipeline.sh: 64)
  --exclude-nodes CSV      Exclude nodes for all steps (e.g. g007 or g007,g008)
  --skip-step1             Pass through to run_pipeline.sh
  --only-step3             Pass through to run_pipeline.sh
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nd2-dir)
            ND2_DIR="$2"
            shift 2
            ;;
        --glob)
            GLOB_PATTERN="$2"
            shift 2
            ;;
        --run-id)
            IGNORED_RUN_ID="$2"
            shift 2
            ;;
        --runs-root)
            RUNS_ROOT="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --template)
            TEMPLATE="$2"
            shift 2
            ;;
        --tile-names)
            TILE_NAMES="$2"
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
        --skip-step1)
            SKIP_STEP1=true
            shift
            ;;
        --only-step3)
            ONLY_STEP3=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$ND2_DIR" ]]; then
    echo "ERROR: --nd2-dir is required."
    usage
    exit 1
fi

if [[ ! -d "$ND2_DIR" ]]; then
    echo "ERROR: ND2 directory not found: $ND2_DIR"
    exit 1
fi

if [[ ! -x "$PIPELINE_SCRIPT" ]]; then
    echo "ERROR: Pipeline script not executable: $PIPELINE_SCRIPT"
    exit 1
fi

if [[ -n "$IGNORED_RUN_ID" ]]; then
    echo "WARNING: --run-id is deprecated and ignored. Outputs now go to runs-root/nd2_id."
fi

RUN_ROOT="${RUNS_ROOT}"
mkdir -p "${RUN_ROOT}/logs"

mapfile -d '' ND2_FILES < <(find "$ND2_DIR" -maxdepth 1 -type f -name "$GLOB_PATTERN" -print0 | sort -z)

if [[ "${#ND2_FILES[@]}" -eq 0 ]]; then
    echo "ERROR: No files matched pattern '$GLOB_PATTERN' in $ND2_DIR"
    exit 1
fi

echo "================================================"
echo "Batch ND2 Pipeline Submission"
echo "================================================"
echo "submit_tag:  $SUBMIT_TAG"
echo "nd2_dir:     $ND2_DIR"
echo "glob:        $GLOB_PATTERN"
echo "runs_root:   $RUNS_ROOT"
echo "nd2_count:   ${#ND2_FILES[@]}"
if [[ -n "$EXCLUDE_NODES" ]]; then
    echo "exclude:     $EXCLUDE_NODES"
fi
echo "pipeline:    $PIPELINE_SCRIPT"
echo "================================================"
echo ""

submitted=0
declare -a STEP3_JOB_IDS=()
for nd2_file in "${ND2_FILES[@]}"; do
    echo "Submitting: $nd2_file"

    cmd=(
        bash "$PIPELINE_SCRIPT"
        --nd2 "$nd2_file"
        --runs-root "$RUNS_ROOT"
    )

    if [[ -n "$CHECKPOINT" ]]; then
        cmd+=(--checkpoint "$CHECKPOINT")
    fi
    if [[ -n "$TEMPLATE" ]]; then
        cmd+=(--template "$TEMPLATE")
    fi
    if [[ -n "$TILE_NAMES" ]]; then
        cmd+=(--tile-names "$TILE_NAMES")
    fi
    if [[ -n "$MAX_ARRAY_TASKS" ]]; then
        cmd+=(--max-array-tasks "$MAX_ARRAY_TASKS")
    fi
    if [[ -n "$EXCLUDE_NODES" ]]; then
        cmd+=(--exclude-nodes "$EXCLUDE_NODES")
    fi
    if [[ "$SKIP_STEP1" == true ]]; then
        cmd+=(--skip-step1)
    fi
    if [[ "$ONLY_STEP3" == true ]]; then
        cmd+=(--only-step3)
    fi

    submit_log="$(mktemp "${RUN_ROOT}/logs/submit_${SUBMIT_TAG}_XXXXXX.log")"
    if ! "${cmd[@]}" | tee "$submit_log"; then
        echo "ERROR: Submission failed for ${nd2_file}. See ${submit_log}"
        exit 1
    fi

    step3_job_id="$(sed -n 's/^PIPELINE_JOB_IDS .* STEP3=\([0-9][0-9]*\).*$/\1/p' "$submit_log" | tail -n 1)"
    if [[ -n "${step3_job_id}" ]]; then
        STEP3_JOB_IDS+=("${step3_job_id}")
    else
        echo "WARNING: Could not parse Step3 job id from ${submit_log}"
    fi

    submitted=$((submitted + 1))
    echo ""
done

echo "Batch submission complete: ${submitted} ND2 jobs submitted."

if [[ "${#STEP3_JOB_IDS[@]}" -gt 0 ]]; then
    dep_list="$(IFS=:; echo "${STEP3_JOB_IDS[*]}")"
    SAFE_TAG="$(echo "$SUBMIT_TAG" | tr -c 'A-Za-z0-9_-' '_' | cut -c1-30)"
    SUMMARY_JOB=$(sbatch --parsable \
        --dependency="afterany:${dep_list}" \
        --chdir "$WORK_DIR" \
        --export="ALL,RUN_ID=no_run_id,RUNS_ROOT=${RUNS_ROOT},RUN_ROOT=${RUN_ROOT},WORK_DIR=${WORK_DIR}" \
        --job-name="nd2_summary_${SAFE_TAG}" \
        --output "${RUN_ROOT}/logs/nd2_summary_%j.out" \
        --error "${RUN_ROOT}/logs/nd2_summary_%j.err" \
        slurm_jobs/nd2/step4_summarize_run.sl)

    echo ""
    echo "Summary job submitted: ${SUMMARY_JOB}"
    echo "  report path: ${RUN_ROOT}/run_summary.md"
    echo "  status table: ${RUN_ROOT}/case_status_and_durations.tsv"
    echo "  logs: ${RUN_ROOT}/logs/nd2_summary_${SUMMARY_JOB}.out"
else
    echo "WARNING: No Step3 job ids parsed; summary job was not submitted."
fi
