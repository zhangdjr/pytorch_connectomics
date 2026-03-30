#!/bin/bash
#SBATCH --job-name=nd2_summary
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=short
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=liupen@bc.edu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

WORK_DIR="${WORK_DIR:-${REPO_DIR}}"
RUN_ID="${RUN_ID:-}"
RUNS_ROOT="${RUNS_ROOT:-}"
RUN_ROOT="${RUN_ROOT:-}"

if [ -z "${RUN_ROOT}" ]; then
    if [ -n "${RUNS_ROOT}" ] && [ -n "${RUN_ID}" ]; then
        RUN_ROOT="${RUNS_ROOT}/${RUN_ID}"
    fi
fi

if [ -z "${RUN_ROOT}" ]; then
    echo "ERROR: RUN_ROOT is not set and cannot be inferred from RUNS_ROOT/RUN_ID"
    exit 1
fi

echo "============================================"
echo "Step 4: Batch Summary Report"
echo "============================================"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     ${SLURM_NODELIST}"
echo "Start:    $(date)"
echo "run_id:   ${RUN_ID}"
echo "run_root: ${RUN_ROOT}"
echo ""

cd "${WORK_DIR}"
python -u pipelines/nd2/summarize_run.py --run-root "${RUN_ROOT}"

echo ""
echo "Done: $(date)"
echo "Summary:"
echo "  ${RUN_ROOT}/run_summary.md"
echo "  ${RUN_ROOT}/case_status_and_durations.tsv"
echo "  ${RUN_ROOT}/run_summary.json"
