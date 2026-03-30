#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET="${REPO_DIR}/pipelines/nd2/run_pipeline_batch.sh"

if [ ! -x "$TARGET" ]; then
  echo "ERROR: target script not executable: $TARGET"
  exit 1
fi

echo "[compat] slurm_jobs/run_pipeline_batch.sh -> pipelines/nd2/run_pipeline_batch.sh"
exec bash "$TARGET" "$@"
