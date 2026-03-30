# ND2 Pipeline (Current Production Version)

This document describes the **current SLURM ND2 pipeline** used in this repo:

- `slurm_jobs/run_pipeline.sh` (single ND2 launcher)
- `slurm_jobs/run_pipeline_batch.sh` (batch launcher for one ND2 directory)
- `slurm_jobs/step1_extract_tiles.sl`
- `slurm_jobs/step2_infer_tiles_array.sl`
- `slurm_jobs/step3_postprocess.sl`

## 0. Entry Points (Use This)

Recommended default entrypoints (single source of truth):

- `bash pipelines/nd2/run_pipeline.sh --nd2 /path/to/file.nd2`
- `bash pipelines/nd2/run_pipeline_batch.sh --nd2-dir /path/to/nd2_dir`

Legacy entrypoint (kept for backward compatibility):

- `bash run_nd2_pipeline.sh ...` (5-step flow with cell-seg + fiber analysis + merge)
- `bash slurm_jobs/run_pipeline*.sh ...` (compat wrappers to `pipelines/nd2/`)

## 0.5 Repository Layout

Structured folders:

- `pipelines/nd2/`: canonical launcher scripts
- `slurm_jobs/nd2/`: canonical ND2 SLURM step scripts
- `tools/`: ND2/fiber utility scripts

Compatibility wrappers are kept at old locations (`slurm_jobs/run_pipeline*.sh`, `run_nd2_pipeline.sh`).

## 1. What This Pipeline Does

For each ND2 file, the pipeline runs:

1. Step 1 (CPU): extract tiles from ND2
2. Step 2 (GPU array): run tile-level model inference
3. Step 3 (CPU): postprocess predictions and write fiber CSVs

Important:

- Step 2 is **dynamic tile-count aware**.
- Step 3 currently calls `tools/generate_fiber_coordinates.py` (not `tools/fiber_pipeline.py`).
- By default, Step 3 deletes `tiles/` after success (`CLEANUP_TILES=true`).

## 2. Key Dynamic Tile Behavior

- Step 1 writes actual tile names to:
  - `<nd2_root>/meta/tile_names.txt`
- Step 2 is submitted as `--array=0-(N-1)` where `N = --max-array-tasks` (default 64).
- Inside Step 2, tasks with index >= actual tile count exit successfully (no-op).
- Step 2 temporary per-tile YAML is written to `<nd2_root>/meta/tmp/` (not `tutorials/`).
- This supports ND2 files with different tile counts (for example: 4, 8, 12, 14, 23).

## 3. Output Layout

Per ND2:

`<runs_root>/<run_id>/<nd2_id>/`

Subdirectories:

- `input/`: ND2 symlink
- `tiles/`: extracted tile TIFFs + metadata JSON (may be deleted after Step 3)
- `pred/`: model outputs (`*_ch1_prediction.tiff`, `*_ch1_prediction.h5`, optional TTA files)
- `postproc/`: tile CSVs + `all_tiles_fiber_coordinates.csv`
- `logs/`: SLURM logs
- `meta/`: run metadata (`run_context.env`, step summaries, tile_names.txt)
- `qc/`: quality check report (`check_report.json`)

Run-level:

- `<runs_root>/<run_id>/manifest.csv`

## 4. Single ND2 Usage

```bash
cd <repo_root>

bash pipelines/nd2/run_pipeline.sh \
  --nd2 /absolute/path/to/file.nd2 \
  --run-id 20260328_test_single
```

Required:

- `--nd2 /absolute/path/to/file.nd2`

Optional:

- `--run-id <id>`
- `--nd2-id <id>`
- `--runs-root <dir>`
- `--checkpoint <path>`
- `--template <path>`
- `--tile-names <csv>` (seed list; actual list is refreshed from extraction)
- `--max-array-tasks <N>` (default 64)
- `--skip-step1`
- `--only-step3`

## 5. Batch Usage (Directory of ND2 Files)

```bash
cd <repo_root>

bash pipelines/nd2/run_pipeline_batch.sh \
  --nd2-dir /projects/weilab/dataset/barcode/2026/broad_dongqing \
  --glob "*.nd2" \
  --run-id 20260328_batch1 \
  --runs-root /projects/weilab/dataset/barcode/2026/broad_dongqing/fiber_runs \
  --max-array-tasks 64
```

Required:

- `--nd2-dir <directory>`

Optional:

- `--glob <pattern>` (default `*.nd2`)
- `--run-id <id>`
- `--runs-root <dir>`
- `--checkpoint <path>`
- `--template <path>`
- `--tile-names <csv>`
- `--max-array-tasks <N>`
- `--skip-step1`
- `--only-step3`

## 6. Monitoring

Queue:

```bash
squeue -u $USER
```

Accounting (replace IDs):

```bash
sacct -j <jobid_or_arrayid> --format=JobID,JobName,State,ExitCode,Start,End
```

Live logs:

```bash
tail -f <nd2_root>/logs/nd2_extract_<jobid>.out
tail -f <nd2_root>/logs/nd2_infer_<array_jobid>_0.out
tail -f <nd2_root>/logs/nd2_postproc_<jobid>.out
```

## 7. Completion Criteria (Recommended)

An ND2 is considered successful when:

- Step 1/2/3 states are all `COMPLETED`
- `<nd2_root>/postproc/all_tiles_fiber_coordinates.csv` exists
- `<nd2_root>/meta/step3_postproc_summary.txt` contains `master_csv_exists=True`

Quick check example:

```bash
run_root=/path/to/fiber_runs/<run_id>
for d in "$run_root"/*; do
  [ -d "$d" ] || continue
  nd2=$(basename "$d")
  if [ -f "$d/postproc/all_tiles_fiber_coordinates.csv" ]; then
    echo "$nd2: OK"
  else
    echo "$nd2: MISSING_MASTER_CSV"
  fi
done
```

## 8. Cleanup Behavior

Default:

- Step 3 runs with `CLEANUP_TILES=true`.
- After successful postprocess, `tiles/` is deleted to save disk.

If you want to keep tiles:

```bash
CLEANUP_TILES=false bash slurm_jobs/run_pipeline.sh --nd2 /path/to/file.nd2
```

Or for batch:

```bash
CLEANUP_TILES=false bash slurm_jobs/run_pipeline_batch.sh --nd2-dir /path/to/nd2_dir
```

## 9. Notes and Scope

- This README describes the **SLURM ND2 inference/postprocess pipeline** only.
- `tools/fiber_pipeline.py` is a separate analysis pipeline and is not called by current Step 3.
- For large batches, keep a unique `run_id` per submission and avoid reusing old `run_id` unless you intentionally overwrite/retry within the same run namespace.
