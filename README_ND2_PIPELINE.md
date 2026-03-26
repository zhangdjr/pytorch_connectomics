# ND2 Pipeline README

This document describes the production-style ND2 inference pipeline based on:

- `slurm_jobs/run_pipeline.sh` (single ND2 launcher)
- `slurm_jobs/run_pipeline_batch.sh` (batch launcher for a directory of ND2 files)

The pipeline runs 3 stages per ND2:

1. Extract tiles from ND2
2. Run tile-level inference (SLURM array)
3. Run post-processing and generate CSV outputs

---

## 1. Key Concepts

### `run_id`
Batch namespace. All ND2 jobs in the same batch can share one `run_id`.

Default behavior:
- If `--run-id` is not provided, it is auto-generated as `YYYYMMDD_HHMMSS`.

### `nd2_id`
File namespace under one `run_id`.

Default behavior:
- If `--nd2-id` is not provided, it uses ND2 filename stem.

---

## 2. Output Layout

For each ND2:

`<runs_root>/<run_id>/<nd2_id>/`

Subdirectories:

- `input/`     ND2 symlink
- `tiles/`     extracted tile TIFFs + metadata JSON
- `pred/`      prediction outputs (`*_prediction.h5`, `*_prediction.tiff`)
- `postproc/`  per-tile CSVs + `all_tiles_fiber_coordinates.csv`
- `logs/`      SLURM logs for extract/infer/postproc
- `meta/`      run metadata (`run_context.env`, summaries)
- `qc/`        quality check report (`check_report.json`)

Run-level file:

- `<runs_root>/<run_id>/manifest.csv`

Default `runs_root`:

- `/projects/weilab/dataset/barcode/2026/umich/fiber_runs`

---

## 3. Single ND2 Usage

From repo root:

```bash
cd /projects/weilab/liupeng/projects/umich-fiber/pytorch_connectomics

bash slurm_jobs/run_pipeline.sh \
  --nd2 /projects/weilab/dataset/barcode/2026/umich/A1-2003.nd2 \
  --run-id 20260325_prod_v1
```

Required:

- `--nd2 /absolute/path/to/file.nd2`

Optional:

- `--run-id <id>`
- `--nd2-id <id>`
- `--runs-root <dir>`
- `--checkpoint <path>`
- `--template <path>`
- `--tile-names A1,A2,A3,B4,B3,B2,B1,C1,C2,C3,D2,D1,E1`
- `--skip-step1`
- `--only-step3`

---

## 4. Batch Usage (`--nd2-dir`)

Submit all ND2 files in one directory with a shared `run_id`:

```bash
cd /projects/weilab/liupeng/projects/umich-fiber/pytorch_connectomics

bash slurm_jobs/run_pipeline_batch.sh \
  --nd2-dir /projects/weilab/dataset/barcode/2026/umich \
  --glob "*.nd2" \
  --run-id 20260325_batch1
```

Required:

- `--nd2-dir <directory>`

Optional:

- `--glob <pattern>` (default: `*.nd2`)
- `--run-id <id>` (shared across the batch)
- `--runs-root <dir>`
- `--checkpoint <path>`
- `--template <path>`
- `--tile-names <csv>`
- `--skip-step1`
- `--only-step3`

---

## 5. Monitoring

Check queue:

```bash
squeue -u $USER
```

Tail logs (paths are printed by `run_pipeline.sh` on submission):

```bash
tail -f <nd2_root>/logs/nd2_extract_<jobid>.out
tail -f <nd2_root>/logs/nd2_infer_<array_jobid>_0.out
tail -f <nd2_root>/logs/nd2_postproc_<jobid>.out
```

---

## 6. Notes

- The pipeline is now isolated per ND2 to avoid cross-file cache/output collisions.
- Step 2 writes predictions into per-ND2 `pred/` via `test.data.output_path`.
- Step 3 validates prediction TIFF presence before post-processing.
- `manifest.csv` is append-only submission metadata for batch tracking.

