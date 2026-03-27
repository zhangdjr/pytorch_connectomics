# Fiber Analysis Pipeline

Automated pipeline for analyzing fiber structures in confocal microscopy ND2 images. Takes a raw ND2 file and produces a CSV spreadsheet with measurements for every detected fiber.

---

## What Does This Pipeline Do?

Given an ND2 microscopy file, the pipeline automatically:

1. **Extracts** each tile and channel from the ND2 file
2. **Detects fibers** using a trained deep learning model (GPU)
3. **Detects cell bodies** using micro-sam segmentation
4. **Analyzes each fiber** — measures length, extracts signal intensities from all 4 channels (DAPI, fiber, cfos, timestamp), links fibers to their parent cell bodies, and filters out invalid detections
5. **Produces a CSV** with one row per fiber and 11 measurement columns

**You only need to run one command.** Everything else is automatic.

---

## How to Run (Step by Step)

### 1. Log in to the BC cluster

```bash
ssh your_username@a002.bc.edu
```

### 2. Go to the project folder

```bash
cd /home/zhangdjr/projects/umich-fiber/pytorch_connectomics
```

> If you cloned your own copy, `cd` to that directory instead.

### 3. Process one ND2 file

```bash
sbatch run_nd2_pipeline.sh /path/to/your_file.nd2
```

**Example:**
```bash
sbatch run_nd2_pipeline.sh /projects/weilab/dataset/barcode/2026/broad_dongqing/1-A1-2005.nd2
```

That's it! The job is now running on the cluster. You'll get an email when it starts and when it finishes.

### 4. Process many ND2 files at once

```bash
for nd2 in /projects/weilab/dataset/barcode/2026/broad_dongqing/*.nd2; do
    sbatch run_nd2_pipeline.sh "$nd2"
done
```

Each file runs as a separate job in parallel. No need to wait for one to finish before starting the next.

### 5. Check job status

```bash
squeue -u $USER
```

### 6. Find your results

Results are saved to `fiber_results/{nd2_name}/`:

```
fiber_results/
└── 1-A1-2005/                                # one folder per ND2 file
    ├── 1-A1-2005_combined.csv                # <-- SUMMARY CSV (all tiles combined)
    ├── 1-A1-2005_combined_profiles.npz       # <-- FULL INTENSITY PROFILES (all tiles)
    ├── 1-A1-2005_A1.csv                      # per-tile CSV
    ├── 1-A1-2005_A1_profiles.npz             # per-tile profiles
    ├── fiber_seg/                            # fiber segmentation masks
    └── cache/                                # intermediate files (cell seg, skeletons)
```

**The file you want is `{nd2_name}_combined.csv`** — open it in Excel, Google Sheets, or Python/R.

> **Note:** The `tiles/` directory is automatically deleted after the pipeline finishes to save disk space (~95% savings). Tiles can always be re-extracted from the original ND2 file.

---

## What's in the CSV?

Each row is one detected fiber. Columns:

| Column | What it means |
|--------|---------------|
| `fiber_id` | Unique ID for each fiber |
| `nd2_name` | Source ND2 file name |
| `tile_name` | Which tile the fiber came from (e.g., A1, B5) |
| `is_valid` | `True` if the fiber passed quality filters, `False` otherwise |
| `parent_cell_id` | Which cell body the fiber belongs to (0 = no cell found) |
| `fiber_length_um` | Length of the fiber in micrometers |
| `pca_linearity` | How straight the fiber is (1.0 = perfectly straight) |
| `centroid_z_um` | Fiber midpoint Z coordinate in micrometers |
| `centroid_y_um` | Fiber midpoint Y coordinate in micrometers |
| `centroid_x_um` | Fiber midpoint X coordinate in micrometers |
| `mean_soma_dapi` | Average DAPI brightness at the cell body |

> **Tip:** Filter by `is_valid == True` to get only high-quality fiber measurements.
>
> Per-channel intensity statistics are NOT in the CSV — use the intensity profiles (NPZ) instead. See below.

### Full Intensity Profiles

In addition to the summary CSV, the pipeline saves **full 1000-point intensity profiles** for every fiber. These are the raw signal values sampled along the fiber's skeleton from one end to the other.

The profiles are saved as `.npz` files (a compressed NumPy format):

```
fiber_results/1-A1-2005/
├── 1-A1-2005_combined_profiles.npz    # <-- all tiles combined
├── 1-A1-2005_A1_profiles.npz          # per-tile profiles
└── 1-A1-2005_B5_profiles.npz
```

**How to load profiles in Python:**
```python
import numpy as np

data = np.load("fiber_results/1-A1-2005/1-A1-2005_combined_profiles.npz")

# What's inside:
data["fiber_ids"]   # (N,) array of fiber IDs
data["is_valid"]    # (N,) boolean — True for valid fibers
data["tile_names"]  # (N,) array of tile names (e.g., "A1", "B5")
data["dapi"]        # (N, 1000) — DAPI intensity profile for each fiber
data["fiber"]       # (N, 1000) — fiber channel intensity
data["cfos"]        # (N, 1000) — cfos intensity
data["timestamp"]   # (N, 1000) — timestamp intensity

# Example: plot cfos profile for fiber #42
import matplotlib.pyplot as plt
idx = np.where(data["fiber_ids"] == 42)[0][0]
plt.plot(data["cfos"][idx])
plt.xlabel("Position along fiber (0 = one end, 999 = other end)")
plt.ylabel("cfos intensity")
plt.title(f"Fiber {42} cfos profile")
plt.show()

# Example: get all valid cfos profiles
valid = data["is_valid"]
cfos_valid = data["cfos"][valid]  # shape (N_valid, 1000)
```

Each profile has 1000 evenly-spaced points along the fiber's skeleton. Point 0 is one end of the fiber, point 999 is the other end.

---

## Runtime

- **~30 min** per ND2 file with 4 tiles (on 1 A100 GPU)
- **~1–2 hours** for larger ND2 files with 12–23 tiles
- Multiple files run in parallel as separate SLURM jobs

---

## First-Time Setup

You only need to do this once. If you're using the shared project at `/home/zhangdjr/projects/umich-fiber/pytorch_connectomics`, this is already done — skip to "How to Run" above.

### Clone the repo

```bash
git clone git@github.com:zhangdjr/pytorch_connectomics.git
cd pytorch_connectomics
```

### Create the conda environments

Two environments are needed (the pipeline switches between them automatically):

```bash
# Main environment (fiber segmentation + analysis)
conda env create -f environment.yml
conda activate pytc
pip install -e .

# Cell segmentation environment (micro-sam)
conda env create -f environment_microsam.yml
```

> **Why two envs?** micro-sam requires packages that conflict with the GPU version of PyTorch. The pipeline handles switching automatically — you don't need to worry about it.

### Verify the model checkpoint exists

```bash
ls outputs/fiber_retrain_all/20260311_223801/checkpoints/last.ckpt
```

If it's missing, copy it from the shared location:
```bash
cp /projects/weilab/zhangdjr/umich-fiber/pytorch_connectomics/outputs/fiber_retrain_all/20260311_223801/checkpoints/last.ckpt \
   outputs/fiber_retrain_all/20260311_223801/checkpoints/
```

---

## Troubleshooting

### How do I check if my job is still running?
```bash
squeue -u $USER
```
If it shows your job with state `R` (running) or `PD` (pending), it's still going.

### How do I see the job log?
```bash
cat logs/nd2_pipe_JOBID.out    # main output
cat logs/nd2_pipe_JOBID.err    # errors (if any)
```
Replace `JOBID` with the number printed when you ran `sbatch`.

### My job failed — what do I do?
Check the error log first:
```bash
cat logs/nd2_pipe_JOBID.err
```
Common issues:
- **Out of memory** — the ND2 file might be very large. Contact the pipeline maintainer.
- **File not found** — double-check that the ND2 file path is correct and you have read permission.

### I want to re-run a file from scratch
Delete the old results folder and re-submit:
```bash
rm -rf fiber_results/1-A1-2005/
sbatch run_nd2_pipeline.sh /path/to/1-A1-2005.nd2
```

### I only want to re-run part of the pipeline
The pipeline caches intermediate results. Delete the specific cache file to re-run that step:
```bash
# Re-run cell segmentation for tile A1
rm fiber_results/{nd2_name}/cache/A1_cell_seg.npz

# Re-run skeletonization for tile A1
rm fiber_results/{nd2_name}/cache/A1_skeletons.npz

# Then re-submit the job — it will skip completed steps and redo the deleted ones
sbatch run_nd2_pipeline.sh /path/to/your_file.nd2
```

---

## Viewing Results in 3D (Optional)

You can visually inspect the results using Neuroglancer (a browser-based 3D viewer):

```bash
conda activate pytc
python -i verify_1a12005.py
```

This prints a URL. To open it in your browser, set up an SSH tunnel from your **local machine** (not the cluster):
```bash
ssh -L 8889:localhost:8889 your_username@login.bc.edu
```
Then paste the URL into your browser.

**Navigation:** scroll = Z slices, Ctrl+scroll = zoom, click a fiber = highlight it.

---

## Advanced: Manual Step-by-Step

If you need to run individual steps separately (e.g., for debugging or custom workflows):

```bash
# 1. Extract tiles from ND2
conda activate pytc
python extract_nd2_tile.py --nd2 /path/to/file.nd2 --output /path/to/tiles --all-channels

# 2. Fiber segmentation inference (GPU)
python scripts/main.py --config /path/to/config.yaml --mode test \
    --checkpoint outputs/fiber_retrain_all/20260311_223801/checkpoints/last.ckpt

# 3. Cell segmentation (requires microsam env)
conda activate microsam
python cell_seg_microsam.py --tile-dir /path/to/tiles --output-dir /path/to/cache

# 4. Fiber analysis (per tile)
conda activate pytc
python fiber_pipeline.py --tile A1 --nd2-name my_sample \
    --tile-dir /path/to/tiles --pred-dir /path/to/fiber_seg --output-dir /path/to/output
```

---

## Advanced: Training / Fine-Tuning the Model

See `tutorials/fiber_retrain_all.yaml` for the full training config.

```bash
python scripts/main.py --config tutorials/fiber_retrain_all.yaml
```

- **Data format:** single-channel TIFF (raw) + integer TIFF (instance mask, 0 = background)
- **Training data:** `/projects/weilab/dataset/barcode/2026/umich/`

---

## Project Structure

```
run_nd2_pipeline.sh                # THE MAIN SCRIPT — processes one ND2 file end-to-end
fiber_pipeline.py                  # Analysis engine (skeletonize → signals → CSV)
cell_seg_microsam.py               # Cell body segmentation (micro-sam)
extract_nd2_tile.py                # ND2 tile/channel extraction
scripts/main.py                    # Deep learning inference entry point
connectomics/                      # Core library (model, data, inference)
tutorials/                         # YAML config files
slurm_jobs/                        # Additional SLURM scripts (legacy)
environment.yml                    # Conda env spec (pytc)
environment_microsam.yml           # Conda env spec (microsam)
```

### Model checkpoint
```
outputs/fiber_retrain_all/20260311_223801/checkpoints/last.ckpt
```
MedNeXt-S, 256 MB, trained on UMich hippocampus CA1 data at 162.9 nm XY / 0.4 µm Z resolution.
