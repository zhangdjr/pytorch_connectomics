# Fiber Analysis Pipeline

Automated pipeline for analyzing fiber structures in confocal microscopy ND2 images. Takes a raw ND2 file and produces a CSV with measurements for every detected fiber.

## What Does This Pipeline Do?

Given an ND2 microscopy file, the pipeline automatically:

1. **Extracts** each tile and channel from the ND2 file
2. **Detects fibers** using a trained deep learning model (GPU)
3. **Detects cell bodies** using micro-sam segmentation
4. **Analyzes each fiber** — measures length, extracts signal intensities from all 4 channels (DAPI, fiber, cfos, timestamp), links fibers to their parent cell bodies, and filters out invalid detections
5. **Produces a CSV** with one row per fiber and 11 measurement columns

---

## Option A: Jupyter Notebook (Any Machine with GPU)

Use this if you **don't have BC cluster access**. Works on Google Colab, a local workstation, or any machine with an NVIDIA GPU.

1. Open **`fiber_analysis_pipeline.ipynb`** in Jupyter or upload it to [Google Colab](https://colab.research.google.com/)
2. In the **Configuration** cell, set your ND2 file path:
   ```python
   ND2_FILE = "/path/to/your/sample.nd2"
   ```
3. **Run all cells** top to bottom — everything installs and runs automatically
4. Results appear in `pytorch_connectomics/fiber_results/{nd2_name}/`

**Runtime:** ~30 min for a 4-tile ND2 on a single GPU. First run takes a few extra minutes to install packages and download the model (~267 MB).

On Colab, download results from the file browser (left sidebar) or:
```python
from google.colab import files
files.download(f"fiber_results/{ND2_BASENAME}/{ND2_BASENAME}_combined.csv")
```

---

## Option B: BC Cluster (SLURM)

```bash
# Log in
ssh your_username@a002.bc.edu
cd /home/zhangdjr/projects/umich-fiber/pytorch_connectomics

# Process one ND2 file
sbatch run_nd2_pipeline.sh /path/to/your_file.nd2

# Process many at once
for nd2 in /projects/weilab/dataset/barcode/2026/broad_dongqing/*.nd2; do
    sbatch run_nd2_pipeline.sh "$nd2"
done

# Check status
squeue -u $USER
```

Results are saved to `/projects/weilab/dataset/barcode/2026/broad_dongqing/fiber_results/{nd2_name}/`:
```
.../fiber_results/1-A1-2005/
├── 1-A1-2005_combined.csv           # <-- SUMMARY CSV (all tiles)
├── 1-A1-2005_combined_profiles.npz  # <-- INTENSITY PROFILES (all tiles)
├── 1-A1-2005_A1.csv                 # per-tile CSV
├── 1-A1-2005_A1_profiles.npz       # per-tile profiles
├── fiber_seg/                       # segmentation masks
└── cache/                           # intermediate files
```

**Runtime:** ~30 min per 4-tile ND2, ~1–2 hours for 12–23 tile files.

---

## What's in the CSV?

Each row is one detected fiber. Filter by `is_valid == True` for high-quality measurements only.

| Column | What it means |
|--------|---------------|
| `fiber_id` | Unique ID for each fiber |
| `nd2_name` | Source ND2 file name |
| `tile_name` | Which tile (e.g., A1, B5) |
| `is_valid` | `True` if passed quality filters |
| `parent_cell_id` | Which cell body it belongs to (0 = none found) |
| `fiber_length_um` | Length in micrometers |
| `pca_linearity` | How straight (1.0 = perfectly straight) |
| `centroid_z_um` | Fiber midpoint Z (µm) |
| `centroid_y_um` | Fiber midpoint Y (µm) |
| `centroid_x_um` | Fiber midpoint X (µm) |
| `mean_soma_dapi` | DAPI brightness at the cell body |

---

## Intensity Profiles (NPZ)

The pipeline also saves **1000-point intensity profiles** along each fiber's skeleton for all 4 channels.

```python
import numpy as np

data = np.load("/projects/weilab/dataset/barcode/2026/broad_dongqing/fiber_results/1-A1-2005/1-A1-2005_combined_profiles.npz")

data["fiber_ids"]   # (N,) fiber IDs
data["is_valid"]    # (N,) boolean
data["tile_names"]  # (N,) tile names
data["dapi"]        # (N, 1000) DAPI intensity along fiber
data["fiber"]       # (N, 1000) fiber channel
data["cfos"]        # (N, 1000) cfos channel
data["timestamp"]   # (N, 1000) timestamp channel

# Plot one fiber's cfos profile
import matplotlib.pyplot as plt
idx = np.where(data["fiber_ids"] == 42)[0][0]
plt.plot(data["cfos"][idx])
plt.xlabel("Position along fiber")
plt.ylabel("cfos intensity")
plt.show()
```

---

## Model Checkpoint

```
outputs/fiber_retrain_all/20260311_223801/checkpoints/last.ckpt
```
MedNeXt-S, 267 MB, trained on UMich hippocampus CA1 data at 162.9 nm XY / 0.4 µm Z resolution. Tracked via Git LFS — downloaded automatically when you clone the repo.
