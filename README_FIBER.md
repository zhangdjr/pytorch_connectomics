# Fiber Segmentation Pipeline

Fork of [PyTorch Connectomics](https://github.com/PytorchConnectomics/pytorch_connectomics) for **fiber instance segmentation** on confocal microscopy data (UMich barcode project).

**Pipeline:** Raw ND2/TIFF → deep learning inference → instance segmentation masks + CSV fiber coordinates.

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> pytorch_connectomics
cd pytorch_connectomics
conda env create -f environment.yml
conda activate pytc
pip install -e .

# 2. Copy the model checkpoint
mkdir -p checkpoints
cp /projects/weilab/zhangdjr/umich-fiber/pytorch_connectomics/outputs/fiber_retrain_all/20260311_223801/checkpoints/last.ckpt checkpoints/
```

---

## Running Inference

### On TIFF files

1. Copy and edit an existing config — update `test.data.test_image` with your file paths:
   - **Single TIFFs:** use `tutorials/fiber_retrain_all_infer_new.yaml` as a template
   - **ND2 tiles:** use `tutorials/fiber_nd2_all_tiles.yaml` as a template

2. Submit via SLURM (see `slurm_jobs/` for working examples):
   ```bash
   sbatch slurm_jobs/infer_nd2_all_tiles.sl
   ```
   Or run directly:
   ```bash
   export PYTHONPATH=/projects/weilab/weidf/lib/pytorch_connectomics/lib/MedNeXt:$(pwd):$PYTHONPATH
   python scripts/main.py --config tutorials/your_config.yaml --mode test --checkpoint checkpoints/last.ckpt
   ```

3. Results appear in `outputs/<experiment_name>/<timestamp>/results/`:
   - `*_prediction.h5` — Instance segmentation (HDF5)
   - `*_prediction.tiff` — Instance segmentation (TIFF)
   - Each voxel value = fiber instance ID (0 = background)

### On ND2 files

1. Extract tiles: `python extract_nd2_tile.py --nd2_path /path/to/file.nd2 --output_dir /path/to/nd2_tiles/ --channel 1`
2. Run inference as above using the extracted TIFFs
3. Generate CSV: `python generate_fiber_coordinates.py --tile_dir /path/to/nd2_tiles/ --pred_dir outputs/.../results/ --output_dir fiber_analysis/my_results/`

See `slurm_jobs/infer_nd2_all_tiles.sl` for a complete end-to-end example (extract → infer → CSV).

---

## Fiber Analysis Pipeline

After fiber segmentation inference, run the full analysis pipeline to get per-fiber CSV readouts (skeleton geometry, channel signals, cell association, validation).

### Prerequisites
Two conda environments are needed:
- **`pytc`** — main env for inference + pipeline (PyTorch, neuroglancer, scipy, optuna, etc.)
- **`microsam`** — separate env for micro-sam cell segmentation (can't coexist with pytc's GPU PyTorch)

```bash
# pytc env (main)
conda env create -f environment.yml
conda activate pytc
pip install -e .

# microsam env (cell segmentation only)
conda env create -f environment_microsam.yml
```

> **Why two envs?** micro-sam's conda-forge dependencies replace GPU PyTorch with CPU-only.
> The SLURM scripts handle env activation automatically (`cell_seg_microsam.sl` → microsam, `fiber_pipeline_*.sl` → pytc).

### Step 1: Extract tiles from ND2
```bash
conda activate pytc
python extract_nd2_tile.py --nd2_path /path/to/file.nd2 --output_dir /path/to/nd2_tiles/ --all_channels
```

### Step 2: Run fiber segmentation inference
```bash
sbatch slurm_jobs/infer_nd2_all_tiles.sl
```

### Step 3: Run micro-sam cell segmentation
```bash
# Edit cell_seg_microsam.py to set ND2_NAME and TILE_DIR, then:
sbatch slurm_jobs/cell_seg_microsam.sl
```
Outputs: `fiber_analysis/{nd2_name}/cache/{tile}_cell_seg.npz`

### Step 4: Run fiber analysis pipeline
```bash
# Single tile:
sbatch slurm_jobs/fiber_pipeline_a1.sl

# All 13 tiles in parallel (array job):
sbatch slurm_jobs/fiber_pipeline_all.sl
```
Outputs: `fiber_analysis/{nd2_name}/{nd2_name}_{tile}.csv`

### Step 5: Verify in Neuroglancer
```bash
conda activate pytc
python -i verify_a1.py   # or verify_a2.py
# Then SSH tunnel: ssh -L 8889:localhost:8889 user@login.bc.edu
```

### Output CSV Columns (31)

| Column | Description |
|--------|-------------|
| `fiber_id` | Fiber instance ID (from segmentation) |
| `nd2_name` / `tile_name` | Source tile identifiers |
| `is_valid` | Passes all validation filters |
| `parent_cell_id` | Cell label at soma (from micro-sam cell seg) |
| `fiber_length_um` | Skeleton arc length in µm |
| `pca_linearity` | PCA ratio (1.0 = perfectly linear) |
| `centroid_{z,y,x}_um` | Skeleton midpoint in µm |
| `{dapi,fiber,cfos,timestamp}_{mean,median,min,max,std}` | Per-channel signal stats along skeleton |
| `mean_soma_dapi` | Mean DAPI brightness at soma |

---

## Viewing Results

### Neuroglancer (3D viewer)

```bash
python neuroglancer_all_nd2_tiles.py   # Multi-tile ND2 viewer
python neuroglancer_all_volumes.py      # Single-volume viewer
python -i verify_a1.py                 # Single-tile QC (raw + seg + skeletons)
```

The scripts print a `localhost` URL. Since the viewer runs on the compute node, set up an SSH tunnel from your local machine:
```bash
ssh -L 8889:localhost:8889 your_username@login.bc.edu
```
Then open the printed URL in your browser.

**Tips:** Scroll = Z-slices, Ctrl+scroll = zoom, click fiber = highlight, `s` = toggle sidebar.

---

## Training / Fine-Tuning

See `tutorials/fiber_retrain_all.yaml` for the full training config.

```bash
# Train from scratch
python scripts/main.py --config tutorials/fiber_retrain_all.yaml

# Fine-tune from existing checkpoint
python scripts/main.py --config tutorials/your_config.yaml --checkpoint checkpoints/last.ckpt
```

- **Data format:** single-channel TIFF (raw) + integer TIFF (instance mask, 0 = background)
- **Training data:** `/projects/weilab/dataset/barcode/2026/umich/` (e.g., `0112-5-CA1-4_2.tif` + `_mask.tif`)

---

## Troubleshooting

### Stale cached predictions
The inference pipeline caches `*_prediction.h5` and `*_tta_prediction.h5` files. If you re-run with a different model/config, **delete the old cached files first** or inference will be skipped:
```bash
rm -f outputs/<experiment>/<timestamp>/results/<volume>_prediction.h5
rm -f outputs/<experiment>/<timestamp>/results/<volume>_tta_prediction.h5
```

### Stale pipeline caches
The analysis pipeline caches skeletons and cell seg in `fiber_analysis/{nd2_name}/cache/`. Delete relevant NPZ files if you want to re-run a step:
```bash
rm -f fiber_analysis/A1-2003/cache/A1_skeletons.npz   # re-run skeletonization
rm -f fiber_analysis/A1-2003/cache/A1_cell_seg.npz     # re-run cell seg
rm -f fiber_analysis/A1-2003/A1-2003_A1.csv            # re-generate CSV
```

### Few/no fibers detected
Your config is probably missing percentile clipping. The model was trained with clipping, so inference **must** match:
```yaml
test:
  data:
    image_transform:
      clip_percentile_low: 0.005
      clip_percentile_high: 0.995
```
All provided configs already include this — only an issue if you write a config from scratch.

### MedNeXt import error
Add to PYTHONPATH: `export PYTHONPATH=/projects/weilab/weidf/lib/pytorch_connectomics/lib/MedNeXt:$(pwd):$PYTHONPATH`
(Already included in all SLURM scripts.)

### Out of GPU memory
Reduce `inference.sliding_window.window_size` in your config (e.g., `[32, 128, 128]` instead of `[32, 256, 256]`).

---

## Project Structure

```
├── scripts/main.py                # Entry point for training and inference
├── connectomics/                  # Core library (model, data, inference, config)
├── tutorials/                     # YAML configs
├── slurm_jobs/                    # SLURM batch scripts for BC cluster
│   ├── cell_seg_microsam.sl       # Cell seg (microsam env, all tiles)
│   ├── fiber_pipeline_all.sl      # Analysis pipeline (array job, all tiles)
│   ├── fiber_pipeline_a1.sl       # Analysis pipeline (single tile)
│   └── infer_nd2_all_tiles.sl     # Fiber segmentation inference
├── fiber_pipeline.py              # Main analysis pipeline (skeletonize → signals → CSV)
├── cell_seg_microsam.py           # Standalone micro-sam cell segmentation
├── extract_nd2_tile.py            # Extract tiles from ND2 files
├── verify_a1.py / verify_a2.py   # Neuroglancer QC viewers
├── neuroglancer_all_nd2_tiles.py  # Multi-tile 3D viewer
├── neuroglancer_all_volumes.py    # Single-volume 3D viewer
├── generate_fiber_coordinates.py  # Legacy: segmentation masks → coordinate CSV
├── environment.yml                # Conda environment spec (pytc env)
└── environment_microsam.yml       # Conda environment spec (microsam env)
```

### Key configs

| Config | Purpose |
|--------|---------|
| `tutorials/fiber_nd2_all_tiles.yaml` | ND2 multi-tile inference |
| `tutorials/fiber_retrain_all_infer_new.yaml` | Standalone TIFF inference |
| `tutorials/fiber_retrain_all.yaml` | Model training |

### Model checkpoint
```
outputs/fiber_retrain_all/20260311_223801/checkpoints/last.ckpt
```
MedNeXt-S, 256 MB, trained on UMich hippocampus CA1 at 162.9nm XY / 0.4µm Z.
