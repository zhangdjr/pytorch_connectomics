#!/usr/bin/env python3
"""
Neuroglancer verification viewer for A1 tile fiber pipeline output.

Displays:
  - 4 raw channels (DAPI, fiber, cfos, timestamp) as image layers
  - Fiber segmentation as a segmentation overlay
  - Cell segmentation (micro-sam) as a segmentation overlay
  - Fiber skeleton centerlines rasterized as segmentation volumes (visible in 2D)

Usage:
    conda activate pytc
    python -i verify_a1.py

Then open the printed URL in your browser via SSH tunnel.
"""

import re
import csv
import time
import json
import numpy as np
import tifffile
import neuroglancer
from pathlib import Path

# ============================================================================
# Config
# ============================================================================

TILE = "A1"
ND2_NAME = "A1-2003"

TILE_DIR = Path("/projects/weilab/dataset/barcode/2026/umich/nd2_tiles")
PRED_DIR = Path("outputs/fiber_retrain_all/20260311_223801/results")
ANALYSIS_DIR = Path(f"fiber_analysis/{ND2_NAME}")

ANISO_NM = [400.0, 162.9, 162.9]  # Z, Y, X

# ============================================================================
# Load data
# ============================================================================

print("Loading data...")

# Raw channels
ch_files = {
    "dapi":      TILE_DIR / f"{TILE}_ch0_dapi.tif",
    "fiber":     TILE_DIR / f"{TILE}_ch1.tif",
    "cfos":      TILE_DIR / f"{TILE}_ch2_cfos.tif",
    "timestamp": TILE_DIR / f"{TILE}_ch3_timestamp.tif",
}

raw = {}
for name, path in ch_files.items():
    raw[name] = tifffile.imread(str(path))
    print(f"  {name}: {raw[name].shape} [{raw[name].min()}-{raw[name].max()}]")

vol_shape = raw["fiber"].shape  # (Z, Y, X)

# Fiber segmentation
fiber_seg = tifffile.imread(str(PRED_DIR / f"{TILE}_ch1_prediction_fixed.tiff"))
print(f"  fiber_seg: {fiber_seg.shape}, {len(np.unique(fiber_seg))-1} instances")

# Cell segmentation
cell_seg_data = np.load(str(ANALYSIS_DIR / "cache" / f"{TILE}_cell_seg.npz"))
cell_seg = cell_seg_data["cell_seg"]
print(f"  cell_seg: {cell_seg.shape}, max_label={cell_seg.max()}")

# Skeletons
skel_data = np.load(str(ANALYSIS_DIR / "cache" / f"{TILE}_skeletons.npz"), allow_pickle=True)
fiber_ids = skel_data["fiber_ids"]
centerlines = skel_data["centerlines"]  # each (1000, 3) in nm, [Z, Y, X]
print(f"  skeletons: {len(fiber_ids)} fibers")

# CSV results (optional — may not exist yet if pipeline hasn't re-run)
csv_path = ANALYSIS_DIR / f"{ND2_NAME}_{TILE}.csv"
if csv_path.exists():
    with open(csv_path) as f:
        csv_rows = list(csv.DictReader(f))
    valid_ids = set(int(r["fiber_id"]) for r in csv_rows if r["is_valid"] == "True")
    invalid_ids = set(int(r["fiber_id"]) for r in csv_rows if r["is_valid"] == "False")
    print(f"  CSV: {len(csv_rows)} rows, {len(valid_ids)} valid, {len(invalid_ids)} invalid")
else:
    print(f"  CSV not found (pipeline not yet re-run), showing ALL skeletons as valid")
    valid_ids = set(int(fid) for fid in fiber_ids)
    invalid_ids = set()

# ============================================================================
# Build Neuroglancer viewer
# ============================================================================

print("\nSetting up Neuroglancer...")
neuroglancer.set_server_bind_address("0.0.0.0", bind_port=8889)
viewer = neuroglancer.Viewer()

dims = neuroglancer.CoordinateSpace(
    names=["z", "y", "x"],
    units=["nm", "nm", "nm"],
    scales=ANISO_NM,
)

# Color shaders for each channel
SHADERS = {
    "dapi": """
void main() {
  float v = toNormalized(getDataValue());
  emitRGB(vec3(0.3*v, 0.3*v, v*5.0));
}""",
    "fiber": """
void main() {
  float v = toNormalized(getDataValue());
  emitRGB(vec3(0.0, v*4.0, 0.0));
}""",
    "cfos": """
void main() {
  float v = toNormalized(getDataValue());
  emitRGB(vec3(v*5.0, v*2.0, 0.0));
}""",
    "timestamp": """
void main() {
  float v = toNormalized(getDataValue());
  emitRGB(vec3(v*5.0, 0.0, v*5.0));
}""",
}

# --- Rasterize skeletons into label volumes (visible in 2D cross-section) ---
print("Rasterizing skeletons into label volumes...")

valid_skel_vol = np.zeros(vol_shape, dtype=np.uint32)
invalid_skel_vol = np.zeros(vol_shape, dtype=np.uint32)

for fid, cl in zip(fiber_ids, centerlines):
    # cl is (1000, 3) in nm [Z, Y, X] — convert to voxel
    pts_voxel = cl / np.array(ANISO_NM)

    # Use every 5th point for reasonable density on each slice
    pts_sub = pts_voxel[::5]

    zi = np.clip(np.round(pts_sub[:, 0]).astype(int), 0, vol_shape[0] - 1)
    yi = np.clip(np.round(pts_sub[:, 1]).astype(int), 0, vol_shape[1] - 1)
    xi = np.clip(np.round(pts_sub[:, 2]).astype(int), 0, vol_shape[2] - 1)

    if int(fid) in valid_ids:
        valid_skel_vol[zi, yi, xi] = int(fid)
    else:
        invalid_skel_vol[zi, yi, xi] = int(fid)

print(f"  Valid skeleton voxels: {np.count_nonzero(valid_skel_vol)}")
print(f"  Invalid skeleton voxels: {np.count_nonzero(invalid_skel_vol)}")

# ============================================================================
# Add layers
# ============================================================================

with viewer.txn() as s:
    # --- Raw channel layers ---
    for ch_name, vol in raw.items():
        s.layers[ch_name] = neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(data=vol, dimensions=dims),
            shader=SHADERS.get(ch_name, ""),
        )

    # --- Fiber segmentation ---
    s.layers["fiber_seg"] = neuroglancer.SegmentationLayer(
        source=neuroglancer.LocalVolume(
            data=fiber_seg.astype(np.uint32), dimensions=dims
        ),
    )

    # --- Cell segmentation ---
    s.layers["cell_seg"] = neuroglancer.SegmentationLayer(
        source=neuroglancer.LocalVolume(
            data=cell_seg.astype(np.uint32), dimensions=dims
        ),
    )

    # --- Valid skeletons (rasterized — visible in 2D cross-section) ---
    s.layers["valid_skeletons"] = neuroglancer.SegmentationLayer(
        source=neuroglancer.LocalVolume(
            data=valid_skel_vol, dimensions=dims
        ),
    )

    # --- Invalid skeletons (rasterized) ---
    s.layers["invalid_skeletons"] = neuroglancer.SegmentationLayer(
        source=neuroglancer.LocalVolume(
            data=invalid_skel_vol, dimensions=dims
        ),
    )

    # Initial view: center of volume, mid-Z
    s.position = [
        vol_shape[0] // 2,
        vol_shape[1] // 2,
        vol_shape[2] // 2,
    ]
    s.crossSectionScale = 1e-5
    s.layout = "xy"

# ============================================================================
# Print URL and instructions
# ============================================================================

viewer_url = str(viewer)
localhost_url = re.sub(r"http://[^:]+:8889/", "http://localhost:8889/", viewer_url)

print("\n" + "=" * 70)
print("VERIFICATION VIEWER READY")
print("=" * 70)
print(f"\nViewer URL:")
print(f"  {localhost_url}")
print(f"\nSSH tunnel (run on your local machine):")
print(f"  ssh -L 8889:localhost:8889 zhangdjr@login.bc.edu")
print()
print("=" * 70)
print("LAYERS (toggle with checkboxes in the sidebar):")
print("=" * 70)
print("  fiber            — raw fiber channel (Ch1, 488nm)")
print("  dapi             — raw DAPI/nissl channel (Ch0, 405nm)")
print("  cfos             — raw cfos channel (Ch2, 561nm)")
print("  timestamp        — raw timestamp channel (Ch3, 647nm)")
print("  fiber_seg        — fiber instance segmentation")
print("  cell_seg         — micro-sam cell body segmentation (3D-consistent)")
print("  valid_skeletons  — rasterized skeleton voxels (valid fibers)")
print("  invalid_skeletons— rasterized skeleton voxels (invalid fibers)")
print()
print("  Skeleton labels match fiber_seg labels — hover to see fiber ID.")
print()
print("=" * 70)
print("VERIFICATION CHECKLIST:")
print("=" * 70)
print("""
1. FIBER SEGMENTATION vs RAW
   - Toggle 'fiber' + 'fiber_seg' on, others off
   - Scroll through Z slices — do boundaries match visible fibers?

2. SKELETON FIT
   - Toggle 'fiber' + 'valid_skeletons' on
   - Skeleton dots should appear inside fiber cross-sections on each slice
   - Toggle 'invalid_skeletons' — these should be short fragments

3. CELL SEGMENTATION (3D-consistent)
   - Toggle 'dapi' + 'cell_seg' on, others off
   - Same cell should keep the SAME COLOR across Z slices
   - Cell outlines should match bright DAPI regions

4. FIBER-CELL ASSOCIATION
   - Toggle 'fiber_seg' + 'cell_seg' + 'valid_skeletons'
   - Find a skeleton dot inside a cell body
   - Hover fiber_seg for fiber_id, hover cell_seg for cell_id

5. CHANNEL SIGNALS
   - Toggle 'timestamp' (purple) — should look symmetric along fibers
   - Toggle 'cfos' (orange) — signal varies along fibers

Press Ctrl+C to exit when done.
""")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down viewer...")
