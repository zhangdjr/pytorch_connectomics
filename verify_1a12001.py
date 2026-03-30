#!/usr/bin/env python3
"""
Neuroglancer verification viewer for 1-A1-2001.
Loads raw fiber channel directly from original ND2 + fiber_seg + CSV/skeletons.

Usage:
    conda activate pytc
    python -i verify_1a12001.py
"""

import re
import csv
import time
import numpy as np
import nd2
import tifffile
import neuroglancer
from pathlib import Path

# ============================================================================
# Config — change TILE to view a different tile
# ============================================================================

ND2_NAME = "1-A1-2001"
ND2_PATH = f"/projects/weilab/dataset/barcode/2026/broad_dongqing/{ND2_NAME}.nd2"
TILE = "E1"

BASE_DIR = Path(f"/projects/weilab/dataset/barcode/2026/broad_dongqing/fiber_results/{ND2_NAME}")
FIBER_SEG_DIR = BASE_DIR / "fiber_seg"
CACHE_DIR = BASE_DIR / "cache"

ANISO_NM = [400.0, 162.9, 162.9]  # Z, Y, X
FIBER_CH = 1  # Ch1 = 488nm fiber channel

# ============================================================================
# Load data
# ============================================================================

print("Loading data...")

# --- Find tile index from ND2 metadata ---
print(f"  Opening ND2: {ND2_PATH}")
with nd2.ND2File(ND2_PATH) as f:
    tile_info = {}
    for loop in f.experiment:
        if loop.type == 'XYPosLoop':
            for i, point in enumerate(loop.parameters.points):
                tile_info[point.name] = i
            break

    if TILE not in tile_info:
        print(f"  Available tiles: {list(tile_info.keys())}")
        raise ValueError(f"Tile '{TILE}' not found in ND2")

    tile_idx = tile_info[TILE]
    full_data = f.asarray()  # (P, Z, C, Y, X)
    raw_fiber = full_data[tile_idx, :, FIBER_CH, :, :]  # (Z, Y, X)
    print(f"  {TILE} fiber ch: {raw_fiber.shape} [{raw_fiber.min()}-{raw_fiber.max()}] mean={raw_fiber.mean():.1f}")

vol_shape = raw_fiber.shape

# --- Fiber segmentation ---
fiber_seg_path = FIBER_SEG_DIR / f"{TILE}_ch1_prediction.tiff"
fiber_seg = tifffile.imread(str(fiber_seg_path))
print(f"  fiber_seg: {fiber_seg.shape}, {len(np.unique(fiber_seg))-1} instances")

# --- Skeletons ---
skel_path = CACHE_DIR / f"{TILE}_skeletons.npz"
if skel_path.exists():
    skel_data = np.load(str(skel_path), allow_pickle=True)
    fiber_ids = skel_data["fiber_ids"]
    centerlines = skel_data["centerlines"]
    print(f"  skeletons: {len(fiber_ids)} fibers")
else:
    fiber_ids, centerlines = np.array([]), []
    print("  skeletons: not found")

# --- CSV results ---
csv_path = BASE_DIR / f"{ND2_NAME}_{TILE}.csv"
if csv_path.exists():
    with open(csv_path) as f:
        csv_rows = list(csv.DictReader(f))
    valid_ids = set(int(r["fiber_id"]) for r in csv_rows if r["is_valid"] == "True")
    invalid_ids = set(int(r["fiber_id"]) for r in csv_rows if r["is_valid"] == "False")
    print(f"  CSV: {len(csv_rows)} rows, {len(valid_ids)} valid, {len(invalid_ids)} invalid")
else:
    valid_ids = set(int(fid) for fid in fiber_ids)
    invalid_ids = set()
    print("  CSV: not found, treating all as valid")

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

FIBER_SHADER = """
void main() {
  float v = toNormalized(getDataValue());
  emitRGB(vec3(0.0, v*4.0, 0.0));
}"""

# --- Rasterize skeletons ---
valid_skel_vol = np.zeros(vol_shape, dtype=np.uint32)
invalid_skel_vol = np.zeros(vol_shape, dtype=np.uint32)

if len(fiber_ids) > 0:
    print("Rasterizing skeletons...")
    for fid, cl in zip(fiber_ids, centerlines):
        pts_voxel = cl / np.array(ANISO_NM)
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

# --- Add layers ---
with viewer.txn() as s:
    s.layers["fiber_raw"] = neuroglancer.ImageLayer(
        source=neuroglancer.LocalVolume(data=raw_fiber, dimensions=dims),
        shader=FIBER_SHADER,
    )

    s.layers["fiber_seg"] = neuroglancer.SegmentationLayer(
        source=neuroglancer.LocalVolume(
            data=fiber_seg.astype(np.uint32), dimensions=dims
        ),
    )

    if np.count_nonzero(valid_skel_vol) > 0:
        s.layers["valid_skeletons"] = neuroglancer.SegmentationLayer(
            source=neuroglancer.LocalVolume(
                data=valid_skel_vol, dimensions=dims
            ),
        )

    if np.count_nonzero(invalid_skel_vol) > 0:
        s.layers["invalid_skeletons"] = neuroglancer.SegmentationLayer(
            source=neuroglancer.LocalVolume(
                data=invalid_skel_vol, dimensions=dims
            ),
        )

    s.position = [vol_shape[0] // 2, vol_shape[1] // 2, vol_shape[2] // 2]
    s.crossSectionScale = 1e-5
    s.layout = "xy"

# ============================================================================
# Print instructions
# ============================================================================

viewer_url = str(viewer)
localhost_url = re.sub(r"http://[^:]+:8889/", "http://localhost:8889/", viewer_url)

print("\n" + "=" * 70)
print(f"VERIFICATION VIEWER: {ND2_NAME} / {TILE}")
print("=" * 70)
print(f"\nViewer URL:")
print(f"  {localhost_url}")
print(f"\nSSH tunnel (run on your local machine):")
print(f"  ssh -L 8889:localhost:8889 zhangdjr@login.bc.edu")
print()
print("LAYERS:")
print("  fiber_raw        — raw fiber channel (488nm, from ND2)")
print("  fiber_seg        — fiber instance segmentation")
print("  valid_skeletons  — valid fiber skeletons (≥8µm)")
print("  invalid_skeletons— short/invalid fiber skeletons")
print()
print("Press Ctrl+C to exit when done.")
print("=" * 70)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down viewer...")
