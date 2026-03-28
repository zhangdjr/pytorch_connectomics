#!/usr/bin/env python3
"""
Neuroglancer verification viewer for 1-A1-2000 tiles C1-C5.
Check whether C1/C2 truly have no fibers.

Usage:
    conda activate pytc
    python -i verify_1a12000.py
"""

import re
import time
import numpy as np
import tifffile
import neuroglancer
from pathlib import Path

# ============================================================================
# Config
# ============================================================================

ND2_NAME = "1-A1-2000"
TILES = ["C1", "C2", "C3", "C4", "C5"]

BASE_DIR = Path(f"/projects/weilab/dataset/barcode/2026/broad_dongqing/fiber_results/{ND2_NAME}")
TILE_DIR = BASE_DIR / "tiles"
FIBER_SEG_DIR = BASE_DIR / "fiber_seg"

ANISO_NM = [400.0, 162.9, 162.9]  # Z, Y, X

# ============================================================================
# Load data
# ============================================================================

print("Loading data...")

dims = neuroglancer.CoordinateSpace(
    names=["z", "y", "x"],
    units=["nm", "nm", "nm"],
    scales=ANISO_NM,
)

neuroglancer.set_server_bind_address("0.0.0.0", bind_port=8889)
viewer = neuroglancer.Viewer()

with viewer.txn() as s:
    for tile in TILES:
        # Raw fiber channel
        raw_path = TILE_DIR / f"{tile}_ch1.tif"
        if raw_path.exists():
            raw = tifffile.imread(str(raw_path))
            print(f"  {tile}_ch1: {raw.shape} [{raw.min()}-{raw.max()}] mean={raw.mean():.1f}")
            s.layers[f"{tile}_raw"] = neuroglancer.ImageLayer(
                source=neuroglancer.LocalVolume(data=raw, dimensions=dims),
                shader="""
void main() {
  float v = toNormalized(getDataValue());
  emitRGB(vec3(0.0, v*4.0, 0.0));
}""",
            )

        # Prediction
        pred_path = FIBER_SEG_DIR / f"{tile}_ch1_prediction.tiff"
        if pred_path.exists():
            pred = tifffile.imread(str(pred_path))
            n_inst = len(np.unique(pred)) - 1
            print(f"  {tile}_pred: {n_inst} instances")
            s.layers[f"{tile}_pred"] = neuroglancer.SegmentationLayer(
                source=neuroglancer.LocalVolume(
                    data=pred.astype(np.uint32), dimensions=dims
                ),
            )

    vol_shape = raw.shape
    s.position = [vol_shape[0] // 2, vol_shape[1] // 2, vol_shape[2] // 2]
    s.crossSectionScale = 1e-5
    s.layout = "xy"

# ============================================================================
# Print instructions
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
print("Toggle tiles on/off in left panel to compare C1 (0 fibers) vs C4/C5 (many fibers)")
print()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down viewer...")
