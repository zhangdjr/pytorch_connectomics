"""Resample resolution-mismatched volumes to training resolution (162.9nm XY, 400nm Z)."""
import numpy as np
import tifffile
from scipy.ndimage import zoom
import os

INPUT_DIR = "/projects/weilab/dataset/barcode/2026/umich/processed_ch1"
OUTPUT_DIR = "/projects/weilab/dataset/barcode/2026/umich/processed_ch1_resampled"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training resolution
TARGET_XY = 162.9  # nm
TARGET_Z = 400.0   # nm

# Volumes that need resampling (source_xy_nm, source_z_nm)
volumes = {
    "0203-5-40x-001_ch1.tif": {"xy": 108.6, "z": 400.0},   # 1.5x zoom, Z matches
    "0203-5-60x_ch1.tif":     {"xy": 72.2,  "z": 300.0},    # 60x, both mismatch
    "0203-5-60x001_ch1.tif":  {"xy": 108.3, "z": 300.0},    # 60x+1.5x zoom, both mismatch
}

for fname, res in volumes.items():
    print(f"\n{'='*60}")
    print(f"Processing: {fname}")
    print(f"{'='*60}")
    
    path = os.path.join(INPUT_DIR, fname)
    vol = tifffile.imread(path)
    print(f"  Input shape: {vol.shape}, dtype: {vol.dtype}")
    print(f"  Source resolution: XY={res['xy']}nm, Z={res['z']}nm")
    print(f"  Target resolution: XY={TARGET_XY}nm, Z={TARGET_Z}nm")
    
    # Compute zoom factors (source/target because we want fewer pixels at coarser resolution)
    zoom_z = res["z"] / TARGET_Z
    zoom_xy = res["xy"] / TARGET_XY
    
    print(f"  Zoom factors: Z={zoom_z:.4f}, Y={zoom_xy:.4f}, X={zoom_xy:.4f}")
    
    new_shape = (
        int(round(vol.shape[0] * zoom_z)),
        int(round(vol.shape[1] * zoom_xy)),
        int(round(vol.shape[2] * zoom_xy)),
    )
    print(f"  Output shape: {new_shape}")
    
    # Resample with order=1 (bilinear) to avoid ringing artifacts
    resampled = zoom(vol.astype(np.float32), (zoom_z, zoom_xy, zoom_xy), order=1)
    
    # Clip and convert back to uint16
    resampled = np.clip(resampled, 0, 65535).astype(np.uint16)
    print(f"  Resampled shape: {resampled.shape}, dtype: {resampled.dtype}")
    print(f"  Stats: min={resampled.min()}, max={resampled.max()}, mean={resampled.mean():.1f}")
    
    out_path = os.path.join(OUTPUT_DIR, fname.replace("_ch1.tif", "_ch1_resampled.tif"))
    tifffile.imwrite(out_path, resampled)
    print(f"  Saved: {out_path}")
    
    del vol, resampled

print(f"\n✅ All resampled volumes saved to: {OUTPUT_DIR}")
