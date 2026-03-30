#!/usr/bin/env python3
"""
Neuroglancer viewer for all 13 ND2 tiles with raw images and fiber segmentations.
Displays tiles in their correct spatial positions using stage coordinates.

Usage:
    python tools/neuroglancer_all_nd2_tiles.py
    python tools/neuroglancer_all_nd2_tiles.py --tiles A1,A2,B1
    python tools/neuroglancer_all_nd2_tiles.py --tiles A1,A2,B1 --no-raw
    
Then open the printed URL in your browser (use SSH tunnel if remote).
"""

import argparse
import neuroglancer
import numpy as np
import tifffile
import json
from pathlib import Path

# Paths
tile_dir = Path('/projects/weilab/dataset/barcode/2026/umich/nd2_tiles')
# Current ND2 pipeline writes per-tile TIFF predictions here.
pred_dir = Path('last/results')

parser = argparse.ArgumentParser(description='Visualize ND2 tiles in Neuroglancer')
parser.add_argument(
    '--tiles',
    help='Comma-separated tile names to load, e.g. A1,A2,B1. Defaults to all tiles.',
)
parser.add_argument(
    '--no-raw',
    action='store_true',
    help='Do not load raw image layers; only load segmentation layers.',
)
args = parser.parse_args()

# Get all tiles
tile_files = sorted(tile_dir.glob('*_ch1.tif'))
if args.tiles:
    selected_tiles = {tile.strip() for tile in args.tiles.split(',') if tile.strip()}
    tile_files = [p for p in tile_files if p.stem.replace('_ch1', '') in selected_tiles]
    missing_tiles = sorted(selected_tiles - {p.stem.replace('_ch1', '') for p in tile_files})
    if missing_tiles:
        print(f"Warning: requested tiles not found: {', '.join(missing_tiles)}")

if not tile_files:
    raise SystemExit("No tiles selected for visualization.")

print(f"Found {len(tile_files)} tiles")

# Load all tiles and metadata
tiles_data = []
for tile_file in tile_files:
    tile_name = tile_file.stem.replace('_ch1', '')
    meta_file = tile_dir / f"{tile_name}_metadata.json"
    pred_candidates = [
        pred_dir / f"{tile_name}_ch1_prediction.tiff",
        pred_dir / f"{tile_name}_ch1_prediction_fixed.tiff",
    ]
    pred_file = next((p for p in pred_candidates if p.exists()), None)
    
    # Load metadata
    with open(meta_file) as f:
        metadata = json.load(f)
    
    # Load raw and prediction
    raw = None if args.no_raw else tifffile.imread(str(tile_file))
    if pred_file is not None:
        pred = tifffile.imread(str(pred_file))
    else:
        print(f"Warning: No prediction for {tile_name}")
        pred = None

    if raw is not None:
        volume_shape = raw.shape
    elif pred is not None:
        volume_shape = pred.shape
    else:
        with tifffile.TiffFile(str(tile_file)) as tif:
            volume_shape = tif.series[0].shape
    
    tiles_data.append({
        'name': tile_name,
        'raw': raw,
        'pred': pred,
        'stage_pos': metadata['stage_position_um'],
        'pixel_size': metadata['pixel_size_um'],
        'shape': volume_shape,
    })
    print(f"  {tile_name}: {volume_shape}, stage=({metadata['stage_position_um']['x']:.1f}, {metadata['stage_position_um']['y']:.1f}, {metadata['stage_position_um']['z']:.1f}) µm")

# Voxel sizes in µm
pixel_size = tiles_data[0]['pixel_size']  # Same for all tiles
voxel_size_um = [pixel_size['z'], pixel_size['y'], pixel_size['x']]  # Z, Y, X

print(f"\nVoxel size: {voxel_size_um[0]:.4f} × {voxel_size_um[1]:.4f} × {voxel_size_um[2]:.4f} µm")

# Compute relative offsets: subtract minimum stage position so tiles are near origin
min_x = min(t['stage_pos']['x'] for t in tiles_data)
min_y = min(t['stage_pos']['y'] for t in tiles_data)
min_z = min(t['stage_pos']['z'] for t in tiles_data)
print(f"Stage origin (subtracted): ({min_x:.2f}, {min_y:.2f}, {min_z:.2f}) µm")

for tile in tiles_data:
    # Relative offset in voxels
    rel_x_um = tile['stage_pos']['x'] - min_x
    rel_y_um = tile['stage_pos']['y'] - min_y
    rel_z_um = tile['stage_pos']['z'] - min_z
    tile['offset_voxels'] = [
        int(round(rel_z_um / voxel_size_um[0])),  # Z
        int(round(rel_y_um / voxel_size_um[1])),  # Y
        int(round(rel_x_um / voxel_size_um[2])),  # X
    ]
    print(f"  {tile['name']}: offset = {tile['offset_voxels']} voxels")

# Create Neuroglancer viewer
neuroglancer.set_server_bind_address('0.0.0.0', bind_port=8888)
viewer = neuroglancer.Viewer()

dims = neuroglancer.CoordinateSpace(
    names=['z', 'y', 'x'],
    units=['um', 'um', 'um'],
    scales=voxel_size_um,
)

with viewer.txn() as s:
    for tile in tiles_data:
        # Raw image layer
        if tile['raw'] is not None:
            raw_layer = neuroglancer.LocalVolume(
                data=tile['raw'],
                dimensions=dims,
                voxel_offset=tile['offset_voxels'],
            )
            s.layers[f'{tile["name"]}_raw'] = neuroglancer.ImageLayer(
                source=raw_layer,
            )

        # Segmentation layer
        if tile['pred'] is not None:
            seg_layer = neuroglancer.LocalVolume(
                data=tile['pred'].astype(np.uint32),
                dimensions=dims,
                voxel_offset=tile['offset_voxels'],
            )
            s.layers[f'{tile["name"]}_seg'] = neuroglancer.SegmentationLayer(
                source=seg_layer,
            )

    # Set initial view: center of all tiles, mid-Z
    tile_width = tiles_data[0]['shape'][2]   # X pixels
    tile_height = tiles_data[0]['shape'][1]  # Y pixels
    tile_depth = tiles_data[0]['shape'][0]   # Z slices

    all_cx = [t['offset_voxels'][2] + tile_width // 2 for t in tiles_data]
    all_cy = [t['offset_voxels'][1] + tile_height // 2 for t in tiles_data]
    center_z = tile_depth // 2
    center_y = (min(all_cy) + max(all_cy)) // 2
    center_x = (min(all_cx) + max(all_cx)) // 2

    s.position = [center_z, center_y, center_x]
    s.crossSectionScale = 1e-5  # Zoom in

print("\n" + "="*70)
print("NEUROGLANCER VIEWER READY")
print("="*70)

# Replace hostname with localhost for SSH tunnel (handles any hostname)
import re
viewer_url = str(viewer)
localhost_url = re.sub(r'http://[^:]+:8888/', 'http://localhost:8888/', viewer_url)

print(f"\nViewer URL (use this one):")
print(f"  {localhost_url}")
print("\nSSH tunnel command (run on your local machine):")
print(f"  ssh -L 8888:localhost:8888 zhangdjr@login.bc.edu")
print("\nThen open the URL above in your browser.")
print("\nPress Ctrl+C to exit.")
print("="*70)

# Keep server running
try:
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down viewer...")
