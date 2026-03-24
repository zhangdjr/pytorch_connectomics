#!/usr/bin/env python3
"""
Post-process fiber segmentation predictions from all ND2 tiles:
  1. Fix z-padding bug (crop 128 → original Z slices)
  2. Compute per-fiber centroids using scipy
  3. Convert pixel coordinates to stage coordinates
  4. Verify coordinates automatically
  5. Generate per-tile CSVs and a combined master CSV

Usage:
    python generate_fiber_coordinates.py \
        --pred_dir outputs/fiber_retrain_all/20260311_223801/results \
        --meta_dir /projects/weilab/dataset/barcode/2026/umich/nd2_tiles \
        --output_dir fiber_analysis/nd2_all_tiles
"""

import argparse
import json
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from scipy.ndimage import center_of_mass, sum_labels


def fix_z_padding(pred, expected_z):
    """Crop z-padded prediction back to original depth."""
    if pred.shape[0] == expected_z:
        return pred
    if pred.shape[0] > expected_z:
        pad = (pred.shape[0] - expected_z) // 2
        return pred[pad:pad + expected_z]
    raise ValueError(f"Prediction Z={pred.shape[0]} < expected Z={expected_z}")


def process_tile(pred_path, metadata, output_dir):
    """
    Process a single tile: fix z-padding, compute centroids, generate CSV.

    Returns a DataFrame with fiber coordinates for this tile.
    """
    tile_name = metadata['tile_name']
    tile_idx = metadata['tile_idx']
    expected_z = metadata['n_slices']
    pixel_size = metadata['pixel_size_um']
    stage_pos = metadata['stage_position_um']

    # Load and fix z-padding
    pred = tifffile.imread(str(pred_path))
    original_z = pred.shape[0]
    pred = fix_z_padding(pred, expected_z)

    if original_z != expected_z:
        print(f"  Z-padding fixed: {original_z} → {expected_z}")

        # Save fixed prediction
        fixed_path = pred_path.parent / pred_path.name.replace('.tiff', '_fixed.tiff')
        tifffile.imwrite(str(fixed_path), pred, compression='zlib')
        print(f"  Saved fixed: {fixed_path.name}")

    # Get instance IDs
    instance_ids = np.unique(pred)
    instance_ids = instance_ids[instance_ids > 0]
    n_fibers = len(instance_ids)
    print(f"  Fibers: {n_fibers}")

    if n_fibers == 0:
        print(f"  ⚠ No fibers found in {tile_name}")
        return pd.DataFrame()

    # Compute centroids + voxel counts (scipy, fast)
    centroids = center_of_mass(pred > 0, pred, instance_ids)
    voxel_counts = sum_labels(np.ones_like(pred), pred, instance_ids)

    rows = []
    for i, fid in enumerate(instance_ids):
        cz, cy, cx = centroids[i]
        rows.append({
            'fiber_id': int(fid),
            'tile_id': tile_idx,
            'tile_name': tile_name,
            'instance_id': int(fid),
            'pixel_x': cx,
            'pixel_y': cy,
            'pixel_z': cz,
            'stage_x_um': stage_pos['x'] + cx * pixel_size['x'],
            'stage_y_um': stage_pos['y'] + cy * pixel_size['y'],
            'stage_z_um': stage_pos['z'] + cz * pixel_size['z'],
            'voxel_count': int(voxel_counts[i]),
        })

    df = pd.DataFrame(rows)

    # --- Verification: 5 random fibers ---
    n_check = min(5, len(df))
    sample = df.sample(n_check, random_state=42)
    passed = 0
    for _, row in sample.iterrows():
        fid = int(row['fiber_id'])
        px, py, pz = int(row['pixel_x']), int(row['pixel_y']), int(row['pixel_z'])
        if 0 <= pz < pred.shape[0] and 0 <= py < pred.shape[1] and 0 <= px < pred.shape[2]:
            if pred[pz, py, px] == fid:
                passed += 1
    print(f"  Verification: {passed}/{n_check} centroid checks passed")

    # --- Sanity check: stage coord ranges ---
    tile_width_um = pred.shape[2] * pixel_size['x']
    tile_height_um = pred.shape[1] * pixel_size['y']
    tile_depth_um = pred.shape[0] * pixel_size['z']
    x_range = df['stage_x_um'].max() - df['stage_x_um'].min()
    y_range = df['stage_y_um'].max() - df['stage_y_um'].min()
    z_range = df['stage_z_um'].max() - df['stage_z_um'].min()
    x_ok = x_range <= tile_width_um
    y_ok = y_range <= tile_height_um
    z_ok = z_range <= tile_depth_um
    print(f"  Stage range X: {x_range:.1f} µm (max {tile_width_um:.1f}) {'✓' if x_ok else '✗'}")
    print(f"  Stage range Y: {y_range:.1f} µm (max {tile_height_um:.1f}) {'✓' if y_ok else '✗'}")
    print(f"  Stage range Z: {z_range:.1f} µm (max {tile_depth_um:.1f}) {'✓' if z_ok else '✗'}")

    # Save per-tile CSV
    csv_path = output_dir / f"{tile_name}_fiber_coordinates.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path.name}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Generate fiber coordinate CSVs from predictions')
    parser.add_argument('--pred_dir', required=True, help='Directory containing prediction TIFFs')
    parser.add_argument('--meta_dir', required=True, help='Directory containing metadata JSONs')
    parser.add_argument('--output_dir', required=True, help='Output directory for CSVs')
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    meta_dir = Path(args.meta_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover tiles from metadata files
    meta_files = sorted(meta_dir.glob('*_metadata.json'))
    if not meta_files:
        print(f"ERROR: No metadata files found in {meta_dir}")
        return

    print(f"Found {len(meta_files)} tile metadata files\n")

    all_dfs = []
    for meta_file in meta_files:
        with open(meta_file) as f:
            metadata = json.load(f)

        tile_name = metadata['tile_name']

        # Find matching prediction file
        pred_path = pred_dir / f"{tile_name}_ch1_prediction.tiff"
        if not pred_path.exists():
            print(f"⚠ Skipping {tile_name}: prediction not found at {pred_path}")
            continue

        print(f"Processing {tile_name} (tile {metadata['tile_idx']})...")
        df = process_tile(pred_path, metadata, output_dir)
        if not df.empty:
            all_dfs.append(df)
        print()

    # Combine into master CSV with global fiber IDs
    if all_dfs:
        master = pd.concat(all_dfs, ignore_index=True)
        master['fiber_id'] = range(1, len(master) + 1)  # Global IDs

        master_path = output_dir / 'all_tiles_fiber_coordinates.csv'
        master.to_csv(master_path, index=False)

        print(f"{'='*60}")
        print(f"MASTER CSV COMPLETE")
        print(f"{'='*60}")
        print(f"Total fibers: {len(master)}")
        print(f"Tiles processed: {len(all_dfs)}")
        print(f"Saved: {master_path}")

        # Per-tile summary
        summary = master.groupby('tile_name').agg(
            fiber_count=('instance_id', 'count'),
            median_voxels=('voxel_count', 'median'),
        )
        print(f"\nPer-tile summary:")
        print(summary.to_string())
    else:
        print("ERROR: No tiles were processed successfully")


if __name__ == '__main__':
    main()
