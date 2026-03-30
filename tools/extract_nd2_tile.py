#!/usr/bin/env python3
"""
Extract tiles from ND2 file with metadata.
Loads the ND2 file once and extracts all tiles in a single pass.

Usage:
    python tools/extract_nd2_tile.py --nd2 /path/to/file.nd2 --output /path/to/tiles/
    python tools/extract_nd2_tile.py --nd2 /path/to/file.nd2 --output /path/to/tiles/ --all-channels
    python tools/extract_nd2_tile.py --tiles 0 3 5      # Extract specific tiles
"""

import nd2
import numpy as np
import tifffile
import json
import argparse
from pathlib import Path


# Channel naming convention
CHANNEL_SUFFIXES = {
    0: "ch0_dapi",
    1: "ch1",
    2: "ch2_cfos",
    3: "ch3_timestamp",
}


def extract_all_tiles(nd2_path, output_dir, tile_indices=None, all_channels=False):
    """
    Extract tiles from ND2 file.
    
    Args:
        nd2_path: Path to ND2 file
        output_dir: Output directory for TIFFs and metadata JSONs
        tile_indices: List of tile indices to extract (None = all)
        all_channels: If True, extract all channels. If False, only Ch1 (fiber).
    
    Returns:
        List of tile names extracted.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening ND2 file: {nd2_path}")
    with nd2.ND2File(nd2_path) as f:
        # --- File structure ---
        print(f"\nFile structure:")
        print(f"  Shape: {f.shape}")
        print(f"  Dimensions: {f.sizes}")

        sizes = dict(f.sizes)
        has_positions_axis = "P" in sizes
        n_positions = int(sizes.get("P", 1))
        n_z = int(sizes.get("Z", 1))
        n_channels = int(sizes.get("C", 1))

        # --- Voxel size (from ND2 metadata) ---
        voxel_size = f.voxel_size()
        # voxel_size() returns (z, y, x) in µm
        pixel_size_um = {
            'x': float(voxel_size.x),
            'y': float(voxel_size.y),
            'z': float(voxel_size.z),
        }
        print(f"  Pixel size: x={pixel_size_um['x']:.4f}, y={pixel_size_um['y']:.4f}, z={pixel_size_um['z']:.4f} µm")

        # --- Stage positions (from experiment loops) ---
        tile_info = {}
        for loop in f.experiment:
            if loop.type == 'XYPosLoop':
                for i, point in enumerate(loop.parameters.points):
                    tile_info[i] = {
                        'name': point.name,
                        'stage_x': float(point.stagePositionUm.x),
                        'stage_y': float(point.stagePositionUm.y),
                        'stage_z': float(point.stagePositionUm.z),
                    }
                break  # Only need the first XYPosLoop

        if not tile_info:
            # Some ND2 files contain a single FOV and may not provide XYPosLoop.
            if n_positions == 1:
                tile_info[0] = {
                    "name": "P0",
                    "stage_x": 0.0,
                    "stage_y": 0.0,
                    "stage_z": 0.0,
                }
                print("\nNo XYPosLoop metadata found; using single-tile fallback name 'P0'.")
            else:
                raise RuntimeError("Could not find XYPosLoop in ND2 experiment metadata")

        # Ensure tile metadata covers all positions.
        if len(tile_info) < n_positions:
            for i in range(len(tile_info), n_positions):
                tile_info[i] = {
                    "name": f"P{i}",
                    "stage_x": 0.0,
                    "stage_y": 0.0,
                    "stage_z": 0.0,
                }

        print(f"\nFound {len(tile_info)} tile positions:")
        for i, info in tile_info.items():
            print(f"  Tile {i} ({info['name']}): ({info['stage_x']:.2f}, {info['stage_y']:.2f}, {info['stage_z']:.2f}) µm")

        # --- Determine which tiles to extract ---
        if tile_indices is None:
            tile_indices = list(range(n_positions))
        for idx in tile_indices:
            if idx >= n_positions:
                raise ValueError(f"Tile {idx} out of range (0-{n_positions-1})")

        # --- Load full array once ---
        print(f"\nLoading full ND2 array into memory...")
        full_data = f.asarray()
        if not has_positions_axis:
            # ND2 without explicit position axis: treat as a single-tile file.
            full_data = np.expand_dims(full_data, axis=0)
        # Expected canonical shape here: (P, Z, C, Y, X)
        if full_data.ndim != 5:
            raise RuntimeError(
                f"Unexpected ND2 array shape after normalization: {full_data.shape}. "
                "Expected 5D shape (P, Z, C, Y, X)."
            )
        print(f"  Array shape: {full_data.shape}, dtype: {full_data.dtype}")

        # --- Determine channels to extract ---
        if all_channels:
            channels_to_extract = list(range(n_channels))
        else:
            channels_to_extract = [1]  # Ch1 = FITC (fibers) only

        # --- Extract each tile ---
        tile_names = []

        for tile_idx in tile_indices:
            info = tile_info[tile_idx]
            tile_name = info['name']
            tile_names.append(tile_name)

            print(f"\n{'='*60}")
            print(f"Tile {tile_idx}: {tile_name}")
            print(f"  Stage: ({info['stage_x']:.2f}, {info['stage_y']:.2f}, {info['stage_z']:.2f}) µm")

            for ch_idx in channels_to_extract:
                volume = full_data[tile_idx, :, ch_idx, :, :]
                suffix = CHANNEL_SUFFIXES.get(ch_idx, f"ch{ch_idx}")
                tiff_path = output_dir / f"{tile_name}_{suffix}.tif"
                tifffile.imwrite(str(tiff_path), volume, compression='zlib')
                print(f"  Ch{ch_idx} ({suffix}): {volume.shape}, [{volume.min()}-{volume.max()}] → {tiff_path}")

            # Save metadata JSON
            metadata = {
                'tile_idx': tile_idx,
                'tile_name': tile_name,
                'shape': list(full_data[tile_idx, :, 0, :, :].shape),
                'n_slices': n_z,
                'n_channels': n_channels,
                'pixel_size_um': pixel_size_um,
                'stage_position_um': {
                    'x': info['stage_x'],
                    'y': info['stage_y'],
                    'z': info['stage_z'],
                },
            }
            json_path = output_dir / f"{tile_name}_metadata.json"
            with open(str(json_path), 'w') as jf:
                json.dump(metadata, jf, indent=2)

        # --- Summary ---
        print(f"\n{'='*60}")
        print(f"EXTRACTION COMPLETE: {len(tile_names)} tiles, {len(channels_to_extract)} channels each")
        print(f"{'='*60}")
        for name in tile_names:
            print(f"  {name}")

        return tile_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract tiles from ND2 file')
    parser.add_argument('--nd2', default='/projects/weilab/dataset/barcode/2026/umich/A1-2003.nd2',
                        help='Path to ND2 file')
    parser.add_argument('--output', default='/projects/weilab/dataset/barcode/2026/umich/nd2_tiles',
                        help='Output directory')
    parser.add_argument('--tiles', nargs='*', type=int, default=None,
                        help='Tile indices to extract (default: all)')
    parser.add_argument('--all-channels', action='store_true',
                        help='Extract all channels (default: Ch1 fiber only)')
    args = parser.parse_args()

    extract_all_tiles(args.nd2, args.output, args.tiles, all_channels=args.all_channels)
