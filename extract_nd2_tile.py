#!/usr/bin/env python3
"""
Extract all tiles (Ch1 FITC) from ND2 file with metadata.
Loads the ND2 file once and extracts all tiles in a single pass.

Usage:
    python extract_nd2_tile.py                    # Extract all tiles
    python extract_nd2_tile.py --tiles 0 3 5      # Extract specific tiles
"""

import nd2
import numpy as np
import tifffile
import json
import argparse
from pathlib import Path


def extract_all_tiles(nd2_path, output_dir, tile_indices=None):
    """
    Extract Ch1 (FITC/fiber channel) from all tiles in one pass.
    
    Args:
        nd2_path: Path to ND2 file
        output_dir: Output directory for TIFFs and metadata JSONs
        tile_indices: List of tile indices to extract (None = all)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening ND2 file: {nd2_path}")
    with nd2.ND2File(nd2_path) as f:
        # --- File structure ---
        print(f"\nFile structure:")
        print(f"  Shape: {f.shape}")
        print(f"  Dimensions: {f.sizes}")

        n_positions = f.sizes['P']
        n_z = f.sizes['Z']
        n_channels = f.sizes['C']

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
            raise RuntimeError("Could not find XYPosLoop in ND2 experiment metadata")

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
        full_data = f.asarray()  # Shape: (P, Z, C, Y, X)
        print(f"  Array shape: {full_data.shape}, dtype: {full_data.dtype}")

        # --- Extract each tile ---
        channel_idx = 1  # Ch1 = FITC (fibers)
        results = []

        for tile_idx in tile_indices:
            info = tile_info[tile_idx]
            tile_name = info['name']

            print(f"\n{'='*60}")
            print(f"Tile {tile_idx}: {tile_name}")
            print(f"  Stage: ({info['stage_x']:.2f}, {info['stage_y']:.2f}, {info['stage_z']:.2f}) µm")

            volume = full_data[tile_idx, :, channel_idx, :, :]
            print(f"  Shape: {volume.shape}, range: [{volume.min()}, {volume.max()}]")

            # Save TIFF
            tiff_path = output_dir / f"{tile_name}_ch1.tif"
            tifffile.imwrite(str(tiff_path), volume, compression='zlib')
            print(f"  Saved: {tiff_path}")

            # Save metadata JSON
            metadata = {
                'tile_idx': tile_idx,
                'tile_name': tile_name,
                'shape': list(volume.shape),
                'n_slices': n_z,
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
            print(f"  Saved: {json_path}")

            results.append((tile_name, str(tiff_path), str(json_path)))

        # --- Summary ---
        print(f"\n{'='*60}")
        print(f"EXTRACTION COMPLETE: {len(results)} tiles")
        print(f"{'='*60}")
        for name, tiff, meta in results:
            print(f"  {name}: {tiff}")

        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract tiles from ND2 file')
    parser.add_argument('--nd2', default='/projects/weilab/dataset/barcode/2026/umich/A1-2003.nd2',
                        help='Path to ND2 file')
    parser.add_argument('--output', default='/projects/weilab/dataset/barcode/2026/umich/nd2_tiles',
                        help='Output directory')
    parser.add_argument('--tiles', nargs='*', type=int, default=None,
                        help='Tile indices to extract (default: all)')
    args = parser.parse_args()

    extract_all_tiles(args.nd2, args.output, args.tiles)
