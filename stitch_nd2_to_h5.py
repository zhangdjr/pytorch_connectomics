#!/usr/bin/env python3
"""
Stitch all ND2 tiles into a single global H5 volume for full-image inference.

Reads directly from the ND2 file (no need to pre-extract tiles).
Produces two outputs:
  - nd2_stitched.h5   : stitched image (uint16, chunked for sliding-window access)
  - nd2_mask.h5       : binary coverage mask (uint8, 1=tissue, 0=empty)

Usage:
    python stitch_nd2_to_h5.py
    python stitch_nd2_to_h5.py --nd2 /path/to/file.nd2 --output_dir /path/to/out/
"""

import argparse
import json
import numpy as np
import h5py
from pathlib import Path


def build_linear_weight(size_z, size_y, size_x, ramp_yx):
    """
    Build a per-tile weight map for linear blending in overlap zones.
    Ramp is applied at all 4 lateral (Y/X) edges; Z edges get full weight.
    """
    w = np.ones((size_z, size_y, size_x), dtype=np.float32)
    for i in range(ramp_yx):
        fade = (i + 1) / (ramp_yx + 1)
        w[:, i, :]    = np.minimum(w[:, i, :],    fade)
        w[:, -(i+1), :] = np.minimum(w[:, -(i+1), :], fade)
        w[:, :, i]    = np.minimum(w[:, :, i],    fade)
        w[:, :, -(i+1)] = np.minimum(w[:, :, -(i+1)], fade)
    return w


def stitch_from_nd2(nd2_path, output_dir, channel_idx=1):
    """
    Read ND2, stitch all tiles into a global volume with linear blending.

    Args:
        nd2_path:    Path to ND2 file
        output_dir:  Directory to save nd2_stitched.h5 and nd2_mask.h5
        channel_idx: Which channel to use (default: 1 = FITC/fibers)
    """
    import nd2
    import tifffile

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening ND2: {nd2_path}")
    with nd2.ND2File(nd2_path) as f:
        print(f"  Shape: {f.shape},  Sizes: {f.sizes}")

        n_pos   = f.sizes['P']
        n_z     = f.sizes['Z']
        voxel   = f.voxel_size()
        px_xy   = float(voxel.x)   # um/pixel (XY)
        px_z    = float(voxel.z)   # um/slice (Z)
        tile_y  = f.sizes['Y']
        tile_x  = f.sizes['X']

        print(f"  Voxel size: xy={px_xy:.4f} um,  z={px_z:.4f} um")
        print(f"  Tile size: {n_z} × {tile_y} × {tile_x}")

        # --- Read stage positions ---
        stage_positions = {}
        for loop in f.experiment:
            if loop.type == 'XYPosLoop':
                for i, pt in enumerate(loop.parameters.points):
                    stage_positions[i] = {
                        'name': pt.name,
                        'x': float(pt.stagePositionUm.x),
                        'y': float(pt.stagePositionUm.y),
                        'z': float(pt.stagePositionUm.z),
                    }
                break

        if not stage_positions:
            raise RuntimeError("No XYPosLoop found in ND2 metadata")

        print(f"\n  Stage positions for {n_pos} tiles:")
        for i, sp in stage_positions.items():
            print(f"    [{i:2d}] {sp['name']}: "
                  f"({sp['x']:.1f}, {sp['y']:.1f}, {sp['z']:.1f}) um")

        # --- Compute global bounding box ---
        # Stage position is the top-left corner of the tile (convention in ND2)
        min_x = min(sp['x'] for sp in stage_positions.values())
        min_y = min(sp['y'] for sp in stage_positions.values())
        min_z = min(sp['z'] for sp in stage_positions.values())

        # Compute pixel offsets for each tile (top-left corner in global canvas)
        offsets = {}   # tile_idx -> (oz, oy, ox)
        for i, sp in stage_positions.items():
            ox = round((sp['x'] - min_x) / px_xy)
            oy = round((sp['y'] - min_y) / px_xy)
            oz = round((sp['z'] - min_z) / px_z)
            offsets[i] = (oz, oy, ox)

        # Global canvas size
        max_oz = max(oz for oz, oy, ox in offsets.values())
        max_oy = max(oy for oz, oy, ox in offsets.values())
        max_ox = max(ox for oz, oy, ox in offsets.values())
        canvas_z = max_oz + n_z
        canvas_y = max_oy + tile_y
        canvas_x = max_ox + tile_x

        print(f"\n  Global canvas: {canvas_z} × {canvas_y} × {canvas_x} pixels")
        print(f"               = {canvas_z*px_z:.1f} × {canvas_y*px_xy:.1f} × {canvas_x*px_xy:.1f} um")
        mem_gb = canvas_z * canvas_y * canvas_x * 4 / 1024**3
        print(f"  Working memory (float32): {mem_gb:.1f} GB")

        # --- Allocate canvas ---
        canvas = np.zeros((canvas_z, canvas_y, canvas_x), dtype=np.float32)
        weight = np.zeros_like(canvas)

        # Overlap ramp = ~10% of tile width (matching the physical overlap between tiles)
        ramp_px = max(10, round(tile_x * 0.10))

        # --- Load ND2 array once and place tiles ---
        print(f"\n  Loading full ND2 array...")
        full_data = f.asarray()   # (P, Z, C, Y, X)
        print(f"  Array shape: {full_data.shape}")

        for i in range(n_pos):
            oz, oy, ox = offsets[i]
            tile = full_data[i, :, channel_idx, :, :].astype(np.float32)
            w = build_linear_weight(n_z, tile_y, tile_x, ramp_px)

            canvas[oz:oz+n_z, oy:oy+tile_y, ox:ox+tile_x] += tile * w
            weight[oz:oz+n_z, oy:oy+tile_y, ox:ox+tile_x] += w

            sp = stage_positions[i]
            print(f"  Placed tile {i:2d} ({sp['name']}):  offset=({oz},{oy},{ox})")

    # --- Normalize and save ---
    print(f"\n  Normalizing...")
    mask = weight > 0
    canvas[mask] /= weight[mask]
    stitched = canvas.astype(np.uint16)
    coverage_mask = mask.any(axis=0).astype(np.uint8)   # (Y, X) 2D coverage map

    # Chunk size aligned with sliding window for fast sequential access
    chunk_z = min(n_z, 32)
    chunk_yx = 256
    chunks = (chunk_z, chunk_yx, chunk_yx)

    out_h5 = output_dir / 'nd2_stitched.h5'
    print(f"  Saving stitched volume → {out_h5}")
    with h5py.File(out_h5, 'w') as f:
        f.create_dataset('main', data=stitched, chunks=chunks, compression='lzf')
        # Store voxel size as attributes for downstream use
        f['main'].attrs['voxel_size_um'] = [px_z, px_xy, px_xy]
        f['main'].attrs['stage_origin_um'] = [min_z, min_y, min_x]
        f['main'].attrs['px_xy_um'] = px_xy
        f['main'].attrs['px_z_um']  = px_z

    out_mask = output_dir / 'nd2_mask.h5'
    print(f"  Saving coverage mask  → {out_mask}")
    with h5py.File(out_mask, 'w') as f:
        f.create_dataset('main', data=coverage_mask,
                         chunks=(chunk_yx, chunk_yx), compression='lzf')

    # Also save per-tile metadata JSON (needed by generate_fiber_coordinates.py)
    meta_dir = output_dir / 'tile_metadata'
    meta_dir.mkdir(exist_ok=True)
    for i, sp in stage_positions.items():
        meta = {
            'tile_idx': i,
            'tile_name': sp['name'],
            'shape': [n_z, tile_y, tile_x],
            'n_slices': n_z,
            'pixel_size_um': {'x': px_xy, 'y': px_xy, 'z': px_z},
            'stage_position_um': {'x': sp['x'], 'y': sp['y'], 'z': sp['z']},
            'global_offset_px': {'z': offsets[i][0], 'y': offsets[i][1], 'x': offsets[i][2]},
        }
        json_path = meta_dir / f"{sp['name']}_metadata.json"
        with open(json_path, 'w') as jf:
            json.dump(meta, jf, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Stitched: {out_h5}   shape={stitched.shape}")
    print(f"  Mask:     {out_mask}  shape={coverage_mask.shape}")
    print(f"  Metadata: {meta_dir}/")
    print(f"{'='*60}")

    return out_h5, out_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stitch ND2 tiles into single H5 volume')
    parser.add_argument('--nd2', default='/projects/weilab/dataset/barcode/2026/umich/A1-2003.nd2',
                        help='Path to ND2 file')
    parser.add_argument('--output_dir', default='/projects/weilab/dataset/barcode/2026/umich/',
                        help='Output directory')
    parser.add_argument('--channel', type=int, default=1,
                        help='Channel index (default: 1 = FITC/fibers)')
    args = parser.parse_args()

    stitch_from_nd2(args.nd2, args.output_dir, args.channel)
