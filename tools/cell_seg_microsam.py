#!/usr/bin/env python3
"""
Standalone cell segmentation using micro-sam.
Runs in the 'microsam' conda env, outputs NPZ files consumed by fiber_pipeline.py.

Usage:
    conda activate microsam
    python tools/cell_seg_microsam.py --tile-dir /path/to/tiles --output-dir /path/to/cache
    python tools/cell_seg_microsam.py --tile-dir /path/to/tiles --output-dir /path/to/cache --tile A1

Or via SLURM (see slurm_jobs/cell_seg_microsam.sl).
"""

import argparse
import os
import numpy as np
import tifffile
from pathlib import Path

from micro_sam.automatic_segmentation import (
    get_predictor_and_segmenter,
    automatic_instance_segmentation,
)


def detect_tiles(tile_dir):
    """Auto-detect tile names from DAPI files in the tile directory."""
    tile_dir = Path(tile_dir)
    tiles = sorted(set(
        p.name.replace("_ch0_dapi.tif", "")
        for p in tile_dir.glob("*_ch0_dapi.tif")
    ))
    return tiles


def run_cell_seg(tile_name, tile_dir, output_dir, model_type="vit_b_lm"):
    """Run micro-sam cell segmentation on DAPI channel for one tile."""
    tile_dir = Path(tile_dir)
    output_dir = Path(output_dir)

    dapi_path = tile_dir / f"{tile_name}_ch0_dapi.tif"
    output_path = output_dir / f"{tile_name}_cell_seg.npz"

    os.makedirs(output_dir, exist_ok=True)

    if output_path.exists():
        print(f"  Cached: {output_path}")
        return

    print(f"Loading DAPI: {dapi_path}")
    dapi = tifffile.imread(str(dapi_path))  # (Z, Y, X) uint16
    print(f"  Shape: {dapi.shape}, dtype: {dapi.dtype}, range: [{dapi.min()}-{dapi.max()}]")

    print(f"Running micro-sam ({model_type}) on full 3D volume...")
    predictor, segmenter = get_predictor_and_segmenter(model_type)
    seg = automatic_instance_segmentation(
        predictor, segmenter, input_path=dapi, verbose=True
    )

    seg = seg.astype(np.int32)
    n_labels = len(np.unique(seg)) - 1
    print(f"  Segmentation: {seg.shape}, {n_labels} cell instances")

    np.savez_compressed(str(output_path), cell_seg=seg)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="micro-sam cell segmentation")
    parser.add_argument("--tile", type=str, default="all",
                        help="Tile name (e.g. A1) or 'all' for all tiles")
    parser.add_argument("--tile-dir", type=str,
                        default="/projects/weilab/dataset/barcode/2026/umich/nd2_tiles",
                        help="Directory containing extracted tile TIFFs")
    parser.add_argument("--output-dir", type=str,
                        default="fiber_analysis/A1-2003/cache",
                        help="Output directory for cell seg NPZ files")
    parser.add_argument("--model_type", type=str, default="vit_b_lm",
                        help="micro-sam model type (default: vit_b_lm)")
    args = parser.parse_args()

    if args.tile == "all":
        tiles = detect_tiles(args.tile_dir)
        if not tiles:
            print(f"ERROR: No DAPI tiles found in {args.tile_dir}")
            return
        print(f"Auto-detected {len(tiles)} tiles: {', '.join(tiles)}")
    else:
        tiles = [args.tile]

    for tile in tiles:
        print(f"\n{'='*60}")
        print(f"Cell Segmentation: {tile}")
        print(f"{'='*60}")
        run_cell_seg(tile, args.tile_dir, args.output_dir, model_type=args.model_type)

    print("\nDone.")


if __name__ == "__main__":
    main()
