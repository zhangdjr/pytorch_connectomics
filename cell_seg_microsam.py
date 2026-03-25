#!/usr/bin/env python3
"""
Standalone cell segmentation using micro-sam.
Runs in the 'microsam' conda env, outputs NPZ files consumed by fiber_pipeline.py.

Usage:
    conda activate microsam
    python cell_seg_microsam.py --tile A1 [--model_type vit_b_lm]

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


# ============================================================================
# Config
# ============================================================================

ND2_NAME = "A1-2003"
TILE_DIR = Path("/projects/weilab/dataset/barcode/2026/umich/nd2_tiles")
OUTPUT_DIR = Path(f"fiber_analysis/{ND2_NAME}/cache")

ALL_TILES = ["A1", "A2", "A3", "B4", "B3", "B2", "B1", "C1", "C2", "C3", "D2", "D1", "E1"]


def run_cell_seg(tile_name, model_type="vit_b_lm"):
    """Run micro-sam cell segmentation on DAPI channel for one tile."""

    dapi_path = TILE_DIR / f"{tile_name}_ch0_dapi.tif"
    output_path = OUTPUT_DIR / f"{tile_name}_cell_seg.npz"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    parser.add_argument("--tile", type=str, default="A1",
                        help="Tile name (e.g. A1) or 'all' for all tiles")
    parser.add_argument("--model_type", type=str, default="vit_b_lm",
                        help="micro-sam model type (default: vit_b_lm)")
    args = parser.parse_args()

    tiles = ALL_TILES if args.tile == "all" else [args.tile]

    for tile in tiles:
        print(f"\n{'='*60}")
        print(f"Cell Segmentation: {ND2_NAME} / {tile}")
        print(f"{'='*60}")
        run_cell_seg(tile, model_type=args.model_type)

    print("\nDone.")


if __name__ == "__main__":
    main()
