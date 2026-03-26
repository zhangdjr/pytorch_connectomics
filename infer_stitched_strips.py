#!/usr/bin/env python3
"""
Memory-efficient strip-based inference on the stitched ND2 volume.

The full stitched volume (56 x 10625 x 8538) requires ~57 GB GPU output buffer,
which exceeds the 47 GB GPU. This script splits along the X-axis into strips
of ~2000 px (fits in ~23 GB GPU), accumulates raw predictions on CPU/disk,
blends overlaps, then decodes instances once from the full prediction.

Usage:
    python infer_stitched_strips.py \\
        --stitched /projects/weilab/dataset/barcode/2026/umich/nd2_stitched.h5 \\
        --checkpoint checkpoints/last.ckpt \\
        --config tutorials/fiber_nd2_stitched.yaml \\
        --output_dir outputs/fiber_nd2_stitched/results
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch
import h5py
from omegaconf import OmegaConf

# ── Config ──────────────────────────────────────────────────────────────────
STRIP_WIDTH = 2000          # X-pixels per strip (controls GPU memory)
OVERLAP     = 300           # X-pixel overlap between adjacent strips
ROI_SIZE    = (32, 256, 256)
SW_OVERLAP  = 0.5
SW_BATCH    = 2
EMPTY_THRESH = 0.02


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_model(cfg, checkpoint_path: str, device: torch.device):
    """Build ConnectomicsModule and load checkpoint weights."""
    from connectomics.training.lit.model import ConnectomicsModule

    module = ConnectomicsModule(cfg)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    module.load_state_dict(state, strict=True)
    module.eval()
    module.to(device)
    print(f"  Loaded checkpoint: {checkpoint_path}")
    return module


def apply_activations(pred: torch.Tensor) -> torch.Tensor:
    """Apply sigmoid to channels 0-1, tanh to channel 2 (same as YAML config)."""
    out = pred.clone()
    out[:, 0:2] = torch.sigmoid(pred[:, 0:2])
    out[:, 2:3] = torch.tanh(pred[:, 2:3])
    return out


def run_strip_inference(module, strip_tensor: torch.Tensor, device: torch.device,
                        empty_thresh: float = EMPTY_THRESH) -> np.ndarray:
    """
    Run sliding-window inference on a single strip.

    Args:
        module:       ConnectomicsModule (eval mode, on device)
        strip_tensor: (1, 1, Z, Y, X_strip) float32 on CPU
        device:       GPU device

    Returns:
        pred_np: (3, Z, Y, X_strip) float32 numpy, on CPU
    """
    from monai.inferers import SlidingWindowInferer

    inferer = SlidingWindowInferer(
        roi_size=ROI_SIZE,
        sw_batch_size=SW_BATCH,
        overlap=SW_OVERLAP,
        mode='gaussian',
        sigma_scale=0.25,
        padding_mode='reflect',
    )

    net = module.model   # MedNeXtWrapper

    def forward_fn(x: torch.Tensor) -> torch.Tensor:
        """Move patch to GPU, run model, return on CPU (keeps output buffer on CPU)."""
        x_gpu = x.to(device)
        with torch.no_grad():
            # EmptyPatchSkipWrapper: check if patch is empty first
            if x_gpu.abs().max().item() < empty_thresh:
                B, C, Z, Y, X = x_gpu.shape
                return torch.zeros(B, 3, Z, Y, X, dtype=x_gpu.dtype, device='cpu')
            out = net(x_gpu)           # (B, 3, Z, Y, X) on GPU
            return out.cpu()            # accumulate on CPU

    # Run inferer with CPU input so output accumulates on CPU
    strip_cpu = strip_tensor.float()   # keep on CPU
    with torch.no_grad():
        pred = inferer(strip_cpu, forward_fn)   # (1, 3, Z, Y, X_strip) on CPU

    # Apply activations (sigmoid / tanh)
    pred = apply_activations(pred[0])   # (3, Z, Y, X_strip)
    return pred.numpy()


def load_config(config_path: str):
    """Load Hydra-style YAML config via OmegaConf."""
    from connectomics.config import load_config as _load_cfg
    return _load_cfg(config_path)


# ── Main ─────────────────────────────────────────────────────────────────────

def infer_strips(stitched_h5: str, checkpoint: str, config_path: str,
                 output_dir: str, strip_width: int = STRIP_WIDTH,
                 overlap: int = OVERLAP):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load volume shape ────────────────────────────────────────────────────
    print(f"\nOpening stitched volume: {stitched_h5}")
    with h5py.File(stitched_h5, 'r') as f:
        vol_shape = f['main'].shape           # (Z, Y, X)
        voxel_size = f['main'].attrs.get('voxel_size_um', [0.4, 0.163, 0.163])
    Z, Y, X = vol_shape
    print(f"  Volume shape: {Z} x {Y} x {X}")

    # ── Compute strip ranges ─────────────────────────────────────────────────
    strips = []
    x_start = 0
    while x_start < X:
        x_end = min(x_start + strip_width, X)
        strips.append((x_start, x_end))
        if x_end == X:
            break
        x_start = x_end - overlap

    print(f"  Strips ({len(strips)} total, width={strip_width}, overlap={overlap}):")
    for i, (xs, xe) in enumerate(strips):
        print(f"    [{i}] X=[{xs}, {xe}]  width={xe-xs}")

    # ── Allocate CPU prediction buffer (float16 to save RAM) ─────────────────
    pred_buf  = np.zeros((3, Z, Y, X), dtype=np.float16)
    weight_buf = np.zeros((X,), dtype=np.float32)   # per-X-column blending weight

    # ── Load config & model ──────────────────────────────────────────────────
    cfg = load_config(config_path)
    print("\nLoading model & checkpoint...")
    module = load_model(cfg, checkpoint, device)

    # ── Process each strip ───────────────────────────────────────────────────
    for strip_idx, (x0, x1) in enumerate(strips):
        print(f"\n{'='*60}")
        print(f"Strip {strip_idx+1}/{len(strips)}:  X=[{x0}, {x1}]")
        print(f"{'='*60}")

        # Read strip from disk
        with h5py.File(stitched_h5, 'r') as f:
            strip_data = f['main'][:, :, x0:x1].astype(np.float32)   # (Z, Y, X_w)

        # Normalize 0-1 with percentile clipping (same as YAML config)
        lo = np.percentile(strip_data[strip_data > 0], 0.5) if strip_data.max() > 0 else 0
        hi = np.percentile(strip_data, 99.5)
        if hi > lo:
            strip_data = np.clip(strip_data, lo, hi)
            strip_data = (strip_data - lo) / (hi - lo)
        else:
            strip_data = np.zeros_like(strip_data)

        # Shape: (1, 1, Z, Y, X_w)
        strip_tensor = torch.from_numpy(strip_data[None, None])
        x_width = x1 - x0
        print(f"  Strip tensor shape: {list(strip_tensor.shape)}")
        print(f"  GPU memory before inference: "
              f"{torch.cuda.memory_allocated()/1e9:.1f} GB")

        pred_strip = run_strip_inference(module, strip_tensor, device)
        # pred_strip: (3, Z, Y, x_width)

        print(f"  Strip inference done. pred shape: {pred_strip.shape}")
        print(f"  GPU memory after inference: "
              f"{torch.cuda.memory_allocated()/1e9:.1f} GB")

        # Build linear blend weight for this strip (ramp at edges)
        w = np.ones(x_width, dtype=np.float32)
        ramp = min(overlap // 2, x_width // 4)
        for i in range(ramp):
            fade = (i + 1) / (ramp + 1)
            w[i]            = min(w[i],            fade)
            w[x_width-1-i]  = min(w[x_width-1-i],  fade)

        # Accumulate (weighted sum, then normalize)
        pred_buf[:, :, :, x0:x1]  += (pred_strip * w[None, None, None, :]).astype(np.float16)
        weight_buf[x0:x1]          += w

        del strip_tensor, pred_strip
        torch.cuda.empty_cache()

    # ── Normalize blended predictions ─────────────────────────────────────────
    print("\nNormalizing blended predictions...")
    mask = weight_buf > 0
    pred_buf[:, :, :, mask] /= weight_buf[None, None, None, mask].astype(np.float16)

    # Save raw prediction (float16 HDF5)
    pred_h5_path = output_dir / 'nd2_stitched_tta_prediction.h5'
    print(f"Saving raw predictions → {pred_h5_path}")
    with h5py.File(pred_h5_path, 'w') as f:
        f.create_dataset('main', data=pred_buf,
                         chunks=(1, min(Z, 32), min(Y, 256), min(X, 256)),
                         compression='lzf')

    # ── Decode instances in strips (avoids full-volume float64 OOM) ──────────
    import gc
    del pred_buf
    gc.collect()

    seg_h5_path = decode_in_strips(
        str(pred_h5_path), config_path, str(output_dir),
        strip_width=strip_width, overlap=overlap
    )

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Raw predictions: {pred_h5_path}")
    print(f"  Segmentation:    {seg_h5_path}")
    print(f"{'='*60}")

    return seg_h5_path


def stitch_instances(seg_left: np.ndarray, seg_right: np.ndarray,
                     overlap: int, global_id_offset: int,
                     iou_threshold: float = 0.3) -> tuple:
    """
    Stitch two adjacent X-strip segmentations by matching instances in the overlap zone.

    Instances in the right strip that overlap sufficiently with a left-strip instance
    are remapped to the same ID. Unmatched right instances get new global IDs.

    Returns:
        (seg_right_remapped, new_global_offset)
    """
    # overlap zone in local coordinates
    left_zone  = seg_left[:,  :, -overlap:]   # (Z, Y, overlap)
    right_zone = seg_right[:, :, :overlap]    # (Z, Y, overlap)

    left_ids  = np.unique(left_zone[left_zone  > 0])
    right_ids = np.unique(right_zone[right_zone > 0])

    # Build mapping: right_id → left_id (if matched) or new global id
    remap = {}
    used_left = set()

    for rid in right_ids:
        r_mask = (right_zone == rid)
        r_vol  = r_mask.sum()
        if r_vol == 0:
            continue

        best_iou  = 0.0
        best_lid  = -1
        for lid in left_ids:
            if lid in used_left:
                continue
            l_mask    = (left_zone  == lid)
            intersect = (r_mask & l_mask).sum()
            if intersect == 0:
                continue
            union = (r_mask | l_mask).sum()
            iou   = intersect / union
            if iou > best_iou:
                best_iou = iou
                best_lid = lid

        if best_iou >= iou_threshold and best_lid > 0:
            remap[rid] = best_lid
            used_left.add(best_lid)
        else:
            global_id_offset += 1
            remap[rid] = global_id_offset

    # Remap all right-strip IDs (including those not in overlap)
    right_only_ids = np.unique(seg_right[seg_right > 0])
    seg_remapped = seg_right.copy()
    for rid in right_only_ids:
        if rid in remap:
            seg_remapped[seg_right == rid] = remap[rid]
        else:
            global_id_offset += 1
            seg_remapped[seg_right == rid] = global_id_offset

    print(f"    Matched {sum(1 for v in remap.values() if v <= (global_id_offset - len(right_only_ids) + len(remap)))} / "
          f"{len(right_ids)} right-strip instances in overlap zone")

    return seg_remapped, global_id_offset


def decode_in_strips(pred_h5: str, config_path: str, output_dir: str,
                     strip_width: int = STRIP_WIDTH, overlap: int = OVERLAP):
    """
    Decode raw predictions from H5 in X-strips and stitch instances at boundaries.

    Each strip is decoded independently (fits in RAM), then instances in the
    overlap zones are matched and remapped to consistent global IDs.
    The final segmentation is written to a pre-allocated output H5.
    """
    import gc
    output_dir = Path(output_dir)
    pred_h5_path = Path(pred_h5)
    cfg = load_config(config_path)

    from connectomics.inference.io import apply_decode_mode, apply_postprocessing

    # Get volume shape
    with h5py.File(pred_h5_path, 'r') as f:
        C, Z, Y, X = f['main'].shape
    print(f"\nPrediction shape: {C} x {Z} x {Y} x {X}")

    # Compute strip ranges (same logic as inference)
    strips = []
    x_start = 0
    while x_start < X:
        x_end = min(x_start + strip_width, X)
        strips.append((x_start, x_end))
        if x_end == X:
            break
        x_start = x_end - overlap

    print(f"Decoding {len(strips)} strips (width={strip_width}, overlap={overlap}):")
    for i, (xs, xe) in enumerate(strips):
        print(f"  [{i}] X=[{xs}, {xe}]")

    # Pre-allocate output segmentation on disk
    seg_h5_path = output_dir / 'nd2_stitched_prediction.h5'
    print(f"\nPre-allocating segmentation H5: {seg_h5_path}")
    with h5py.File(seg_h5_path, 'w') as f:
        f.create_dataset('main',
                         shape=(Z, Y, X),
                         dtype=np.uint32,
                         chunks=(min(Z, 32), min(Y, 256), min(X, 256)),
                         compression='lzf')

    global_id_offset = 0
    prev_seg = None   # right portion of previous strip's segmentation

    for strip_idx, (x0, x1) in enumerate(strips):
        x_width = x1 - x0
        print(f"\n{'='*60}")
        print(f"Decoding strip {strip_idx+1}/{len(strips)}: X=[{x0},{x1}]")
        print(f"{'='*60}")

        # Load prediction strip (float32)
        with h5py.File(pred_h5_path, 'r') as f:
            pred_strip = f['main'][:, :, :, x0:x1].astype(np.float32)  # (C, Z, Y, w)

        # Decode this strip
        seg_strip = apply_decode_mode(cfg, pred_strip)   # (Z, Y, w)
        del pred_strip
        gc.collect()

        # Convert to uint32
        seg_strip = seg_strip.astype(np.uint32)
        n_local = int(seg_strip.max())
        print(f"  Local instances: {n_local}")

        if strip_idx == 0:
            # First strip: assign global IDs starting from 1
            seg_global = seg_strip.copy()
            ids = np.unique(seg_global[seg_global > 0])
            for lid in ids:
                global_id_offset += 1
                seg_global[seg_strip == lid] = global_id_offset
        else:
            # Subsequent strips: stitch with previous strip in overlap zone
            print(f"  Stitching with previous strip (overlap={overlap} px)...")
            seg_global, global_id_offset = stitch_instances(
                prev_seg, seg_strip, overlap, global_id_offset
            )

        # Write the non-overlap portion to output H5
        # (for first strip: write everything; otherwise skip the left overlap zone
        #  which is already covered by the previous strip)
        if strip_idx == 0:
            write_x0, write_x1 = 0, x1 - overlap if len(strips) > 1 else x1
            local_x0 = 0
        elif strip_idx == len(strips) - 1:
            write_x0, write_x1 = x0 + overlap, x1
            local_x0 = overlap
        else:
            write_x0, write_x1 = x0 + overlap, x1 - overlap
            local_x0 = overlap

        write_width = write_x1 - write_x0
        if write_width > 0:
            with h5py.File(seg_h5_path, 'a') as f:
                f['main'][:, :, write_x0:write_x1] = \
                    seg_global[:, :, local_x0:local_x0 + write_width]
            print(f"  Written X=[{write_x0},{write_x1}]")

        # Keep the right overlap portion for next strip stitching
        prev_seg = seg_global[:, :, -overlap:] if strip_idx < len(strips) - 1 else None
        del seg_strip, seg_global
        gc.collect()

    total_instances = global_id_offset
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Total instances: {total_instances}")
    print(f"  Segmentation:    {seg_h5_path}")
    print(f"{'='*60}")
    return seg_h5_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Strip-based inference on stitched ND2 volume')
    parser.add_argument('--stitched',
                        default='/projects/weilab/dataset/barcode/2026/umich/nd2_stitched.h5')
    parser.add_argument('--checkpoint', default='checkpoints/last.ckpt')
    parser.add_argument('--config', default='tutorials/fiber_nd2_stitched.yaml')
    parser.add_argument('--output_dir',
                        default='outputs/fiber_nd2_stitched/results')
    parser.add_argument('--strip_width', type=int, default=STRIP_WIDTH,
                        help='X-pixels per strip (default: 2000, uses ~23 GB GPU)')
    parser.add_argument('--overlap', type=int, default=OVERLAP,
                        help='Overlap pixels between strips (default: 300)')
    parser.add_argument('--decode-only', action='store_true', dest='decode_only',
                        help='Skip inference, decode from existing raw prediction H5')
    parser.add_argument('--pred-h5', type=str, dest='pred_h5', default=None,
                        help='Path to existing raw prediction H5 (used with --decode-only)')
    args = parser.parse_args()

    if args.decode_only:
        pred_h5 = args.pred_h5 or str(Path(args.output_dir) / 'nd2_stitched_tta_prediction.h5')
        decode_in_strips(pred_h5, args.config, args.output_dir,
                         strip_width=args.strip_width, overlap=args.overlap)
    else:
        infer_strips(
            stitched_h5=args.stitched,
            checkpoint=args.checkpoint,
            config_path=args.config,
            output_dir=args.output_dir,
            strip_width=args.strip_width,
            overlap=args.overlap,
        )
