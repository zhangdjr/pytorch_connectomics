#!/usr/bin/env python3
"""
View multi-region fiber predictions with semi-transparent overlays.
Run this after inference completes to verify predictions.

Usage: python view_multiregion_predictions.py
"""
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import os
import glob

volumes = {
    'A1-2002': {
        'raw': '/projects/weilab/dataset/barcode/2026/umich/processed_ch1/A1-2002_ch1.tif',
        'label': 'A1-2002 (77 slices)',
    },
    'A1-2003_s01': {
        'raw': '/projects/weilab/dataset/barcode/2026/umich/processed_ch1/A1-2003_s01_ch1.tif',
        'label': 'A1-2003 Series 01 (56 slices)',
    },
    'A1-2003_s02': {
        'raw': '/projects/weilab/dataset/barcode/2026/umich/processed_ch1/A1-2003_s02_ch1.tif',
        'label': 'A1-2003 Series 02 (56 slices)',
    },
    'A1-2007': {
        'raw': '/projects/weilab/dataset/barcode/2026/umich/processed_ch1/A1-2007_ch1.tif',
        'label': 'A1-2007 (79 slices)',
    },
}

results_dir = 'outputs/fiber_retrain_all/20260311_223801/results'
out_dir = 'fiber_analysis/multiregion_overlays'
os.makedirs(out_dir, exist_ok=True)


def load_tiff(path):
    frames = []
    with Image.open(path) as img:
        for i in range(img.n_frames):
            img.seek(i)
            frames.append(np.array(img))
    return np.stack(frames, axis=0)


def instance_colormap(seg):
    """Create random but consistent colors for each instance."""
    ids = np.unique(seg)
    ids = ids[ids > 0]
    colors = np.zeros((*seg.shape, 3), dtype=np.float32)
    for i, uid in enumerate(ids):
        hue = (i * 0.618033988749895) % 1.0  # golden ratio for spread
        rgb = hsv_to_rgb([hue, 0.8, 0.9])
        colors[seg == uid] = rgb
    return colors


def check_z_alignment(raw, pred, name):
    """Check if prediction z-dimension matches raw, fix if needed."""
    if raw.shape[0] == pred.shape[0]:
        print(f"  ✅ {name}: z-dimensions match ({raw.shape[0]} slices)")
        return pred
    
    print(f"  ⚠️  {name}: z-mismatch! raw={raw.shape[0]}, pred={pred.shape[0]}")
    
    # Find first/last non-empty slices
    non_empty = [z for z in range(pred.shape[0]) if pred[z].max() > 0]
    if not non_empty:
        print(f"  ❌ Prediction is all zeros!")
        return pred[:raw.shape[0]]
    
    first, last = non_empty[0], non_empty[-1]
    actual_slices = last - first + 1
    print(f"  Non-empty range: [{first}, {last}] ({actual_slices} slices)")
    
    if actual_slices == raw.shape[0]:
        print(f"  Cropping prediction[{first}:{last+1}] to fix alignment")
        return pred[first:last+1]
    else:
        print(f"  ⚠️  Cannot auto-fix, returning center crop")
        offset = (pred.shape[0] - raw.shape[0]) // 2
        return pred[offset:offset + raw.shape[0]]


print("="*60)
print("Multi-Region Fiber Prediction Viewer")
print("="*60)

# Find prediction files
pred_files = glob.glob(os.path.join(results_dir, 'A1-*_prediction.tiff'))
if not pred_files:
    print(f"\n❌ No prediction files found in {results_dir}/")
    print("Make sure inference has completed. Check logs/ for status.")
    exit(1)

print(f"\nFound {len(pred_files)} prediction files:")
for f in sorted(pred_files):
    print(f"  {os.path.basename(f)}")

for name, info in volumes.items():
    # Find matching prediction file
    pred_candidates = [f for f in pred_files if name.replace('_s', '_s') in os.path.basename(f).replace('-', '_').replace(' ', '_')]
    if not pred_candidates:
        # Try simpler matching
        pred_candidates = glob.glob(os.path.join(results_dir, f'{name}*_prediction.tiff'))
    if not pred_candidates:
        # Try even simpler - just the base name
        pred_candidates = glob.glob(os.path.join(results_dir, f'*{name.split("_")[0]}*_prediction.tiff'))
    
    if not pred_candidates:
        print(f"\n⚠️  No prediction found for {name}, skipping...")
        continue
    
    pred_path = pred_candidates[0]
    print(f"\n{'='*60}")
    print(f"Volume: {info['label']}")
    print(f"  Raw: {info['raw']}")
    print(f"  Pred: {pred_path}")
    
    raw = load_tiff(info['raw'])
    pred = load_tiff(pred_path)
    
    print(f"  Raw shape: {raw.shape}, Pred shape: {pred.shape}")
    
    # Check and fix z-alignment
    pred = check_z_alignment(raw, pred, name)
    
    n_instances = len(np.unique(pred)) - 1  # exclude 0
    print(f"  Total unique instances: {n_instances}")
    
    # Pick representative z-slices
    n_z = raw.shape[0]
    z_slices = [n_z // 5, n_z // 3, n_z // 2, 2 * n_z // 3, 4 * n_z // 5]
    z_slices = [min(z, n_z - 1) for z in z_slices]
    
    fig, axes = plt.subplots(len(z_slices), 3, figsize=(24, 8 * len(z_slices)))
    fig.suptitle(f'{info["label"]} — {n_instances} total instances', fontsize=16, y=0.995)
    
    for idx, z in enumerate(z_slices):
        raw_slice = raw[z].astype(np.float32)
        # Clip and normalize for display
        p_low, p_high = np.percentile(raw_slice, [0.5, 99.5])
        raw_norm = np.clip((raw_slice - p_low) / max(p_high - p_low, 1), 0, 1)
        
        pred_slice = pred[z]
        n_inst_slice = len(np.unique(pred_slice)) - 1
        
        # Raw image
        axes[idx, 0].imshow(raw_norm, cmap='gray', vmin=0, vmax=1)
        axes[idx, 0].set_title(f'z={z} — Raw', fontsize=12)
        axes[idx, 0].axis('off')
        
        # Prediction only
        colors = instance_colormap(pred_slice)
        axes[idx, 1].imshow(colors)
        axes[idx, 1].set_title(f'z={z} — Prediction ({n_inst_slice} instances)', fontsize=12)
        axes[idx, 1].axis('off')
        
        # Overlay: raw + semi-transparent prediction
        overlay = np.stack([raw_norm]*3, axis=-1)  # grayscale to RGB
        mask = pred_slice > 0
        if mask.any():
            overlay[mask] = overlay[mask] * 0.5 + colors[mask] * 0.5
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title(f'z={z} — Overlay (50% transparent)', fontsize=12)
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{name}_overlay.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved overlay: {out_path}")
    
    # Also save a zoomed crop
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    mid_z = n_z // 2
    raw_slice = raw[mid_z].astype(np.float32)
    p_low, p_high = np.percentile(raw_slice, [0.5, 99.5])
    raw_norm = np.clip((raw_slice - p_low) / max(p_high - p_low, 1), 0, 1)
    pred_slice = pred[mid_z]
    colors = instance_colormap(pred_slice)
    
    # Full view
    overlay_full = np.stack([raw_norm]*3, axis=-1)
    mask = pred_slice > 0
    if mask.any():
        overlay_full[mask] = overlay_full[mask] * 0.5 + colors[mask] * 0.5
    
    axes[0, 0].imshow(raw_norm, cmap='gray')
    axes[0, 0].set_title(f'z={mid_z} Full — Raw')
    axes[0, 1].imshow(colors)
    axes[0, 1].set_title(f'z={mid_z} Full — Prediction')
    axes[0, 2].imshow(overlay_full)
    axes[0, 2].set_title(f'z={mid_z} Full — Overlay')
    
    # Zoomed center crop (512x512)
    cy, cx = raw.shape[1] // 2, raw.shape[2] // 2
    s = 256
    axes[1, 0].imshow(raw_norm[cy-s:cy+s, cx-s:cx+s], cmap='gray')
    axes[1, 0].set_title(f'z={mid_z} Zoom — Raw')
    axes[1, 1].imshow(colors[cy-s:cy+s, cx-s:cx+s])
    axes[1, 1].set_title(f'z={mid_z} Zoom — Prediction')
    axes[1, 2].imshow(overlay_full[cy-s:cy+s, cx-s:cx+s])
    axes[1, 2].set_title(f'z={mid_z} Zoom — Overlay')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{name}_zoom.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved zoom: {out_path}")

print(f"\n{'='*60}")
print(f"All overlays saved to: {out_dir}/")
print("Done!")
