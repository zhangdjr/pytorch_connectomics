#!/usr/bin/env python3
"""Evaluate best Optuna trial with precision/recall per volume."""
import sys
import glob
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from connectomics.data.io import read_volume
from connectomics.metrics.segmentation_numpy import adapted_rand
from connectomics.decoding.segmentation import decode_instance_binary_contour_distance
from connectomics.decoding.utils import remove_small_instances

# Best trial 53 parameters
binary_threshold = (0.33130717636508034, 6.468528151338387e-05)
contour_threshold = (0.09244451324114467, 1.036914970461398)
distance_threshold = (-0.637476675014741, -0.19816581852258097)
min_seed_size = 42
thres_small = 1200

# File paths
pred_dir = Path("/projects/weilab/zhangdjr/umich-fiber/pytorch_connectomics/outputs/fiber_linghu26/20260220_122809/results")
label_dir = Path("/projects/weilab/dataset/barcode/2026/PT37_round2")

pred_files = sorted(pred_dir.glob("*_tta_prediction.h5"))
label_names = [
    "1-mask.tif", "2-mask.tif", "3-mask.tif",
    "4_1-mask.tif", "4_2-mask.tif", "4_3-mask.tif",
    "5_1-mask.tif", "5_2-mask.tif",
    "6_1-mask.tif", "6_2-mask.tif",
]

print(f"{'Vol':>10s}  {'ARE':>8s}  {'Prec':>8s}  {'Recall':>8s}")
print("-" * 42)

all_are, all_prec, all_rec = [], [], []

for pred_file, label_name in zip(pred_files, label_names):
    pred = read_volume(str(pred_file))
    gt = read_volume(str(label_dir / label_name))

    seg = decode_instance_binary_contour_distance(
        pred,
        binary_threshold=binary_threshold,
        contour_threshold=contour_threshold,
        distance_threshold=distance_threshold,
        min_seed_size=min_seed_size,
    )
    seg = remove_small_instances(seg, thres_small=thres_small, mode="background")

    are, prec, rec = adapted_rand(seg, gt, all_stats=True)
    all_are.append(are)
    all_prec.append(prec)
    all_rec.append(rec)

    vol_name = pred_file.stem.replace("_tta_prediction", "")
    print(f"{vol_name:>10s}  {are:8.4f}  {prec:8.4f}  {rec:8.4f}")

print("-" * 42)
print(f"{'Mean':>10s}  {np.mean(all_are):8.4f}  {np.mean(all_prec):8.4f}  {np.mean(all_rec):8.4f}")
print(f"\nMichael baseline:  ARE=0.383  Prec=0.563  Recall=0.681")
