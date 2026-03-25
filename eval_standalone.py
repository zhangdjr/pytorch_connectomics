"""Standalone evaluation replicating exactly what Optuna trial #22 did."""
import sys
sys.path.insert(0, "/projects/weilab/weidf/lib/pytorch_connectomics/lib/MedNeXt")
sys.path.insert(0, "/home/zhangdjr/projects/umich-fiber/pytorch_connectomics")

import numpy as np
import h5py
import tifffile
from connectomics.decoding.segmentation import decode_instance_binary_contour_distance
from connectomics.decoding.utils import remove_small_instances
from connectomics.metrics.segmentation_numpy import adapted_rand

# --- Load cached TTA predictions (intermediate, before decoding) ---
pred_path = "outputs/fiber_retrain_all/20260311_223801/results/0112-5-CA1-4_2_tta_prediction.h5"
print(f"Loading predictions: {pred_path}")
with h5py.File(pred_path, "r") as f:
    key = list(f.keys())[0]
    prediction = f[key][:]
print(f"  Prediction shape: {prediction.shape}, dtype: {prediction.dtype}")

# --- Load ground truth ---
gt_path = "/projects/weilab/dataset/barcode/2026/umich/seg 0112-5-CA1-4_2 [ready to use].tif"
print(f"Loading ground truth: {gt_path}")
ground_truth = tifffile.imread(gt_path)
print(f"  Ground truth shape: {ground_truth.shape}, dtype: {ground_truth.dtype}")
print(f"  GT unique instances: {len(np.unique(ground_truth)) - 1}")

# --- Optuna best params (trial #22) ---
best_decoding_params = dict(
    binary_threshold=[0.7503891044149624, 0.0046306263327246],
    contour_threshold=[0.48708101851244906, 1.025119772551816],
    distance_threshold=[-0.6695257662389609, -0.07270575159140885],
    min_seed_size=40,
    # NOTE: NO min_instance_size — Optuna didn't pass it during tuning
)

print("\n=== Test 1: Replicate Optuna trial #22 exactly ===")
print(f"  Decoding params: {best_decoding_params}")
seg1 = decode_instance_binary_contour_distance(prediction, **best_decoding_params)
print(f"  After decode: {len(np.unique(seg1)) - 1} instances")

# Apply postprocessing (what Optuna did)
seg1_pp = remove_small_instances(seg1, thres_small=870, mode="background")
print(f"  After postprocess (thres_small=870): {len(np.unique(seg1_pp)) - 1} instances")

are1, prec1, rec1 = adapted_rand(seg1_pp, ground_truth, all_stats=True)
print(f"  ARE={are1:.4f}  Precision={prec1:.4f}  Recall={rec1:.4f}")

print("\n=== Test 2: min_instance_size=870 in decode (what we tried) ===")
seg2 = decode_instance_binary_contour_distance(
    prediction, **best_decoding_params, min_instance_size=870
)
print(f"  After decode: {len(np.unique(seg2)) - 1} instances")

are2, prec2, rec2 = adapted_rand(seg2, ground_truth, all_stats=True)
print(f"  ARE={are2:.4f}  Precision={prec2:.4f}  Recall={rec2:.4f}")

print("\n=== Test 3: Original tune-test (min_instance_size=100, no postprocess) ===")
seg3 = decode_instance_binary_contour_distance(
    prediction, **best_decoding_params, min_instance_size=100
)
print(f"  After decode: {len(np.unique(seg3)) - 1} instances")

are3, prec3, rec3 = adapted_rand(seg3, ground_truth, all_stats=True)
print(f"  ARE={are3:.4f}  Precision={prec3:.4f}  Recall={rec3:.4f}")

print("\nDone!")
