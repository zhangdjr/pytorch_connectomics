#!/usr/bin/env python3
"""
Full fiber analysis pipeline: segmentation mask → skeletonization → signal extraction →
normalization → validation → CSV export.

Takes existing fiber segmentation masks (from MedNeXt) and raw multi-channel data,
produces one CSV per tile with fiber-level readouts.

Channel mapping (A1-2003.nd2, standard ordering):
  Ch0: 405nm — nissl/DAPI (neuron labeling)
  Ch1: 488nm — fiber signal (segmentation)
  Ch2: 561nm — cfos
  Ch3: 647nm — timestamp (symmetric, used for midpoint detection)

Usage:
    python tools/fiber_pipeline.py --tile A1 --nd2-name A1-2003
    python tools/fiber_pipeline.py --tile A1 --nd2-name A1-2003 --steps cell_seg,skeletonize,extract,normalize,validate,csv
"""

import os
import argparse
import warnings
import json
import csv
import numpy as np
import h5py
import tifffile
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.interpolate import splprep, splev, interp1d, interpn
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mode, pearsonr
from sklearn.decomposition import PCA
from skimage.measure import label, regionprops

import optuna

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    # Paths
    "tile_dir": "/projects/weilab/dataset/barcode/2026/umich/nd2_tiles",
    "pred_dir": "outputs/fiber_retrain_all/20260311_223801/results",
    "output_dir": "fiber_analysis",

    # Voxel size in nm (Z, Y, X) — from ND2 metadata
    "anisotropy_nm": [400.0, 162.9, 162.9],

    # Channel mapping (index → biological label)
    # Standard ordering for A1-2003.nd2
    "channel_names": {0: "dapi", 1: "fiber", 2: "cfos", 3: "timestamp"},
    "fiber_channel": 1,       # used for tail cropping
    "timestamp_channel": 3,   # used for midpoint detection (symmetric)
    "dapi_channel": 0,        # used for cell segmentation

    # Cell segmentation: pre-computed by cell_seg_microsam.py (microsam env)
    # No config needed here — just loads the NPZ from cache/

    # Skeletonization
    "skeleton": {
        "manual_z_scale": 0.33,  # compress Z for PCA (Z is coarse)
        "percentile_fit": [0.1, 0.9],
        "num_centerline_points": 1000,
        "spline_smoothing": None,
        "extrapolate": [-0.2, 1.2],
    },

    # Signal normalization
    "normalize": {
        "gaussian_filter_ratio": 0.01,
        "midpoint_range": [0.45, 0.55],
        "scale_range": [-0.2, 0.2],
        "num_trials": 100,
    },

    # Fiber validation
    "validate": {
        "thres_length_um": 8.0,   # minimum fiber length in µm
        "thres_pca_ratio": 0.0,    # minimum PCA linearity
        "thres_mean_soma": 0.0,    # minimum soma channel brightness
        "one_per_soma": False,
    },

    # Parallelism
    "n_jobs": -1,
}


def get_config():
    """Return a copy of the default config."""
    import copy
    return copy.deepcopy(DEFAULT_CONFIG)


# ============================================================================
# Step 1: Cell Segmentation (load pre-computed micro-sam output)
# ============================================================================

def load_cell_segmentation(output_path):
    """Load pre-computed cell segmentation from micro-sam (separate env).
    
    Run cell_seg_microsam.py in the 'microsam' conda env first to generate
    the NPZ file, then this pipeline loads it.
    """
    print(f"\n{'='*60}")
    print("STEP: Cell Segmentation (micro-sam, pre-computed)")
    print(f"{'='*60}")

    if not os.path.exists(output_path):
        raise FileNotFoundError(
            f"Cell segmentation not found: {output_path}\n"
            f"Run cell_seg_microsam.py in the 'microsam' conda env first:\n"
            f"  conda activate microsam\n"
            f"  python tools/cell_seg_microsam.py --tile <TILE>"
        )

    cell_seg = np.load(output_path)["cell_seg"]
    n_labels = len(np.unique(cell_seg)) - 1
    n_cells_per_slice = [len(np.unique(cell_seg[z])) - 1 for z in range(cell_seg.shape[0])]
    print(f"  Loaded: {output_path}")
    print(f"  Shape: {cell_seg.shape}, {n_labels} total 3D cell labels")
    print(f"  Cells per slice: min={min(n_cells_per_slice)}, max={max(n_cells_per_slice)}, mean={np.mean(n_cells_per_slice):.1f}")
    return cell_seg


# ============================================================================
# Step 2: Skeletonization (ported from Jason's pipeline)
# ============================================================================

def fit_cylinder_spline_pca(points, manual_z_scale=1.0, percentile_fit=(0.0, 1.0),
                            spline_smoothing=None, npoints_geodesic=1000):
    """
    Fit a spline to a deformed cylindrical point cloud using PCA coordinate system.
    Returns: (evaluate_spline_fn, total_arc_length)
    """
    assert len(percentile_fit) == 2

    # Scale Z for PCA
    points = points * np.array([manual_z_scale, 1.0, 1.0])
    center = np.mean(points, axis=0)
    radius = np.max(np.linalg.norm(points - center, axis=1))
    if radius == 0:
        radius = 1.0
    points = (points - center) / radius

    # PCA to find principal axis
    pca = PCA(n_components=3)
    pca.fit(points)
    points_pca = pca.transform(points)

    # Sort by first PCA component
    t_values = points_pca[:, 0]
    t_sorted, sorted_unique_indices = np.unique(t_values, return_index=True)
    points_pca_sorted = points_pca[sorted_unique_indices]

    # Normalize parameter to [0, 1]
    t_range = t_sorted.max() - t_sorted.min()
    if t_range == 0:
        t_range = 1.0
    t_normalized = (t_sorted - t_sorted.min()) / t_range

    # Filter to percentile range
    used = (t_normalized >= percentile_fit[0]) & (t_normalized <= percentile_fit[1])
    points_pca_sorted = points_pca_sorted[used]
    t_normalized = t_normalized[used]

    if len(points_pca_sorted) < 4:
        raise ValueError(f"Too few points ({len(points_pca_sorted)}) for spline fitting")

    # Fit spline
    tck, u = splprep(
        [points_pca_sorted[:, 0], points_pca_sorted[:, 1], points_pca_sorted[:, 2]],
        u=t_normalized, s=spline_smoothing,
    )

    # Arc length reparameterization
    u_dense = np.linspace(0, 1, npoints_geodesic)
    spline_points_dense = np.array(splev(u_dense, tck)).T
    diff = np.diff(spline_points_dense, axis=0)
    distances = np.sqrt(np.sum(diff**2, axis=1))
    arc_lengths = np.concatenate([[0], np.cumsum(distances)])
    arc_lengths_normalized = arc_lengths / arc_lengths[-1] if arc_lengths[-1] > 0 else arc_lengths

    arc_to_param = interp1d(arc_lengths_normalized, u_dense, kind="linear",
                            bounds_error=False, fill_value="extrapolate")

    def evaluate_spline(s_values):
        u_values = arc_to_param(s_values)
        spline_pca = np.array(splev(u_values, tck)).T
        points_xyz = pca.inverse_transform(spline_pca)
        points_xyz = points_xyz * radius + center
        points_xyz = points_xyz * np.array([1.0 / manual_z_scale, 1.0, 1.0])
        return points_xyz

    return evaluate_spline, arc_lengths[-1]


def skeletonize_single_fiber(fiber_mask_crop, bbox, anisotropy_nm, skel_cfg):
    """Skeletonize a single fiber instance. Returns (centerline_points, total_length_nm)."""
    points = np.argwhere(fiber_mask_crop) * np.array(anisotropy_nm)
    if len(points) < 10:
        return None, 0

    try:
        spline_fn, _ = fit_cylinder_spline_pca(
            points,
            manual_z_scale=skel_cfg["manual_z_scale"],
            percentile_fit=skel_cfg["percentile_fit"],
            spline_smoothing=skel_cfg["spline_smoothing"],
        )

        # Compute physical arc length from non-extrapolated core [0, 1].
        core_pts = spline_fn(np.linspace(0, 1, 200))
        total_length_nm = np.sum(np.linalg.norm(np.diff(core_pts, axis=0), axis=1))

        # Sanity check: reject divergent spline fits (no fiber > 500 um in one tile).
        if total_length_nm > 500_000:
            return None, 0

        extrapolate = skel_cfg["extrapolate"]
        centerline = spline_fn(
            np.linspace(extrapolate[0], extrapolate[1], skel_cfg["num_centerline_points"])
        )
        # Shift back to global coordinates
        centerline[:, 0] += bbox[0] * anisotropy_nm[0]
        centerline[:, 1] += bbox[2] * anisotropy_nm[1]
        centerline[:, 2] += bbox[4] * anisotropy_nm[2]
        return centerline, total_length_nm
    except Exception as e:
        return None, 0


def run_skeletonization(fiber_seg, config):
    """Skeletonize all fiber instances."""
    print(f"\n{'='*60}")
    print("STEP: Fiber Skeletonization")
    print(f"{'='*60}")

    aniso = config["anisotropy_nm"]
    skel_cfg = config["skeleton"]
    fiber_ids = np.unique(fiber_seg)
    fiber_ids = fiber_ids[fiber_ids != 0]
    print(f"  Found {len(fiber_ids)} fiber instances")

    # Get bounding boxes for all fibers
    props = regionprops(fiber_seg)
    fiber_data = []
    for prop in props:
        bbox = prop.bbox  # (z_min, y_min, x_min, z_max, y_max, x_max)
        crop = fiber_seg[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
        mask = (crop == prop.label)
        fiber_data.append((prop.label, mask, (bbox[0], bbox[3], bbox[1], bbox[4], bbox[2], bbox[5])))

    def process_fiber(label_id, mask, bbox):
        centerline, length = skeletonize_single_fiber(mask, bbox, aniso, skel_cfg)
        return label_id, centerline, length

    results = list(tqdm(
        Parallel(n_jobs=config["n_jobs"], return_as="generator")(
            delayed(process_fiber)(lid, mask, bbox) for lid, mask, bbox in fiber_data
        ),
        total=len(fiber_data),
        desc="  Skeletonizing",
    ))

    skeletons = {}
    failed = 0
    for label_id, centerline, length in results:
        if centerline is not None:
            skeletons[label_id] = {"centerline": centerline, "length_nm": length}
        else:
            failed += 1

    print(f"  Successfully skeletonized: {len(skeletons)}/{len(fiber_data)} ({failed} failed)")
    return skeletons


# ============================================================================
# Step 3: Signal Extraction
# ============================================================================

def extract_signals_along_skeleton(volume, centerline, anisotropy_nm, method="linear"):
    """Sample a 3D volume along skeleton points using interpolation."""
    points = [np.arange(volume.shape[i], dtype=float) * anisotropy_nm[i] for i in range(3)]
    signals = interpn(points, volume, centerline, method=method,
                      bounds_error=False, fill_value=0)
    return signals


def run_signal_extraction(skeletons, raw_channels, cell_seg, fiber_seg, config):
    """Extract signals from all channels along each skeleton (vectorized)."""
    print(f"\n{'='*60}")
    print("STEP: Signal Extraction (vectorized)")
    print(f"{'='*60}")

    aniso = config["anisotropy_nm"]
    ch_names = config["channel_names"]

    # Collect all skeleton points into one big array for batched interpn
    fid_list = list(skeletons.keys())
    if not fid_list:
        print("  No skeletons found - skipping signal extraction")
        return skeletons

    all_centerlines = [skeletons[fid]["centerline"] for fid in fid_list]
    n_pts_per_fiber = [len(cl) for cl in all_centerlines]
    all_points = np.concatenate(all_centerlines, axis=0)  # (N_total, 3) in nm
    print(f"  Batched {len(fid_list)} fibers, {len(all_points)} total points")

    # Build grid coordinates once
    vol_shape = cell_seg.shape  # all volumes share same shape
    grid = [np.arange(vol_shape[i], dtype=float) * aniso[i] for i in range(3)]

    # Extract each raw channel in one interpn call
    extracted = {}
    for ch_idx, ch_name in ch_names.items():
        if ch_idx in raw_channels:
            print(f"  Extracting {ch_name} (linear)...")
            extracted[ch_name] = interpn(
                grid, raw_channels[ch_idx].astype(float), all_points,
                method="linear", bounds_error=False, fill_value=0
            )

    # Cell segmentation (nearest)
    print(f"  Extracting cell_seg (nearest)...")
    extracted["cell_seg"] = interpn(
        grid, cell_seg.astype(float), all_points,
        method="nearest", bounds_error=False, fill_value=0
    ).astype(np.int32)

    # Fiber segmentation (nearest)
    print(f"  Extracting fiber_seg (nearest)...")
    extracted["fiber_seg"] = interpn(
        grid, fiber_seg.astype(float), all_points,
        method="nearest", bounds_error=False, fill_value=0
    ).astype(np.int32)

    # Split batched results back to per-fiber
    seg_keys = {"cell_seg", "fiber_seg"}
    offset = 0
    for fid, n_pts in zip(fid_list, n_pts_per_fiber):
        signals = {}
        for key, vals in extracted.items():
            signals[key] = vals[offset:offset + n_pts]
            if key not in seg_keys:
                signals[key] = signals[key].astype(float)
        skeletons[fid]["signals"] = signals
        offset += n_pts

    print(f"  Extracted signals for {len(skeletons)} fibers")
    return skeletons


# ============================================================================
# Step 4: Signal Normalization (midpoint detection)
# ============================================================================

def get_tail_crop_points(signal, extrapolate, gaussian_filter_ratio):
    """Find crop points by detecting where signal drops off at the tails."""
    smoothed = gaussian_filter1d(signal, int(gaussian_filter_ratio * len(signal)))
    num_points = signal.shape[0]
    start = int((0 - extrapolate[0]) / (extrapolate[1] - extrapolate[0]) * num_points)
    stop = int((1 - extrapolate[0]) / (extrapolate[1] - extrapolate[0]) * num_points)

    while (start > 0 and signal[start] > 0
           and smoothed[start - 1] < smoothed[start]):
        start -= 1
    while (stop < len(signal) - 1 and signal[stop] > 0
           and smoothed[stop + 1] < smoothed[stop]):
        stop += 1

    return start, stop


def apply_midpoint_scale(data, index, scale, num_points, calculate_pearson=False):
    """Split data at midpoint, optionally scale halves, resample to fixed length."""
    dtype = data.dtype
    is_float = np.issubdtype(dtype, np.floating)
    index = int(data.shape[0] * index)
    x, y = data[:index][::-1], data[index:]

    if len(x) < 2 or len(y) < 2:
        if calculate_pearson:
            return data, 0.0
        return data

    if scale < 0:
        x = x[:max(1, int((1 - abs(scale)) * x.shape[0]))]
    elif scale > 0:
        y = y[:max(1, int((1 - abs(scale)) * y.shape[0]))]

    half = num_points // 2
    if len(x) < 2 or len(y) < 2:
        if calculate_pearson:
            return data, 0.0
        return data

    x = interp1d(np.linspace(0, 1, x.shape[0]), x,
                  kind="linear" if is_float else "nearest"
                  )(np.linspace(0, 1, half)).astype(dtype)
    y = interp1d(np.linspace(0, 1, y.shape[0]), y,
                  kind="linear" if is_float else "nearest"
                  )(np.linspace(0, 1, half)).astype(dtype)

    result = np.concatenate([x[::-1], y])
    if calculate_pearson:
        if np.std(x) == 0 or np.std(y) == 0:
            return result, 0.0
        pearson = pearsonr(x, y).statistic
        return result, pearson
    return result


def get_midpoint_scale(signal, midpoint_range, scale_range, num_trials, num_points):
    """Use Optuna to find optimal midpoint and scale for the timestamp channel."""
    def objective(trial):
        index = trial.suggest_float("index", midpoint_range[0], midpoint_range[1])
        scale = trial.suggest_float("scale", scale_range[0], scale_range[1])
        _, pearson = apply_midpoint_scale(signal, index, scale, num_points, calculate_pearson=True)
        return -pearson

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=num_trials)
    return study.best_params["index"], study.best_params["scale"]


def get_geodesic(skel, center_idx):
    """Compute signed geodesic distance from center point along skeleton."""
    dist = np.cumsum(np.linalg.norm(np.diff(skel, axis=0), axis=1))
    dist = np.concatenate([[0], dist])
    dist -= dist[center_idx]
    return dist


def normalize_single_fiber(skel_data, config):
    """Normalize signals for a single fiber: crop tails, find midpoint, recenter."""
    signals = skel_data["signals"]
    centerline = skel_data["centerline"]
    skel_cfg = config["skeleton"]
    norm_cfg = config["normalize"]
    n_pts = skel_cfg["num_centerline_points"]

    # Get fiber channel name for tail cropping
    fiber_ch_name = config["channel_names"][config["fiber_channel"]]
    timestamp_ch_name = config["channel_names"][config["timestamp_channel"]]

    if fiber_ch_name not in signals:
        return None

    # Crop tails based on fiber channel
    try:
        start, stop = get_tail_crop_points(
            signals[fiber_ch_name], skel_cfg["extrapolate"], norm_cfg["gaussian_filter_ratio"]
        )
    except Exception:
        start, stop = 0, len(signals[fiber_ch_name]) - 1

    if stop - start < 10:
        return None

    centerline = centerline[start:stop + 1]
    signals = {k: v[start:stop + 1] for k, v in signals.items()}

    # Find midpoint using timestamp channel
    if timestamp_ch_name in signals and len(signals[timestamp_ch_name]) > 10:
        try:
            midpoint, scale = get_midpoint_scale(
                signals[timestamp_ch_name],
                norm_cfg["midpoint_range"],
                norm_cfg["scale_range"],
                norm_cfg["num_trials"],
                n_pts,
            )
        except Exception:
            midpoint, scale = 0.5, 0.0
    else:
        midpoint, scale = 0.5, 0.0

    # Recenter all signals and skeleton coordinates
    recentered = {}
    for key, sig in signals.items():
        recentered[key] = apply_midpoint_scale(sig, midpoint, scale, n_pts)

    # Also recenter skeleton coordinates
    skel_z = apply_midpoint_scale(centerline[:, 0].astype(float), midpoint, scale, n_pts)
    skel_y = apply_midpoint_scale(centerline[:, 1].astype(float), midpoint, scale, n_pts)
    skel_x = apply_midpoint_scale(centerline[:, 2].astype(float), midpoint, scale, n_pts)
    centerline_recentered = np.stack([skel_z, skel_y, skel_x], axis=-1)

    # Compute geodesic distance from midpoint
    geodesic = get_geodesic(centerline_recentered, n_pts // 2)

    return {
        "signals": recentered,
        "centerline": centerline_recentered,
        "geodesic": geodesic,
        "midpoint": midpoint,
        "scale": scale,
    }


def run_normalization(skeletons, config):
    """Normalize signals for all fibers."""
    print(f"\n{'='*60}")
    print("STEP: Signal Normalization")
    print(f"{'='*60}")

    results = {}
    failed = 0
    for fid in tqdm(skeletons, desc="  Normalizing"):
        result = normalize_single_fiber(skeletons[fid], config)
        if result is not None:
            results[fid] = result
            results[fid]["length_nm"] = skeletons[fid]["length_nm"]
            # Keep original centerline for correct spatial coordinates in CSV
            results[fid]["centerline_original"] = skeletons[fid]["centerline"]
        else:
            failed += 1

    print(f"  Normalized: {len(results)}/{len(skeletons)} ({failed} failed)")
    return results


# ============================================================================
# Step 5: Fiber Validation
# ============================================================================

def validate_fibers(normalized, config):
    """Filter fibers by length, PCA linearity, and soma presence."""
    print(f"\n{'='*60}")
    print("STEP: Fiber Validation")
    print(f"{'='*60}")

    val_cfg = config["validate"]
    aniso = config["anisotropy_nm"]
    nm_per_um = 1000.0
    fiber_seg_nz = config.get("_fiber_seg_nz", 54)

    valid = {}
    stats = {"total": len(normalized), "short": 0, "low_pca": 0, "no_soma": 0, "divergent": 0}

    for fid, data in normalized.items():
        centerline = data["centerline"]

        # Fiber length in um: recompute from original centerline core [0,1].
        cl_orig = data.get("centerline_original", centerline)
        ext = config["skeleton"]["extrapolate"]
        n = len(cl_orig)
        total_range = ext[1] - ext[0]
        if total_range <= 0:
            total_range = 1.0
        core_start = int(round((0 - ext[0]) / total_range * n))
        core_end = int(round((1 - ext[0]) / total_range * n))
        core_start = max(0, min(n, core_start))
        core_end = max(core_start + 1, min(n, core_end))
        core = cl_orig[core_start:core_end]
        if len(core) < 2:
            core = cl_orig
        length_nm = np.sum(np.linalg.norm(np.diff(core, axis=0), axis=1)) if len(core) > 1 else 0.0
        length_um = length_nm / nm_per_um

        # PCA linearity
        center = np.mean(centerline, axis=0)
        radius = np.max(np.linalg.norm(centerline - center, axis=1))
        if radius > 0:
            pts_norm = (centerline - center) / radius
            pca = PCA(n_components=3)
            pca.fit(pts_norm)
            pca_ratio = pca.explained_variance_ratio_[0]
        else:
            pca_ratio = 0.0

        # Cell association (mode of cell_seg along skeleton)
        cell_labels = data["signals"].get("cell_seg", np.array([0]))
        cell_label = int(mode(cell_labels, keepdims=False).mode)

        # Mean DAPI/soma brightness
        dapi_ch_name = config["channel_names"][config["dapi_channel"]]
        mean_soma = np.mean(data["signals"].get(dapi_ch_name, np.array([0])))

        # Divergent spline check: Z span of core centerline must be physically plausible.
        z_span_um = (core[:, 0].max() - core[:, 0].min()) / nm_per_um if len(core) > 0 else 0.0
        max_z_um = aniso[0] / nm_per_um * fiber_seg_nz * 3

        # Apply filters
        if z_span_um > max_z_um:
            stats["divergent"] += 1
            is_valid = False
        elif length_um < val_cfg["thres_length_um"]:
            stats["short"] += 1
            is_valid = False
        elif pca_ratio < val_cfg["thres_pca_ratio"]:
            stats["low_pca"] += 1
            is_valid = False
        elif mean_soma < val_cfg["thres_mean_soma"]:
            stats["no_soma"] += 1
            is_valid = False
        else:
            is_valid = True

        data["length_um"] = length_um
        data["pca_ratio"] = pca_ratio
        data["cell_label"] = cell_label
        data["mean_soma"] = mean_soma
        data["is_valid"] = is_valid

        valid[fid] = data

    n_valid = sum(1 for d in valid.values() if d["is_valid"])
    print(f"  Total fibers: {stats['total']}")
    print(f"  Divergent spline (Z span > {fiber_seg_nz * 3} slices): {stats['divergent']}")
    print(f"  Too short (<{val_cfg['thres_length_um']}µm): {stats['short']}")
    print(f"  Low PCA ratio: {stats['low_pca']}")
    print(f"  No soma signal: {stats['no_soma']}")
    print(f"  Valid fibers: {n_valid}/{len(valid)}")

    # one_per_soma filter
    if val_cfg["one_per_soma"]:
        cell_to_fibers = {}
        for fid, data in valid.items():
            if data["is_valid"] and data["cell_label"] != 0:
                cell_to_fibers.setdefault(data["cell_label"], []).append(fid)
        for cell_id, fids in cell_to_fibers.items():
            if len(fids) > 1:
                best = max(fids, key=lambda f: valid[f]["length_um"])
                for f in fids:
                    if f != best:
                        valid[f]["is_valid"] = False
        n_valid_after = sum(1 for d in valid.values() if d["is_valid"])
        print(f"  After one-per-soma: {n_valid_after}/{n_valid}")

    return valid


# ============================================================================
# Step 6: CSV Export
# ============================================================================

def export_csv(validated, nd2_name, tile_name, output_dir, config):
    """Export per-fiber CSV with all readouts."""
    print(f"\n{'='*60}")
    print("STEP: CSV Export")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{nd2_name}_{tile_name}.csv")

    ch_names = config["channel_names"]
    nm_per_um = 1000.0
    n_pts = config["skeleton"]["num_centerline_points"]
    mid = n_pts // 2

    rows = []
    for fid, data in validated.items():
        # Use original centerline for correct spatial coordinates
        centerline = data.get("centerline_original", data["centerline"])
        signals = data["signals"]

        # Centroid (midpoint of skeleton) in µm
        centroid_z = centerline[mid, 0] / nm_per_um
        centroid_y = centerline[mid, 1] / nm_per_um
        centroid_x = centerline[mid, 2] / nm_per_um

        row = {
            "fiber_id": fid,
            "nd2_name": nd2_name,
            "tile_name": tile_name,
            "is_valid": data["is_valid"],
            "parent_cell_id": data["cell_label"],
            "fiber_length_um": round(data["length_um"], 2),
            "pca_linearity": round(data["pca_ratio"], 4),
            "centroid_z_um": round(centroid_z, 2),
            "centroid_y_um": round(centroid_y, 2),
            "centroid_x_um": round(centroid_x, 2),
        }

        # Per-channel summary stats
        for ch_idx, ch_name in ch_names.items():
            if ch_name in signals:
                sig = signals[ch_name]
                row[f"{ch_name}_mean"] = round(float(np.mean(sig)), 2)
                row[f"{ch_name}_median"] = round(float(np.median(sig)), 2)
                row[f"{ch_name}_min"] = round(float(np.min(sig)), 2)
                row[f"{ch_name}_max"] = round(float(np.max(sig)), 2)
                row[f"{ch_name}_std"] = round(float(np.std(sig)), 2)

        # DAPI mean (soma brightness)
        row["mean_soma_dapi"] = round(data["mean_soma"], 2)

        rows.append(row)

    # Sort by fiber_id
    rows.sort(key=lambda r: r["fiber_id"])

    # Write CSV
    if rows:
        fieldnames = rows[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    n_valid = sum(1 for r in rows if r["is_valid"])
    print(f"  Wrote {len(rows)} fibers ({n_valid} valid) → {csv_path}")
    return csv_path


def export_profiles(validated, nd2_name, tile_name, output_dir, config):
    """Export full 1000-point intensity profiles as NPZ.

    Saves one NPZ per tile with arrays:
      fiber_ids:  (N,)      — fiber instance IDs
      is_valid:   (N,)      — boolean validity flags
      dapi:       (N, 1000) — DAPI intensity along each fiber
      fiber:      (N, 1000) — fiber channel intensity
      cfos:       (N, 1000) — cfos intensity
      timestamp:  (N, 1000) — timestamp intensity
    """
    print(f"\n{'='*60}")
    print("STEP: Intensity Profile Export")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, f"{nd2_name}_{tile_name}_profiles.npz")

    ch_names = config["channel_names"]

    # Collect arrays sorted by fiber_id
    fids_sorted = sorted(validated.keys())
    fiber_ids = np.array(fids_sorted, dtype=np.int32)
    is_valid = np.array([validated[fid]["is_valid"] for fid in fids_sorted], dtype=bool)

    profiles = {}
    for ch_idx, ch_name in ch_names.items():
        ch_profiles = []
        for fid in fids_sorted:
            sig = validated[fid]["signals"].get(ch_name)
            if sig is not None:
                ch_profiles.append(sig)
            else:
                ch_profiles.append(np.zeros(config["skeleton"]["num_centerline_points"]))
        profiles[ch_name] = np.array(ch_profiles, dtype=np.float32)

    np.savez_compressed(npz_path,
        fiber_ids=fiber_ids,
        is_valid=is_valid,
        **profiles,
    )

    print(f"  Saved {len(fiber_ids)} fiber profiles ({profiles[list(ch_names.values())[0]].shape[1]} points each)")
    print(f"  Channels: {list(ch_names.values())}")
    print(f"  → {npz_path}")
    return npz_path


# ============================================================================
# Main Pipeline
# ============================================================================

def load_tile_data(tile_name, config):
    """Load all data for a single tile."""
    tile_dir = config["tile_dir"]
    pred_dir = config["pred_dir"]
    ch_names = config["channel_names"]

    # Fiber segmentation mask (try _prediction_fixed.tiff first, then _prediction.tiff)
    pred_path = os.path.join(pred_dir, f"{tile_name}_ch1_prediction_fixed.tiff")
    if not os.path.exists(pred_path):
        pred_path = os.path.join(pred_dir, f"{tile_name}_ch1_prediction.tiff")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"Fiber segmentation not found in {pred_dir}/ "
            f"(tried {tile_name}_ch1_prediction_fixed.tiff and {tile_name}_ch1_prediction.tiff)"
        )
    fiber_seg = tifffile.imread(pred_path)
    print(f"  Fiber seg: {fiber_seg.shape}, {len(np.unique(fiber_seg))-1} instances")

    # Raw channels
    channel_files = {
        0: f"{tile_name}_ch0_dapi.tif",
        1: f"{tile_name}_ch1.tif",
        2: f"{tile_name}_ch2_cfos.tif",
        3: f"{tile_name}_ch3_timestamp.tif",
    }
    raw_channels = {}
    for ch_idx, fname in channel_files.items():
        fpath = os.path.join(tile_dir, fname)
        if os.path.exists(fpath):
            raw_channels[ch_idx] = tifffile.imread(fpath)
            print(f"  Ch{ch_idx} ({ch_names[ch_idx]}): {raw_channels[ch_idx].shape}")
        else:
            print(f"  WARNING: Ch{ch_idx} ({ch_names[ch_idx]}) not found: {fpath}")

    return fiber_seg, raw_channels


def run_pipeline(tile_name, nd2_name, config, steps=None):
    """Run the full pipeline on a single tile."""
    all_steps = ["cell_seg", "skeletonize", "extract", "normalize", "validate", "csv"]
    if steps is None:
        steps = all_steps

    print(f"\n{'#'*60}")
    print(f"# FIBER PIPELINE: {nd2_name} / {tile_name}")
    print(f"{'#'*60}")

    output_dir = os.path.join(config["output_dir"], nd2_name)
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Load tile data
    print(f"\nLoading tile data...")
    fiber_seg, raw_channels = load_tile_data(tile_name, config)
    config["_fiber_seg_nz"] = fiber_seg.shape[0]

    # Cell segmentation (pre-computed by cell_seg_microsam.py in 'microsam' env)
    cell_seg = None
    cell_seg_cache = os.path.join(cache_dir, f"{tile_name}_cell_seg.npz")
    if os.path.exists(cell_seg_cache):
        cell_seg = load_cell_segmentation(cell_seg_cache)
    else:
        print(f"\n  WARNING: Cell seg not found at {cell_seg_cache}")
        print(f"  Run: conda activate microsam && python tools/cell_seg_microsam.py --tile {tile_name}")

    if cell_seg is None:
        print("  WARNING: No cell segmentation available, creating zeros")
        cell_seg = np.zeros_like(fiber_seg, dtype=np.int32)

    # Skeletonization
    skeletons = None
    skel_cache = os.path.join(cache_dir, f"{tile_name}_skeletons.npz")
    if "skeletonize" in steps:
        skeletons = run_skeletonization(fiber_seg, config)
        # Cache skeletons
        np.savez_compressed(skel_cache,
            fiber_ids=np.array(list(skeletons.keys())),
            centerlines=[s["centerline"] for s in skeletons.values()],
            lengths=[s["length_nm"] for s in skeletons.values()],
        )
    elif os.path.exists(skel_cache):
        cached = np.load(skel_cache, allow_pickle=True)
        skeletons = {}
        for fid, cl, ln in zip(cached["fiber_ids"], cached["centerlines"], cached["lengths"]):
            skeletons[int(fid)] = {"centerline": cl, "length_nm": float(ln)}
        print(f"\n  Loaded cached skeletons: {len(skeletons)} fibers")

    if skeletons is None:
        raise RuntimeError("No skeletons available. Run 'skeletonize' step first.")

    # Early exit for tiles with no detected fibers.
    if len(skeletons) == 0:
        print(f"\n  No fibers detected in tile {tile_name} - skipping remaining steps")
        print(f"\n{'#'*60}")
        print(f"# PIPELINE COMPLETE: {nd2_name} / {tile_name} (0 fibers)")
        print(f"{'#'*60}")
        return None

    # Signal extraction
    if "extract" in steps:
        skeletons = run_signal_extraction(skeletons, raw_channels, cell_seg, fiber_seg, config)

    # Normalization
    normalized = None
    if "normalize" in steps:
        normalized = run_normalization(skeletons, config)

    if normalized is None:
        normalized = skeletons  # pass through without normalization

    # Validation
    validated = None
    if "validate" in steps:
        validated = validate_fibers(normalized, config)

    if validated is None:
        validated = normalized

    # CSV export
    csv_path = None
    if "csv" in steps:
        csv_path = export_csv(validated, nd2_name, tile_name, output_dir, config)
        export_profiles(validated, nd2_name, tile_name, output_dir, config)

    print(f"\n{'#'*60}")
    print(f"# PIPELINE COMPLETE: {nd2_name} / {tile_name}")
    print(f"{'#'*60}")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fiber analysis pipeline")
    parser.add_argument("--tile", required=True, help="Tile name (e.g., A1)")
    parser.add_argument("--nd2-name", required=True, help="ND2 file name (e.g., A1-2003)")
    parser.add_argument("--steps", default=None,
                        help="Comma-separated steps: cell_seg,skeletonize,extract,normalize,validate,csv")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--tile-dir", default=None, help="Directory containing extracted tile TIFFs")
    parser.add_argument("--pred-dir", default=None, help="Directory containing fiber seg predictions")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for skeletonization")
    args = parser.parse_args()

    config = get_config()
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.tile_dir:
        config["tile_dir"] = args.tile_dir
    if args.pred_dir:
        config["pred_dir"] = args.pred_dir
    config["n_jobs"] = args.n_jobs

    steps = args.steps.split(",") if args.steps else None
    run_pipeline(args.tile, args.nd2_name, config, steps=steps)
