#!/bin/bash
#SBATCH --job-name=test_pad_fix
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Quick test: verify z-padding fix on a single tile
# Expected: prediction Z-dimension should match input (56 slices), NOT 128

source ~/.bashrc
conda activate pytc

export PYTHONPATH=/projects/weilab/weidf/lib/pytorch_connectomics/lib/MedNeXt:/home/zhangdjr/projects/umich-fiber/pytorch_connectomics:$PYTHONPATH

WORK_DIR="/home/zhangdjr/projects/umich-fiber/pytorch_connectomics"
cd $WORK_DIR

echo "=== Z-Padding Fix Verification Test ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo ""

# Clean any cached predictions for A1 so inference runs fresh
rm -f outputs/fiber_retrain_all/20260311_223801/results/A1_ch1_prediction.h5
rm -f outputs/fiber_retrain_all/20260311_223801/results/A1_ch1_tta_prediction.h5
rm -f outputs/fiber_retrain_all/20260311_223801/results/A1_ch1_prediction.tiff

# Run inference on just tile A1 using a minimal config
python -c "
import yaml, sys

# Load the full config and override to just 1 tile
config_path = 'tutorials/fiber_nd2_all_tiles.yaml'
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# Override to single tile
cfg['test']['data']['test_image'] = ['/projects/weilab/dataset/barcode/2026/umich/nd2_tiles/A1_ch1.tif']

# Save temp config in tutorials/ so _base_ relative paths resolve
with open('tutorials/_test_padding_fix.yaml', 'w') as f:
    yaml.dump(cfg, f)

print('Temp config written with single tile A1')
"

python scripts/main.py \
    --config tutorials/_test_padding_fix.yaml \
    --mode test \
    --checkpoint outputs/fiber_retrain_all/20260311_223801/checkpoints/last.ckpt

echo ""
echo "=== Checking output dimensions ==="

python -c "
import tifffile, h5py, os, sys

results_dir = 'outputs/fiber_retrain_all/20260311_223801/results'

# Check TIFF
tiff_path = os.path.join(results_dir, 'A1_ch1_prediction.tiff')
if os.path.exists(tiff_path):
    pred = tifffile.imread(tiff_path)
    print(f'TIFF prediction shape: {pred.shape}')
    if pred.shape[0] == 56:
        print('✅ Z-PADDING FIX VERIFIED: prediction has 56 slices (matches input)')
    elif pred.shape[0] == 128:
        print('❌ BUG STILL PRESENT: prediction has 128 slices (should be 56)')
    else:
        print(f'⚠️  UNEXPECTED: prediction has {pred.shape[0]} slices')
else:
    print(f'No TIFF found at {tiff_path}')

# Check H5
h5_path = os.path.join(results_dir, 'A1_ch1_prediction.h5')
if os.path.exists(h5_path):
    with h5py.File(h5_path, 'r') as f:
        key = list(f.keys())[0]
        shape = f[key].shape
        print(f'H5 prediction shape: {shape}')
else:
    print(f'No H5 found at {h5_path}')

# Also check input for reference
raw_path = '/projects/weilab/dataset/barcode/2026/umich/nd2_tiles/A1_ch1.tif'
raw = tifffile.imread(raw_path)
print(f'Raw input shape: {raw.shape}')
print(f'Expected: prediction Z == raw Z == {raw.shape[0]}')
"

echo ""
echo "=== Test complete: $(date) ==="
