import neuroglancer
import numpy as np
from PIL import Image

volumes = {
    'A1-2002': {
        'raw': '/projects/weilab/dataset/barcode/2026/umich/processed_ch1/A1-2002_ch1.tif',
        'pred': 'outputs/fiber_retrain_all/20260311_223801/results_multiregion_fixed/A1-2002_ch1_prediction_fixed.tiff',
        'res': (400, 162.9, 162.9),
        'label': 'A1-2002 (77 slices)',
    },
    'A1-2003_s01': {
        'raw': '/projects/weilab/dataset/barcode/2026/umich/processed_ch1/A1-2003_s01_ch1.tif',
        'pred': 'outputs/fiber_retrain_all/20260311_223801/results_multiregion_fixed/A1-2003_s01_ch1_prediction_fixed.tiff',
        'res': (400, 162.9, 162.9),
        'label': 'A1-2003 Series 01 (56 slices)',
    },
    'A1-2003_s02': {
        'raw': '/projects/weilab/dataset/barcode/2026/umich/processed_ch1/A1-2003_s02_ch1.tif',
        'pred': 'outputs/fiber_retrain_all/20260311_223801/results_multiregion_fixed/A1-2003_s02_ch1_prediction_fixed.tiff',
        'res': (400, 162.9, 162.9),
        'label': 'A1-2003 Series 02 (56 slices)',
    },
    'A1-2007': {
        'raw': '/projects/weilab/dataset/barcode/2026/umich/processed_ch1/A1-2007_ch1.tif',
        'pred': 'outputs/fiber_retrain_all/20260311_223801/results_multiregion_fixed/A1-2007_ch1_prediction_fixed.tiff',
        'res': (400, 162.9, 162.9),
        'label': 'A1-2007 (79 slices)',
    },
}

neuroglancer.set_server_bind_address(bind_address='0.0.0.0', bind_port=8888)

print("Loading all volumes...")
viewers = {}

for name, info in volumes.items():
    print(f"\nLoading {name}...")
    
    # Load raw
    raw_frames = []
    with Image.open(info['raw']) as img:
        for i in range(img.n_frames):
            img.seek(i)
            raw_frames.append(np.array(img))
    raw = np.stack(raw_frames, axis=0)
    
    # Load prediction
    pred_frames = []
    with Image.open(info['pred']) as img:
        for i in range(img.n_frames):
            img.seek(i)
            pred_frames.append(np.array(img))
    pred = np.stack(pred_frames, axis=0)
    
    print(f"  Raw: {raw.shape}, Pred: {pred.shape}")
    
    # Create viewer
    viewer = neuroglancer.Viewer()
    
    with viewer.txn() as s:
        s.layers['raw'] = neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(
                data=raw,
                dimensions=neuroglancer.CoordinateSpace(
                    names=['z', 'y', 'x'],
                    units=['nm', 'nm', 'nm'],
                    scales=info['res']
                ),
                voxel_offset=[0, 0, 0]
            ),
            shader="""
void main() {
  float val = toNormalized(getDataValue());
  emitGrayscale(val * 3.0);
}
"""
        )
        
        s.layers['segmentation'] = neuroglancer.SegmentationLayer(
            source=neuroglancer.LocalVolume(
                data=pred.astype(np.uint32),
                dimensions=neuroglancer.CoordinateSpace(
                    names=['z', 'y', 'x'],
                    units=['nm', 'nm', 'nm'],
                    scales=info['res']
                ),
                voxel_offset=[0, 0, 0]
            ),
        )
        
        s.layout = 'xy'
        s.position = [min(10, raw.shape[0]//2) * info['res'][0], 
                      raw.shape[1]//2 * info['res'][1], 
                      raw.shape[2]//2 * info['res'][2]]
    
    viewers[name] = viewer

print("\n" + "="*80)
print("NEUROGLANCER VIEWER URLS:")
print("="*80)
for name, viewer in viewers.items():
    label = volumes[name]['label']
    url = str(viewer).replace('http://a002.m31.bc.edu:8888/', 'http://localhost:8888/')
    print(f"\n{label}:")
    print(f"  {url}")

print("\n" + "="*80)
print("\nAll viewers are running. Press Ctrl+D to exit when done.")
print("Make sure your SSH tunnel is active: ssh -L 8888:localhost:8888 zhangdjr@a002.bc.edu")
