# ND2 Pipeline Path Mapping (Batch 2)

This file documents the current path mapping after Batch 2 restructuring.

## Active Runtime Paths (current)

- `pipelines/nd2/run_pipeline.sh`
- `pipelines/nd2/run_pipeline_batch.sh`
- `slurm_jobs/nd2/step1_extract_tiles.sl`
- `slurm_jobs/nd2/step2_infer_tiles_array.sl`
- `slurm_jobs/nd2/step3_postprocess.sl`

These are the canonical scripts used by launcher flow.

## Compatibility Wrapper Paths

- `slurm_jobs/run_pipeline.sh` -> `pipelines/nd2/run_pipeline.sh`
- `slurm_jobs/run_pipeline_batch.sh` -> `pipelines/nd2/run_pipeline_batch.sh`
- `run_nd2_pipeline.sh` -> `pipelines/nd2/run_nd2_pipeline_legacy.sh`

## Additional Paths

- `pipelines/nd2/run_pipeline.sh`
- `pipelines/nd2/run_pipeline_batch.sh`
- `pipelines/nd2/run_nd2_pipeline_legacy.sh`
- `slurm_jobs/nd2/step3_cell_seg_array.sl`
- `slurm_jobs/nd2/step4_fiber_array.sl`
- `slurm_jobs/nd2/step5_merge_outputs.sl`
- `tools/extract_nd2_tile.py`
- `tools/generate_fiber_coordinates.py`
- `tools/cell_seg_microsam.py`
- `tools/fiber_pipeline.py`
- `tools/neuroglancer_all_nd2_tiles.py`
- `tools/neuroglancer_all_volumes.py`
- `tools/stitch_nd2_to_h5.py`
- `tools/infer_stitched_strips.py`
- `tools/resample_volumes.py`
- `tools/view_multiregion_predictions.py`
- `tools/verify_a1.py`
- `tools/verify_a2.py`
- `tools/eval_standalone.py`

## Temporary Config Output

Step 2 now writes generated per-tile YAML files to:

- `<nd2_root>/meta/tmp/`

instead of writing into `tutorials/`.
