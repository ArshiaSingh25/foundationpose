# Perfume bottle — FoundationPose

Run 6D pose estimation and tracking for the CAD in [`model/perfume_bottle.obj`](../model/perfume_bottle.obj) on the capture under [`data/perfume/`](../data/perfume/).

## Prerequisites

- FoundationPose installed under [`FoundationPose/`](../FoundationPose/) with weights (see upstream readme).
- Conda env `foundationpose` (or equivalent) with CUDA.
- Capture layout (same as YCB-In-EAT demo):

  ```
  data/perfume/
    cam_K.txt
    rgb/*.png
    depth/*.png      # uint16 depth in millimeters
    masks/000000.png # required for frame-0 registration only
  ```

## Quick start

```bash
conda activate foundationpose
cd /home/priyanshi/Downloads/perfume_tracking/perfume_foundationpose

# If you need a first-frame mask:
python annotate_first_frame_mask.py

# Full sequence (poses + saved overlays) — use run.sh so nvdiffrast finds torch libs:
chmod +x run.sh
./run.sh --debug 2

# Smoke test (first 10 frames):
./run.sh --max_frames 10 --debug 2
```

The runner copies a **vertex-colored** mesh to `output/perfume_bottle_fp.obj` from `model/perfume_bottle.obj` (the CAD references materials but has no texture image, which FoundationPose requires).

## Outputs

- `output/foundationpose_perfume/ob_in_cam/<frame>.txt` — 4×4 object-in-camera poses
- `output/foundationpose_perfume/track_vis/<frame>.png` — RGB with bbox and axes

## Options

| Flag | Description |
|------|-------------|
| `--symmetry_y` | Discrete rotation symmetry about Y (useful if the bottle is symmetric) |
| `--max_frames N` | Limit frames for debugging |
| `--mesh_file` / `--scene_dir` / `--debug_dir` | Override defaults |
| `--debug 2` | Save overlay PNGs (default) |
| `--debug 3` | Also open a live OpenCV window (needs `DISPLAY`) |

Validate capture data:

```bash
python check_data.py
```

## Verify environment only

```bash
cd ../FoundationPose && python check_env.py
```
