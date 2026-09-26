# Perfume Bottle 6D Pose — Project Documentation

Estimates the **6D pose** (3D position + 3D orientation) of a perfume bottle
using **FoundationPose**, from an Intel RealSense depth camera.

Two deployment targets exist:

| Target | Camera | Where pose is computed | Status |
|---|---|---|---|
| **Offline / recorded** | D405 (local, recorded sequence) | local RTX 5080 | **Working, verified** |
| **Live / G1 robot** | D435i on Unitree G1 (aarch64 Jetson Orin NX) | local RTX 5080 (frames streamed over LAN) | **Streaming works; auto-init not yet validated on G1** |

---

## Read these, in this order

| # | Document | What it covers |
|---|---|---|
| 1 | [`01_OVERVIEW.md`](01_OVERVIEW.md) | The problem, the approach, and why each choice was made |
| 2 | [`02_ARCHITECTURE.md`](02_ARCHITECTURE.md) | Components, data flow, threading, the two camera paths |
| 3 | [`03_FILES.md`](03_FILES.md) | Every file: what it does, what it exports, who calls it |
| 4 | [`04_FOUNDATIONPOSE.md`](04_FOUNDATIONPOSE.md) | How FoundationPose is wired in, the CAD convention, `register()` vs `track_one()` |
| 5 | [`05_POSE_QUALITY.md`](05_POSE_QUALITY.md) | The `score_pose()` metric, threshold calibration, quality gates |
| 6 | [`06_AUTO_INIT.md`](06_AUTO_INIT.md) | Removing the per-run bounding box: cached registration |
| 7 | [`07_G1_SETUP.md`](07_G1_SETUP.md) | Robot setup, D435i quirks, USB/port issues, the streaming protocol |
| 8 | [`08_RUNBOOK.md`](08_RUNBOOK.md) | Commands to run things, troubleshooting, known issues |
| 9 | [`09_EXPERIMENT_LOG.md`](09_EXPERIMENT_LOG.md) | Chronological findings with the actual measured numbers |

---

## Quick start (offline, verified)

```bash
FP=/home/priyanshi/miniconda3/envs/foundationpose
cd /home/priyanshi/Downloads/perfume_tracking
export CUDA_HOME=$FP/cuda_home PATH=$FP/cuda_home/bin:$PATH \
       LD_LIBRARY_PATH=$FP/cuda_home/lib64 \
       PYTHONPATH=/home/priyanshi/.cache/torch_extensions/py311_cu128/nvdiffrast_plugin
$FP/bin/python -u run_perfume.py --no_show --debug 2
```

Produces 180 poses in `debug_perfume/poses.json`. See
[`08_RUNBOOK.md`](08_RUNBOOK.md) for the live variants.

---

## The one-paragraph version

A perfume bottle CAD (`model/perfume_bottle_base.obj`, 55 × 153 × 30 mm,
base-centered) is fed to FoundationPose together with an RGB frame, an
aligned depth map, camera intrinsics, and a 2D object mask. FoundationPose
registers the CAD on the first frame (a full 6D search) and then tracks it
frame-to-frame (a fast local refinement). The project adds three things on
top of stock FoundationPose: a **ground-truth-free quality metric**
(`score_pose`) so a diverged pose is detected rather than trusted, **calibrated
gates** on that metric so a bad initialisation is rejected before tracking
starts, and a **cached registration** so the user does not have to redraw a
bounding box on every run.
