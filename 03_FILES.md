# 3. File Reference

Every file, what it does, and who depends on it.

---

## 3.1 Project root files

### `perfume_common.py` (144 lines) — the shared foundation

The only module everything else imports. Centralising this is what keeps the
offline and live paths in sync.

**Path setup (the reason import order matters):**
- `PROJ_DIR`, `FP_DIR` — this file's directory, and `FoundationPose/`.
- Inserts `FP_DIR` into `sys.path` **at import time**.
- Then `from estimater import *`.

> **Gotcha:** `perfume_common` must be imported *before* `Utils` or
> `estimater`. Otherwise `ModuleNotFoundError: No module named 'Utils'`.
> `live_g1.py` and the test scripts all import it first for this reason.

**Constants:** `DEFAULT_MESH` (`model/perfume_bottle_base.obj`),
`DEFAULT_DATASET` (`data/perfume`).

**`load_bottle_mesh(mesh_file=DEFAULT_MESH)`**
- `trimesh.load(..., force='mesh', process=False)` — `process=False` keeps the
  original vertex indexing, so the split vertices OBJ export leaves behind can
  be welded deliberately with `merge_vertices()` without losing faces.
- The CAD ships a `.mtl` declaring a flat diffuse colour but **no texture
  image**. trimesh still reports `TextureVisuals`, which makes
  `make_mesh_tensors` dereference a `None` image and crash. The fix is to
  replace the visuals with `ColorVisuals` using a flat grey
  (`Kd = 154,154,154`, i.e. 0.604) matching the material.

**`build_estimator(mesh, debug=0, debug_dir=None)`**
Instantiates `ScorePredictor()`, `PoseRefinePredictor()`,
`dr.RasterizeCudaContext()`, and returns
`FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, ...)`.
This is the single place the pre-trained refiner + scorer are wired in.

**`score_pose(est, pose, K, depth, mask=None, mesh=None)`** → `(silhouette, inlier_fraction, mask_iou)`
The project's central quality metric. Renders the CAD at `pose` with
`nvdiffrast_render`, then:
- `inlier_fraction` = fraction of rendered pixels where rendered and measured
  depth agree within **10 mm**. Needs no ground truth.
- `mask_iou` = IoU between the rendered silhouette and `mask`; **`None`** when
  no mask is passed, so "no mask" is never confused with "no overlap".
- Falls back to `est.mesh_ori` if `mesh` is omitted.

**`pose_to_dict(pose)` / `format_pose(pose, label='')`**
Splits the 4×4 into translation (m and mm), roll/pitch/yaw (deg, with the
singular-case branch for `sy < 1e-6`), a rotation vector, the **long axis in
camera coordinates**, and the raw matrix. rpy and the rotation vector are
derived from the same matrix so they cannot disagree. The output labels itself
*"(mod 180 about long axis)"* because the bottle is a rectangular prism and its
pose is only defined up to that spin.

### `run_perfume.py` (120 lines) — offline entry point

Registers on frame 0 (which has a mask), tracks the rest. Mirrors
`FoundationPose/run_demo.py`.

| Flag | Default | Meaning |
|---|---|---|
| `--mesh_file` | `DEFAULT_MESH` | CAD to use |
| `--test_scene_dir` | `DEFAULT_DATASET` | recorded sequence |
| `--est_refine_iter` | 5 | `register()` refinement iterations |
| `--track_refine_iter` | 2 | `track_one()` iterations |
| `--debug` | 1 | 0 none, 1 overlay, 2 also write PNGs |
| `--debug_dir` | `debug_perfume` | output root |
| `--max_frames` | 0 (all) | limit |
| `--no_show` | off | no OpenCV window |

Verified invocation: `run_perfume.py --no_show --debug 2` → 180 poses.

### `live_perfume.py` (478 lines) — live entry point, local D405

The largest project file. Docstring at the top states the workflow.

| Flag | Default | Meaning |
|---|---|---|
| `--mesh_file` | `DEFAULT_MESH` | CAD |
| `--width` / `--height` / `--fps` | 640 / 480 / 30 | stream config |
| `--warmup` | 15 | frames discarded after start, for auto-exposure |
| `--est_refine_iter` | 5 | `register()` |
| `--track_refine_iter` | 2 | `track_one()` |
| `--out` | `live_poses.jsonl` | pose log |
| `--snap_dir` | `live_snaps` | snapshot dir |

**Gates:** `MIN_FIT = 0.30`, `MIN_IOU = 0.25` (calibrated — see
[`05_POSE_QUALITY.md`](05_POSE_QUALITY.md)).

**Functions:**
- `start_pipeline(width, height, fps, warmup=15)` — starts depth `z16` + colour
  `rgb8`, builds `rs.align(rs.stream.color)`, discards `warmup` frames.
- `grab_frames(pipeline, align, depth_scale)` → `(rgb, depth, K)`.
  **`pipeline.wait_for_frames()` carries no timeout** — a deliberate decision
  (see [`08_RUNBOOK.md`](08_RUNBOOK.md) §7). Depth is converted to float
  metres; `K` is float64 to match what FoundationPose's refiner expects.
- `_depth_seeds(depth, x0, y0, x1, y1)` → `(near, seed_tol, bg_cut)` or `None`.
  Median of the closest surface cluster inside the box, plus a background cut.
- `_pick_and_guard(mask, depth, seeds, box, area_ref, K=None)` — shared tail of
  both mask paths: keeps the blob at the near surface (depth first, box centre
  as tiebreak, larger blob preferred on ties), then enforces the guards
  (min area, depth coverage ≥ 50%, distance ≤ 0.60 m, and the **CAD projected
  size check**).
- `cad_extents_mm()` — cached CAD extents in mm. The mesh is authored in
  **metres**, hence the ×1000.
- `expected_bbox_px(dist_m, K)` — largest plausible on-screen bbox of the
  bottle at a given distance, from the two largest CAD dimensions.
- `mask_from_box(rgb, depth, rect, iters=5, init_mask=None, K=None)` — two paths:
  - **drawn box** → `cv2.grabCut(..., GC_INIT_WITH_RECT)`, clipped to the
    rectangle, then `_depth_seeds` + `_pick_and_guard`.
  - **`init_mask`** → the remembered mask seeds GrabCut
    (`GC_INIT_WITH_MASK`) with a dilated unknown band around the edge, so a
    small move is absorbed. This is how auto-init re-registers without a box.
- `draw_overlay(vis, pose, bbox, K, line, extra)` — 3D box, axis triad, text.
- `main()` — the event loop described in [`02_ARCHITECTURE.md`](02_ARCHITECTURE.md) §2.3.

**Keys:** `b`/Space draw box and register · `r` release/re-register ·
`s` snapshot · `q`/ESC quit.

### `auto_init.py` (216 lines) — removing the per-run box

- `save_state(path, pose, mask, K, note='')` — write `init_state.npz`.
- `load_state(path)` → dict or `None`. Returns `None` on a missing file, an
  unreadable file, or a file missing any of `pose`/`mask`/`K`.
- `try_cached_init(est, K, rgb, depth, mesh, path, min_fit, mask_fn, verbose)`
  → `(pose, mask, how)`. Scores the saved pose; accepts it if it still fits,
  otherwise re-registers from the saved mask. `how` is one of
  `'no cache'`, `'resolution mismatch'`, `'cached pose'`, `'stale mask'`,
  `'register failed'`, `'re-registered'`, `'re-registration did not fit'`.
- `component_extents(labels, idx, depth, K)` — metric w/h of a labelled depth
  component at its own distance.
- `score_candidate(w_m, h_m, extents)` — 0 is a perfect match to a CAD box.
- `find_candidates(depth, K, ...)` — depth-threshold + morphology +
  connected components, filtered by CAD size. **This proved unreliable; see
  [`06_AUTO_INIT.md`](06_AUTO_INIT.md) §4.**
- `auto_register(est, K, rgb, depth, mask_fn, mesh, ...)` — try the top 4
  candidates through the same `mask_fn` and accept the first that fits.
- `STATE_PATH` — `perfume_tracking/init_state.npz`.

### `prepare_mesh.py` (85 lines) — CAD preparation

Reads the source mesh, writes a **metric, base-centered** copy so the reported
pose refers to the centre of the bottle's base (origin at the bottom face, +Y
up). The transform is a rigid translation only — it cannot change accuracy,
only which point the pose refers to. Produced `model/perfume_bottle_base.obj`.

### Verification / analysis scripts

| File | Purpose |
|---|---|
| `check_tracking.py` (83) | Per-frame depth agreement for a pose file. **No ground truth needed**, so it also works on live output. Reports mean/min and counts below a tolerance. |
| `check_iou.py` (88) | Silhouette IoU between CAD-rendered-at-pose and the annotated mask. The decisive model-based check: 0.90+ very good, 0.80–0.90 solid, <0.70 wrong. |
| `verify_pose.py` (94) | Independent sanity check: transforms and projects the CAD corners with the scene intrinsics and compares to the mask, and reports where the bottle's own axes land in camera frame. Catches flipped/transposed rotations. Deliberately does not use FoundationPose internals. |

---

## 3.2 Data and model

| Path | Contents |
|---|---|
| `model/perfume_bottle_base.obj` | **The CAD used for estimation.** 6068 verts, 12128 faces, extents **54.98 × 153.0 × 30.0 mm**, bounds X ∈ [−27.49, 27.49], Y ∈ [0, 153], Z ∈ [−15, 15]. Not watertight (fine — FoundationPose only needs the surface). |
| `model/perfume_bottle.obj` | Original source mesh, not base-centered. |
| `model/material.mtl`, `model/perfume_bottle.mtl` | Materials. `material.mtl` Kd = 0.604 → the 154 grey in `load_bottle_mesh`. |
| `model/source/` | Original CAD drop. |
| `data/perfume/rgb/` | 180 PNG frames. |
| `data/perfume/depth/` | 180 PNG frames (16-bit, mm). |
| `data/perfume/masks/000000.png` | The single annotated mask (frame 0 only). |
| `data/perfume/cam_K.txt` | `fx=392.76, fy=392.18, cx=318.19, cy=245.25` — the **D405** intrinsics. |
| `data/perfume/mask_overlay.png` | Mask visualisation for eyeballing. |

---

## 3.3 `FoundationPose/` (vendored NVIDIA)

Commit `a1b694b`. Not modified except for the added `PerfumerReader` class in
`datareader.py`.

Relevant pieces: `estimater.py` (`FoundationPose.register/track_one`),
`Utils.py` (`nvdiffrast_render`, `make_mesh_tensors`, `draw_xyz_axis`,
`draw_posed_3d_box`, `set_seed`, `set_logging_format`), `datareader.py`,
`weights/2023-10-28-18-33-37/` (247 MB, `model_best.pth` + `config.yml` — this
is the checkpoint actually loaded, per the log line
`Using pretrained model from .../2023-10-28-18-33-37/model_best.pth`).

---

## 3.4 `/tmp/opencode/` — scratch, NOT part of the project

**These live in `/tmp` and will not survive a reboot.** Anything needed to
reproduce the G1 work should be moved into the project. Current contents:

| File | Role |
|---|---|
| `live_g1.py` | **G1 live viewer.** Imports `perfume_common`, `live_perfume`, `auto_init` from the project, adds the TCP client, colour conversion, auto-init and snapshot-on-attempt. |
| `g1_stream.py` | **G1 TCP streamer** (robot side). Deployed to `/home/unitree/g1_stream.py`. |
| `g1_burst.py` | Capture aligned RGB+depth+K to an `.npz` for offline pose testing. Deployed to `/home/unitree/g1_burst.py`. |
| `g1_soak.py`, `rs_probe.py`, `rs_ladder.py` | USB/RealSense diagnostics for the robot. |
| `pose_on_g1.py` | Offline registration+tracking over a captured burst. |
| `test_cache.py` | **Validates cached auto-init** (register once → reuse / re-register / refuse). |
| `test_e2e_init.py` | End-to-end offline register+track on recorded data; reports `dt` between frames. |
| `test_thresholds.py` | Calibrates `MIN_FIT` over the 180 recorded frames. |
| `test_gate.py` | Confirms the gate accepts a good pose and rejects a shifted one. |
| `test_symmetry.py` | Measures the real mesh symmetry. |
| `diag_auto.py` | Pose-projected diagnostics showing why depth-only auto-detect fails. |
| `test_auto_init.py` | Original auto-detect test (flawed ground truth; kept to reproduce history). |
| `test_depth_guard.py`, `test_live_*.py`, `see_frame.py`, `mesh_preview.py`, `tc.py` | Smaller probes. |
| `askpass.sh` | SSH askpass helper (mode 700). |
| `g1` | Reusable remote command wrapper. |
| `attempts/` | Saved registration attempts (rgb, depth, K, mask, rect) for offline debugging. |

---

## 3.5 `perfume_foundationpose/` — parallel implementation, unused

A separate implementation with a `--symmetry_y` option. **Not merged and not
used.** It cannot solve the top-up/top-down ambiguity either — see
[`09_EXPERIMENT_LOG.md`](09_EXPERIMENT_LOG.md) §4. Do not consolidate it
without explicit agreement.
