# 2. Architecture

## 2.1 Layering

```
+------------------------------------------------------------------+
|  Entry points                                                     |
|  run_perfume.py     offline, recorded sequence (verified)          |
|  live_perfume.py    live, local RealSense D405                     |
|  live_g1.py         live, remote D435i over TCP  (/tmp/opencode)  |
+------------------------------------------------------------------+
|  Project services                                                 |
|  perfume_common.py   paths, CAD load, estimator build, score_pose  |
|  auto_init.py        cached registration, size-based detection     |
+------------------------------------------------------------------+
|  Mask extraction (live only)                                      |
|  mask_from_box()  -> GrabCut + depth seeds + guards  (live_perfume)|
+------------------------------------------------------------------+
|  FoundationPose/  (vendored, NVIDIA)                              |
|  estimater.py  FoundationPose.register() / .track_one()           |
|  Utils.py       nvdiffrast_render, draw_xyz_axis, ...             |
|  datareader.py  PerfumerReader  (added by this project)           |
+------------------------------------------------------------------+
|  Weights:  FoundationPose/weights/2023-10-28-18-33-37/           |
+  CAD:      model/perfume_bottle_base.obj                           |
|  Dataset:  data/perfume/{rgb,depth,masks,cam_K.txt}                |
+------------------------------------------------------------------+
```

The dependency rule is one-directional: entry points import project services;
project services import FoundationPose. `perfume_common` must be imported
**before** `Utils`, because it is what puts `FoundationPose/` on `sys.path`
(importing `Utils` first raises `ModuleNotFoundError: No module named 'Utils'`).

## 2.2 The offline data flow

`run_perfume.py`:

```
PerfumerReader(data/perfume)
  ├── get_color(i) -> RGB   (imageio, INTER_NEAREST)
  ├── get_depth(i) -> float metres (/1e3, <0.001 m zeroed)
  ├── get_mask(i)  -> bool  (all-False for unannotated frames)
  └── K            -> 3x3 from cam_K.txt
        |
        v
   frame 0:  est.register(K, rgb, depth, ob_mask=mask, iteration=5)
   frame i:  est.track_one(rgb, depth, K, iteration=2)
        |
        +--> debug_perfume/ob_in_cam/<frame>.txt   (4x4, one per frame)
        +--> debug_perfume/poses.json              (all frames, nested list)
        +--> debug_perfume/track_vis/*.png         (debug >= 2)
        +--> stdout: format_pose() every 20 frames
```

`PerfumerReader` is a small class added to `FoundationPose/datareader.py`. It
mirrors `YcbineoatReader`'s on-disk layout but tolerates a sequence where only
frame 0 is annotated with a mask: `get_mask` returns an all-False mask for
unannotated frames instead of raising. Only `register()` consumes the mask and
it runs on frame 0 only, so the rest are never needed.

## 2.3 The live data flow (local D405)

`live_perfume.py` is a single-threaded event loop. There is no separate render
thread; `cv2.imshow` + `cv2.waitKey(1)` drives it.

```
start_pipeline()
  pipeline.start(depth z16 + color rgb8)
  rs.align(rs.stream.color)        <-- depth is warped into the COLOUR frame
  discard `warmup` (=15) frames     <-- auto-exposure / white balance settling
        |
        v
loop:
  grab_frames()  -> rgb uint8, depth float32 metres, K float64 3x3
                   NOTE: pipeline.wait_for_frames() has NO timeout
  |
  +-- pose is None --> draw hint text, show frame
  |
  +-- pose exists  --> est.track_one(...)
                       |
                       +--> append pose to <out>.jsonl
                       +--> every 5th frame: score_pose()
                       |      inlier < MIN_FIT -> bad_frames++, warn at 5
                       +--> draw_overlay(): 3D box, axis triad, text
  |
  +-- key b/Space -> cv2.selectROI -> mask_from_box -> est.register()
  |                  -> score_pose() -> accept/reject
  +-- key r       -> pose = None (back to the draw-a-box state)
  +-- key s       -> save PNG + 4x4 txt to live_snaps/
  +-- key q/ESC   -> break
```

Two details that are easy to get wrong and are handled explicitly:

- **Intrinsics come from the colour stream**, because depth was aligned into the
  colour frame. Using depth intrinsics with aligned depth is a subtle mismatch.
- **The queue is drained after `selectROI`.** While the user is dragging a box
  the loop is blocked, so the RealSense queue fills and hands back stale frames
  the moment the loop resumes. Five frames are dropped to clear it.

## 2.4 The live data flow (G1 robot, remote camera)

The robot cannot run FoundationPose (aarch64 vs an x86/CUDA `sm_120` build), so
it only *acquires* frames. A tiny TCP server on the robot does the RealSense
work; the local viewer does everything else.

```
   G1 (192.168.1.39, aarch64)                    Local machine (x86-64 + RTX 5080)
   ------------------------------------          ------------------------------------
   g1_stream.py
     rs.pipeline(depth z16 + color bgr8)
     rs.align(color)
     |
     |  TCP 8766, 4-byte big-endian length prefix
     |  payload = [color_len u32][depth_len u32][frame_idx u32]
     |            [JPEG colour][zlib-compressed raw u16 depth]
     v
   listening socket  <-------------------------------  live_g1.py
                                                          |
   camera outage?                                        |
     keep socket open, retry pipeline <------------------ +--> est.track_one()
     (client waits; does not reconnect)                       score_pose()
                                                          draw_overlay()
```

Deliberate properties of this design:

- **The robot never needs the project's Python environment.** It only needs
  `pyrealsense2`, `opencv`, `numpy`. It cannot import `perfume_common`.
- **Depth is sent as raw `uint16`** in the sensor's own units, and the client
  applies the `depth_scale` from the header. Sending metres truncated to
  integers would destroy the data — this was checked.
- **A camera outage does not drop the client.** The server holds the socket
  open and reopens the pipeline; the client waits (180 s tolerance) rather than
  reconnecting, so a session is not lost because the USB flapped.
- **Every accepted client gets the header.** A client connecting *after* an
  outage would otherwise read frame bytes as a header — this bug happened and
  is fixed.
- Colour is sent as `bgr8` (what RealSense natively produces) and the viewer
  converts to RGB for drawing, then back to BGR for `cv2.imshow`. Getting this
  wrong swaps red and blue in the preview.

## 2.5 State

| State | Where | Lifetime |
|---|---|---|
| Current pose | `pose` variable in the live loop | process |
| Pose history | `live_poses.jsonl` / `live_g1_poses.jsonl`, one JSON object per line: `{frame, track, pose}` | append-only, across runs |
| Cached registration | `init_state.npz` — `pose` (4×4), `mask` (bool H×W), `K` (3×3), `note` | across runs |
| Offline poses | `debug_perfume/poses.json`, `debug_perfume/ob_in_cam/*.txt` | per run |
| Snapshots | `live_snaps/`, `live_snaps_g1/` (PNG + 4×4 txt) | on `s` |

`init_state.npz` is the mechanism that removes the per-run bounding box; see
[`06_AUTO_INIT.md`](06_AUTO_INIT.md).
