# 1. Overview and Approach

## 1.1 The problem

Given a perfume bottle sitting on a table, and a depth camera watching it,
compute where the bottle is and how it is oriented, in real time, from a CAD
model — with no other sensor, no fiducial marker, and no ground-truth pose
available at run time.

Concretely, per frame we want a 4×4 homogeneous transform `T` such that a point
`p` in the CAD's own coordinate frame appears at `T · p` in the camera frame.

## 1.2 Why FoundationPose

The bottle is a rigid, known shape with a clean CAD model, so a **model-based**
(synthetic-data) pose estimator is the right tool. FoundationPose
(NVIDIA, `FoundationPose/`) was chosen because:

- **It needs only a CAD model**, not a real object with textured photographs.
  This matters: there is no reference capture of this bottle in the wild.
- **It is robust to occlusion and to textureless surfaces.** Perfume glass and
  coloured plastic are exactly the cases where template matching or
  feature-based methods (e.g. PoseCNN, RGB-D keypoints) struggle.
- **It separates registration from tracking.** `register()` does an expensive
  global search and needs a mask; `track_one()` is a cheap per-frame refinement
  that needs only the previous pose. This is the correct structure for live use:
  pay the search cost once, then track.
- It runs on a single consumer GPU. The local RTX 5080 is far faster than
  anything on the robot.

Alternatives considered and rejected: dense point-cloud ICP (needs a good
initial guess and is slow), AprilTag/ArUco (needs a marker glued to a
transparent bottle), and learned RGB-D pose networks (need per-object training
data).

## 1.3 The approach, step by step

```
   CAD (model/perfume_bottle_base.obj)
                    |
                    v
   +------------------------------------+
   |  FoundationPose estimator          |
   |  register()  -> full 6D search     |   frame 0, needs a mask
   |  track_one() -> fast refinement    |   frames 1..N, needs prev pose
   +------------------------------------+
        ^              ^
        |              |
   object mask     RGB + depth + K
   (GrabCut)       (RealSense, depth aligned to colour)
```

1. **Acquire** an RGB frame, a depth frame, and colour-stream intrinsics `K`.
   Depth is *aligned into the colour camera frame* so that a 2D mask drawn on
   the colour image indexes the same pixels FoundationPose uses for depth.
2. **Get an object mask.** The user drags a box; GrabCut refines it to a mask.
   (This is the step being automated away — see
   [`06_AUTO_INIT.md`](06_AUTO_INIT.md).)
3. **Register** the CAD on that frame: `est.register(K, rgb, depth, ob_mask)`.
   This is the only step that needs the mask.
4. **Track** every subsequent frame: `est.track_one(rgb, depth, K)`, seeded
   from the previous pose.
5. **Score every pose** against the measured depth with `score_pose()`. A pose
   that no longer explains the frame is flagged, not silently accepted.
6. **Report** translation in mm (referring to the centre of the bottle's base)
   plus roll/pitch/yaw, and also the bottle's long axis in camera coordinates,
   which is unambiguous where Euler angles are not.

## 1.4 Design decisions and why

### Base-centered CAD (`prepare_mesh.py`)

The pose is reported as the **centre of the bottle's base**, not the CAD's
original origin. For an object that stands on a table this is the most
interpretable convention: `translation_mm` is directly "how far the bottle's
footprint centre is from the camera". The transform applied is a pure
translation, so it cannot change FoundationPose's accuracy — only which point
the answer refers to.

### Depth agreement as the quality metric (`score_pose`)

The central problem with pose tracking is that **it fails silently**. A
diverged tracker returns confident, plausible-looking numbers, so the failure
is only noticed when the overlay is visibly in the wrong place — which may be
thousands of frames later.

`score_pose()` renders the CAD at the estimated pose and compares the rendered
depth with the measured depth over the rendered silhouette:

```
inlier_fraction = fraction of rendered pixels where |rendered_depth - measured_depth| < 10 mm
```

A correct pose puts the CAD surface on the real surface, so this stays high. A
diverged pose makes the CAD float in mid-air or sink through the table, and the
fraction collapses. Crucially this needs **no ground truth**, so it works
identically on recorded data and on a live feed.

### Calibrated gates, not guessed ones (`MIN_FIT`, `MIN_IOU`)

The thresholds `MIN_FIT = 0.30` and `MIN_IOU = 0.25` were not chosen by
intuition. All 180 recorded frames were scored with their (verified-good)
offline poses, giving a minimum of 43.7% and a median of 58.9%, while a
deliberately displaced pose scores 0%. A 50% gate would have falsely rejected
21 of the 180 good frames. 30% sits in the empty gap. Full numbers in
[`05_POSE_QUALITY.md`](05_POSE_QUALITY.md).

### Never trust a mask blindly

A mask failure is the most common cause of a bad initialisation, and it is
invisible: GrabCut will happily return a small blob of background. The project
therefore cross-checks the mask's **projected size against the CAD's known
dimensions** at the mask's own measured distance, and rejects the registration
before it is allowed to start if the two disagree. This was added after a real
failure — see [`09_EXPERIMENT_LOG.md`](09_EXPERIMENT_LOG.md) §7.

### Compute locally, not on the robot

The G1's Jetson Orin NX is aarch64. The FoundationPose build here is x86-64
with a CUDA `nvdiffrast` extension compiled for `sm_120`; standing up an
equivalent ARM build is a substantial separate effort. The G1 already runs a
Python RealSense stack, so the D435i is opened **there** and the frames are
streamed over LAN to the local machine, which does the pose work. Measured cost
is ~3.3 MB/s at 28 fps, which is trivial for gigabit Ethernet. See
[`07_G1_SETUP.md`](07_G1_SETUP.md).

## 1.5 What is verified and what is not

**Verified:**
- Offline: 180/180 poses, mask IoU 0.40 on the annotated frame, mean depth
  agreement 81.2% (min 67.9%), zero frames below 60%.
- Threshold calibration over all 180 frames.
- Cached auto-initialisation: reuse when unchanged, re-register from the saved
  mask when shifted, correctly refuse when the bottle is absent.
- G1 streaming: sustained 28–29 fps, ~3.3 MB/s, colour correct.

**Not verified / known broken:**
- **Fast-motion divergence on the live path.** Tracking a hand-moved bottle
  produces jumps up to 2171 mm and 136.9° between consecutive frames. The
  tracker currently *warns* but still accepts, renders and logs the bad pose.
  This is the top open item; see [`05_POSE_QUALITY.md`](05_POSE_QUALITY.md) §5.
- **G1 camera stability.** The D435i enumerates as USB 2.0 high-speed and drops
  the stream repeatedly. Needs a true USB 3.x port or hub.
- **Auto-initialisation has never completed on the G1**, because the bottle was
  not in frame during the last attempts. The mechanism is verified offline.
- The D435i path has only ever had a real registration attempt on a frame where
  the mask was wrong; the size guard that now catches this has been unit-tested
  but not yet exercised by a successful G1 registration.
