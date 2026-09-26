# 6. Automatic Initialisation

**Goal:** the user should have to indicate the bottle **once**. Every later run
should produce a pose with no interaction at all.

## 6.1 The problem being solved

Both live loops began in a state where `pose is None` and the only way out was
to press `b` and drag a box. The user's complaint, in effect: *why do I have to
select the bounding box again and again?*

Three approaches were considered:

| Approach | Verdict |
|---|---|
| Detect the bottle from scratch each run | **Rejected** — see §4 |
| Reuse the previous run's pose directly | Unsafe — the camera or the bottle may have moved |
| **Re-validate a cached registration, then fall back** | **Implemented** |

## 6.2 The mechanism

State file: `perfume_tracking/init_state.npz` (`auto_init.STATE_PATH`).

```
np.savez_compressed(path,
    pose = 4x4 float64,
    mask = HxW uint8,
    K    = 3x3 float64,
    note = str)          # provenance, e.g. 'live_g1 manual'
```

Roughly 2 KB, because the mask is stored as a compressed bitmap.

`try_cached_init(est, K, rgb, depth, mesh, path, min_fit=0.30, mask_fn)`:

```
1. load_state(path)                        -> None if missing/unreadable/incomplete
2. if saved mask shape != depth shape      -> 'resolution mismatch'
3. score_pose(saved pose) on the live frame
     fit >= min_fit  -> RETURN (pose, saved mask, 'cached pose')     # nothing moved
4. else re-register from the saved mask:
     mask_from_box(rgb, depth, None, init_mask=saved mask, K=K)
         GrabCut seeded by GC_INIT_WITH_MASK:
           sure FG  = eroded seed (5x5, x2)
           sure BG  = outside a dilated seed (15x15, x3)
           PR FG    = the seed itself
         so a small move is absorbed by the unknown band at the edge
     est.register(...) on that mask
     score_pose(new pose)
       fit >= min_fit -> RETURN (..., 're-registered')               # small move
5. otherwise RETURN (None, None, reason)     -> caller asks for a box
```

`save_state` is called after **any** accepted registration — manual or
automatic — in both live viewers, so the cache stays current.

## 6.3 Verified behaviour

From `/tmp/opencode/test_cache.py` on the 180-frame recorded sequence
(register once on frame 0, then re-initialise with **no box drawn**):

| Frame | Method | Result |
|---|---|---|
| 0 | `cached pose` | fit 54% |
| 5 | `cached pose` | fit 49% |
| 30 | **`re-registered`** from cached mask | fit 60%, IoU 0.28 — accepted |
| 90 | **`re-registered`** from cached mask | fit 69%, IoU 0.24 — accepted |
| 150 | `cached pose` | fit 74% |
| 179 | `cached pose` | fit 74% |
| frame with **no bottle** (flat 1.5 m wall) | **refused** | fit 0%, mask rejected as too far |

The three behaviours that matter are all demonstrated: **reuse** when nothing
moved, **re-register** when it moved slightly, and **refuse** when the bottle is
not there. That last one is what stops a stale cache from producing a confident
wrong pose.

The `mask_fn` call is wrapped in `try/except TypeError` to fall back to a
`mask_fn` that does not accept `K`, and `_pick_and_guard` skips the projected-
size check when `K is None` rather than crashing.

## 6.4 Why depth-only detection was abandoned

`find_candidates()` + `auto_register()` still exist in `auto_init.py` as a
secondary path, but they are **not used** and should not be relied on.

Intended idea: threshold depth to 0.12–1.20 m, morphological-close to fill
stereo dropout, take connected components, and accept any whose metric
projected size matches the CAD's 55 × 153 × 30 mm.

Measured outcome on the recorded data:

- An initial run reported 0/180 detections, which prompted a check of the test
  itself — the ground truth was wrong (it used the scene's *median* depth rather
  than the pose-projected depth).
- With correct pose-projected ground truth, the picture was still bad:

| Measurement | Value |
|---|---|
| Scene depth range (p2 – p98) | **0.18 – 0.26 m** |
| Bottle distance | ~0.186 – 0.195 m |
| Background median | 0.20 – 0.23 m |
| Candidates found | none in most frames |
| When one was found | a **wrong** region, 19 × 14 mm |

The scene is only ~4 cm deep in total. **The bottle and its background are tens
of millimetres apart, and thresholding cannot separate them.** The method fails
not because of a tuning problem but because there is no depth signal to exploit.
A trained detector or a colour/texture feature would be required; that is a
larger project and is not started.

This is the honest reason the cached-registration approach was chosen instead:
it exploits a fact that *is* reliable (the user already told us where the bottle
is, once) rather than one that is not (its depth signature).

## 6.5 Scope and limits — read before relying on it

- The cache is valid **only while the scene is sufficiently unchanged**. A
  different table, a moved camera, or a different bottle invalidates it — and
  step 5 will (correctly) refuse rather than guess, falling back to a box.
- The mask is stored at full frame resolution, so a resolution change is
  detected (`resolution mismatch`). A *focal length* change at the same
  resolution is **not** detected, though the `score_pose` re-validation would
  almost certainly catch it.
- `min_fit` defaults to 0.30, the same value as `MIN_FIT`, so the cache and the
  live gate agree. Both are calibrated for the D405; **they have not been
  recalibrated for the D435i** (see [`05_POSE_QUALITY.md`](05_POSE_QUALITY.md) §5.2).
- The user must still be present for the first run of a new scene. The goal was
  "draw it once", not "never draw a box".

## 6.6 Not yet exercised on the G1

`live_g1.py` has the full auto-init path wired in — `--init_state` (default
`STATE_PATH`), `--no_auto_init` to disable, an `auto_tried` sentinel so the
first frame attempts the cache, and `r` resetting it. It has **not yet
completed a successful G1 registration**, because during the last attempts the
bottle was not in the camera's field of view (nearest depth in frame 1.25 m).
`init_state.npz` does not yet exist.
