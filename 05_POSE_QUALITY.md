# 5. Pose Quality: Metrics, Gates, and Known Divergence

## 5.1 Why a quality metric is the core of this project

Pose tracking fails **silently**. A diverged tracker keeps returning
confident, well-formed, plausible-looking numbers. Without a check you find
out minutes later, from a log full of numbers that are all individually
reasonable and collectively nonsense.

Every gate in this project exists to convert a silent failure into a loud one.

## 5.2 `score_pose` — the primary metric

```python
_, inlier, iou = score_pose(est, pose, K, depth, mask=mask, mesh=mesh)
```

1. Render the CAD at `pose` → silhouette `sil` and rendered depth `rd`.
2. Restrict to pixels where the CAD is visible **and** both depths are valid:
   `both = sil & (rd > 1e-3) & (depth > 1e-3)`.
3. `inlier = mean(|rd - depth| < 0.010)` over `both` — the fraction of the
   rendered object whose predicted surface is within **10 mm** of the real one.
4. If a mask was given, `iou = |sil ∩ mask| / |sil ∪ mask|`, else `None`.

**Why depth agreement and not something else:** it needs no ground truth and no
marker, so the identical function validates offline results and gates live
frames. A correct pose lies *on* the real surface; a drifted one floats in air
or sinks through the table, and the fraction collapses.

**Known limitations, stated honestly:**
- 10 mm is a fixed tolerance. On the D405 at ~0.19 m this is comfortable; on
  the D435i at 1.2 m the same 10 mm is a much stricter relative demand, and
  stereo quantisation at range is coarser than 10 mm. **The metric has not been
  recalibrated for the D435i.** This matters for the G1 thresholds.
- Only 2% of the bottle's surface is observed in a given frame, so `inlier` is
  an unbiased but noisy estimate of pose quality.
- It is symmetric: a pose 180°-symmetric to the truth scores identically,
  which is correct behaviour, not a bug.
- Glass and specular surfaces give holes in the depth map, which lowers the
  achievable ceiling. That is why the D405 bottle tops out near 87% rather
  than 100%.

## 5.3 Threshold calibration

All 180 recorded frames were scored using their offline poses, which are
independently verified good:

| Population | `inlier` |
|---|---|
| Good poses, 180 recorded frames | min **43.7%**, median **58.9%**, max **87.3%** |
| Deliberately displaced pose | **0%** |

So the gap is `0% … 43.7%`. `MIN_FIT = 0.30` sits inside it with margin on both
sides. A 50% gate was rejected because it would have falsely failed **21 of
the 180 good frames** — the distribution is wide, so the gate has to respect
its lower tail, not its median.

`MIN_IOU = 0.25` is chosen against the one annotated mask, which scores
**IoU 0.40** at the correct pose. `check_iou.py`'s own docstring puts 0.80–0.90
as "solid", but that is for poses verified against a hand-drawn mask on a
well-segmented object; 0.40 is what a *good* FoundationPose pose actually
achieves here, so a 0.25 floor is the sensible line.

**Caveat carried forward:** `check_tracking.py` reports a *different* number
(mean 81.2%, min 67.9%) for the same poses, because it is a different
aggregation over a per-frame basis. The `MIN_FIT` calibration is specifically
the `score_pose` population above. Do not mix the two figures.

## 5.4 Where the gates are applied

| Point | Metric | Action on failure |
|---|---|---|
| After `register()` in `live_perfume.do_select` | `inlier` **and** `iou` | Refuse the pose, stay unregistered, tell the user to redraw |
| After `register()` in `live_g1.do_select` | `inlier` **and** `iou` | Same |
| Every 5th tracked frame | `inlier` | `bad_frames++`; print a warning at 5 consecutive; overlay shows `LOST` |
| After cached re-registration | `inlier` | Reject, fall back to asking for a box |
| **Between consecutive tracked frames** | — | **NOT IMPLEMENTED — see §5.5** |

## 5.5 Known problem: fast-motion divergence (OPEN, top priority)

### The measurement

`live_poses_run2.jsonl`, 4416 poses from the live D405 run while the bottle was
being moved by hand:

| Statistic | Value |
|---|---|
| Frames | 4416 |
| Max translation step between frames | **2171 mm** |
| Second largest | 865 mm |
| Max rotation step | **136.9°** |
| p50 rotation step | 5.6° |
| p90 | 17.5° |
| p99 | 48.7° |
| Frames with rotation step > 40° | 87 |
| Final region of the trajectory | x ∈ [−1604, 557], y ∈ [−510, 1221], z ∈ [−521, 824] mm |

The pose walked more than a metre away from the bottle. The first run
(`live_poses.jsonl`, 331 poses) shows the same failure at smaller amplitude.

### Why the existing gate does not catch it

`score_pose` **does** drop below `MIN_FIT` when this happens — the warning
prints and the overlay shows `LOST`. But:

- it is only evaluated every 5th frame, and
- **failing the gate changes nothing**: the pose is still written to the JSONL,
  still rendered as the current pose, and still used as the seed for the next
  `track_one()` call.

So one bad frame is not merely reported, it is *fed forward* as the starting
point for the next one, and the tracker walks away. The gate is a speedometer,
not a brake.

### Candidate fix (proposed, not agreed)

Per-frame motion rejection before the pose is accepted:

| Limit | Value | Rationale |
|---|---|---|
| Max translation step | 60 mm/frame | At 30 fps that is 1.8 m/s — faster than a hand can move the bottle while keeping it in frame |
| Max rotation step | 25°/frame | Comfortably above the p50 of 5.6° and the p90 of 17.5°, well below the 136.9° failures |

On rejection: hold the last known-good pose, do **not** write it to the log, and
either (a) prompt for re-registration, or (b) attempt automatic re-registration
from the cached mask. **The choice between (a) and (b) is unresolved** — it was
put to the user and dismissed, so it has not been decided.

A secondary option is to score **every** frame rather than every fifth, which
costs one extra rasterisation per frame. That is the more expensive half of the
fix; the jump test is nearly free by comparison.

## 5.6 The mask guards

A bad mask is the most common cause of a bad initialisation, and it is
invisible — GrabCut returns *something* for any box. `_pick_and_guard` applies,
in order:

1. **Connected-component selection** — keep the blob at the near surface
   (depth median closest to the near-surface seed); without a depth seed, the
   one nearest the box centre; larger blob wins ties.
2. **Minimum area** — 400 px.
3. **Depth coverage** — ≥ 50% of mask pixels must have valid depth, else the
   surface is too shiny/glassy for the stereo camera.
4. **Distance** — median depth ≤ 0.60 m.
5. **CAD projected-size check** — the mask's bounding box must be at least 25%
   of `expected_bbox_px(median_depth, K)`.

Check 5 exists because of a real failure: a hand-drawn box produced a mask of
**1601 px at 392 mm** spanning ~40×40 px, while a 153 mm bottle at 392 mm
spans **236×85 px**. Ratio 0.17, correctly rejected. Without this check,
`register()` confidently returned a pose for a hand. The bottle's projected
size is computable from the CAD and the measured distance, so a mask that
cannot be the bottle is caught *before* registration rather than after.

`expected_bbox_px` uses the **two largest** CAD dimensions (153 and 55 mm),
because orientation can only ever shrink a projection — so it is an upper
bound, which is the right direction for a sanity check.
