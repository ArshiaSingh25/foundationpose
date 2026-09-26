"""Automatic bottle detection and initialisation, so a box is drawn only once.

Two mechanisms, because they have very different reliability:

* **Cached registration** (`try_cached_init`) - remembers the pose and mask
  from the last accepted registration and re-validates it against the current
  frame. This is the reliable path and it removes the need to redraw the box on
  every run, as long as the camera and bottle have not been moved.

* **Depth-proposal detection** (`auto_register`) - scores regions of the depth
  map against the CAD's three extents and registers the best. Useful only when
  the object is actually separated from its background in depth. On the recorded
  sequence the entire scene spans 0.18-0.26 m, so the bottle has no depth
  separation from what is behind it and this finds nothing; it is kept for
  setups where the object does stand out.

Both are only ever a *proposal*: the resulting pose is scored with the same
rendered-depth fit gate the manual path uses, so a detection that latched onto
the wrong thing is rejected rather than tracked.
"""

import os

import cv2
import numpy as np

# Metric extents of the CAD, in metres, largest last.
# Recomputed from the mesh at import time; this is only a fallback.
_DEFAULT_EXTENTS = (0.030, 0.055, 0.153)

STATE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          'init_state.npz')


def component_extents(depth, labels, idx, K):
  """Metric width/height of a labelled depth component, at its own distance."""
  sel = labels == idx
  d = float(np.median(depth[sel]))
  if d <= 0:
    return None
  ys, xs = np.where(sel)
  fx, fy = K[0, 0], K[1, 1]
  return (float(xs.max() - xs.min() + 1) * d / fx,
          float(ys.max() - ys.min() + 1) * d / fy,
          d, int(sel.sum()))


def score_candidate(w_m, h_m, extents=_DEFAULT_EXTENTS):
  """How well a metric silhouette extent pair matches a box of `extents`.

  0 is a perfect match; larger is worse. Only the two largest dimensions of a
  box can ever bound its projection, so all three pairs are accepted.
  """
  a, b, c = sorted(extents)
  pairs = [(a, b), (a, c), (b, c)]
  obs = sorted((w_m, h_m))
  best = None
  for p, q in pairs:
    e = (abs(obs[0] - p) / p + abs(obs[1] - q) / q) / 2.0
    # orientation is free, so also score the transposed pairing
    e = min(e, (abs(obs[0] - q) / q + abs(obs[1] - p) / p) / 2.0)
    best = e if best is None else min(best, e)
  return best


def save_state(path, pose, mask, K, note=''):
  """Remember an accepted registration so the next run need not be told."""
  np.savez_compressed(path, pose=np.asarray(pose, np.float64).reshape(4, 4),
                      mask=np.asarray(mask, np.uint8),
                      K=np.asarray(K, np.float64).reshape(3, 3),
                      note=str(note))


def load_state(path):
  if not os.path.exists(path):
    return None
  try:
    d = np.load(path, allow_pickle=False)
  except Exception:                                     # noqa: BLE001
    return None
  if not {'pose', 'mask', 'K'} <= set(d.files):
    return None
  return dict(pose=d['pose'], mask=d['mask'].astype(bool), K=d['K'])


def try_cached_init(est, K, rgb, depth, mesh, path=STATE_PATH, min_fit=0.30,
                    mask_fn=None, verbose=True):
  """Re-validate a remembered registration against the current frame.

  Tries the saved pose as-is first, which is exact when nothing has moved. If
  it no longer fits, and the scene still looks similar, the saved mask is used
  to re-register so a small camera or object move is tolerated. Returns
  (pose, mask, how); pose is None when the cache does not apply.
  """
  from perfume_common import score_pose
  st = load_state(path)
  if st is None:
    if verbose:
      print('auto-init: no saved registration, draw a box once with b')
    return None, None, 'no cache'
  if st['mask'].shape != depth.shape:
    if verbose:
      print('auto-init: saved mask is a different resolution, ignoring cache')
    return None, None, 'resolution mismatch'

  _, fit, _ = score_pose(est, st['pose'], K, depth, mesh=mesh)
  if verbose:
    print(f'auto-init: cached pose fits {fit*100:.0f}% '
          f'({"reusing" if fit >= min_fit else "too stale, re-registering"})')
  if fit >= min_fit:
    return st['pose'], st['mask'], 'cached pose'

  if mask_fn is None:
    return None, None, 'stale, no mask_fn'
  try:
    m, msg = mask_fn(rgb, depth, None, init_mask=st['mask'], K=K)
  except TypeError:                       # a mask_fn that takes no K
    m, msg = mask_fn(rgb, depth, None, init_mask=st['mask'])
  if m is None:
    if verbose:
      print(f'auto-init: cached mask no longer usable ({msg})')
    return None, None, 'stale mask'
  try:
    pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=m, iteration=5)
  except Exception as e:                                # noqa: BLE001
    if verbose:
      print(f'auto-init: re-registration raised {type(e).__name__}: {e}')
    return None, None, 'register failed'
  _, fit, iou = score_pose(est, pose, K, depth, mask=m, mesh=mesh)
  if verbose:
    print(f'auto-init: re-registered from cached mask -> fit {fit*100:.0f}%, '
          f'IoU {iou:.2f} {"ACCEPT" if fit >= min_fit else "reject"}')
  if fit >= min_fit:
    return pose, m, 're-registered'
  return None, None, 're-registration did not fit'


def find_candidates(depth, K, extents=_DEFAULT_EXTENTS, near=0.12, far=1.20,
                    min_px=250, max_rel_err=0.75):
  """Propose regions in `depth` that could be the bottle.

  Returns (candidates, labels) sorted best-first. `candidates` entries are
  dicts with the component index, metric size, distance and score.
  """
  band = ((depth > near) & (depth < far)).astype(np.uint8)
  if band.sum() < min_px:
    return [], None
  # close the holes that stereo dropout leaves inside the object
  k = np.ones((5, 5), np.uint8)
  band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, k, iterations=2)
  n, labels, stats, _ = cv2.connectedComponentsWithStats(band, 8)
  out = []
  for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] < min_px:
      continue
    got = component_extents(depth, labels, i, K)
    if got is None:
      continue
    w_m, h_m, d, area = got
    if max(w_m, h_m) > 3 * max(extents) or min(w_m, h_m) < 0.4 * min(extents):
      continue
    err = score_candidate(w_m, h_m, extents)
    if err > max_rel_err:
      continue
    x, y, ww, hh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    out.append(dict(idx=i, score=err, w_m=w_m, h_m=h_m, depth=d, area=area,
                    bbox=(int(x), int(y), int(ww), int(hh))))
  out.sort(key=lambda c: c['score'])
  return out, labels


def auto_register(est, K, rgb, depth, mask_fn, mesh, extents=_DEFAULT_EXTENTS,
                  min_fit=0.30, verbose=True):
  """Detect the bottle and register it without a drawn box.

  `mask_fn(rgb, depth, rect)` is the same GrabCut refinement the manual path
  uses, so a proposal is cleaned up the same way. Returns (pose, mask, info);
  pose is None when nothing convincing was found.
  """
  cands, labels = find_candidates(depth, K, extents)
  if not cands:
    if verbose:
      print('auto-init: no region matches the bottle size at 0.12-1.20 m')
    return None, None, {'reason': 'no candidate', 'candidates': []}

  tried = []
  for c in cands[:4]:
    x, y, w, h = c['bbox']
    pad = int(0.15 * max(w, h))
    rect = (max(0, x - pad), max(0, y - pad),
            min(depth.shape[1] - 1, w + 2 * pad),
            min(depth.shape[0] - 1, h + 2 * pad))
    mask, msg = mask_fn(rgb, depth, rect)
    tried.append(f'{c["w_m"]*1000:.0f}x{c["h_m"]*1000:.0f}mm '
                 f'@{c["depth"]:.2f}m score {c["score"]:.2f} -> {msg}')
    if mask is None:
      continue
    try:
      pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=5)
    except Exception as e:                      # noqa: BLE001
      if verbose:
        print(f'auto-init: registration raised {type(e).__name__}: {e}')
      continue
    from perfume_common import score_pose
    _, fit, iou = score_pose(est, pose, K, depth, mask=mask, mesh=mesh)
    if verbose:
      print(f'auto-init: candidate {c["w_m"]*1000:.0f}x{c["h_m"]*1000:.0f} mm '
            f'@ {c["depth"]:.2f} m -> fit {fit*100:.0f}%, IoU {iou:.2f}'
            f'  {"ACCEPT" if fit >= min_fit else "reject"}')
    if fit >= min_fit:
      info = dict(reason='accepted', candidate=c, fit=fit, iou=iou, tried=tried)
      return pose, mask, info
  if verbose:
    print('auto-init: no candidate produced a trustworthy pose')
  return None, None, {'reason': 'all rejected', 'candidates': cands, 'tried': tried}
