# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to any modifications thereto.  Any use,
# reproduction, disclosure or distribution of this software and any
# modifications thereto is strictly prohibited.

"""Live 6D pose of the perfume bottle from a RealSense D405.

Workflow:
  1. The colour stream opens in a window. Drag a rectangle around the bottle
     and press Enter; GrabCut then refines that box into a mask.
  2. FoundationPose registers the CAD on that frame (full pose search).
  3. Every later frame is tracked from the previous pose (fast).
  4. The box, the axis triad and the live pose are drawn, and each pose is
     appended to a JSONL file.

Keys: r = re-register on the current frame, s = save a snapshot, q/ESC = quit.

Depth is aligned to the colour stream, so the mask taken from the colour
image lines up with the depth pixels FoundationPose consumes. Intrinsics
are read from the colour stream to match that alignment.
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import trimesh

from perfume_common import (DEFAULT_MESH, PROJ_DIR, build_estimator,
                            draw_posed_3d_box, draw_xyz_axis, format_pose,
                            score_pose,
                            load_bottle_mesh, set_logging_format, set_seed)

WINDOW = 'perfume live'

# Pose-quality gates, calibrated against the recorded frames (see
# /tmp/opencode/test_thresholds.py): good poses fit >= 43.7% of the CAD depth
# within 10 mm, a wrong pose fits 0%, and the one ground-truth mask scores
# IoU 0.40. 30% / 0.25 sits in that gap with room on both sides.
MIN_FIT = 0.30
MIN_IOU = 0.25


def start_pipeline(width, height, fps, warmup=15):
  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
  config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

  profile = pipeline.start(config)
  device = profile.get_device()
  depth_scale = device.first_depth_sensor().get_depth_scale()

  # Align depth into the colour camera's frame so mask and depth coincide.
  align = rs.align(rs.stream.color)

  # Discard the first frames: auto-exposure and white balance are still
  # settling, and a dark first frame gives GrabCut nothing to work with.
  for _ in range(warmup):
    pipeline.wait_for_frames()
  return pipeline, align, depth_scale


def grab_frames(pipeline, align, depth_scale):
  """Return (rgb uint8 HxWx3, depth float32 metres, K 3x3).

  Waits indefinitely for a frame: there is no timeout, so a brief hiccup in
  the stream just blocks here instead of raising a spurious timeout error.
  """
  frames = pipeline.wait_for_frames()
  frames = align.process(frames)

  depth_frame = frames.get_depth_frame()
  color_frame = frames.get_color_frame()
  if not depth_frame or not color_frame:
    raise RuntimeError('frame arrived without a depth or colour frame')

  rgb = np.asanyarray(color_frame.get_data())
  depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale

  # float64 to match what FoundationPose's refiner expects (it multiplies K by
  # float64 point arrays directly).
  intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
  K = np.array([[intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0]], dtype=np.float64)
  return rgb, depth, K


def _depth_seeds(depth, x0, y0, x1, y1):
  """Depth ranges that separate the object from the background in a box.

  Returns (near, seed_tol, bg_cut) in metres, or None if depth is unusable.
  `near` is the median of the closest surface cluster, `seed_tol` how far in
  depth to trust as foreground, and anything past `bg_cut` is far enough to be
  a different surface. Returns None when the box has too little depth.
  """
  sub = depth[y0:y1, x0:x1]
  v = sub[(sub > 0.05) & (sub < 6.0)]
  if v.size < 50:
    return None

  p10 = float(np.percentile(v, 10))
  # Median of the near surface cluster, ignoring its own thickness.
  cluster = v[(v >= p10) & (v <= p10 + 0.06)]
  near = float(np.median(cluster)) if cluster.size >= 20 else p10

  # A genuinely separate background surface sits well behind the near one.
  behind = v[v > near + 0.12]
  if behind.size >= 20:
    bg_cut = 0.5 * (float(np.median(behind)) + near + 0.12)
  else:
    bg_cut = near + 0.25
  return near, 0.035, float(min(bg_cut, near + 0.30))


_EXTENTS_MM = None


def cad_extents_mm():
  """CAD bounding-box extents in mm, cached (the mesh is not reloaded).

  The mesh is authored in metres, hence the x1000.
  """
  global _EXTENTS_MM
  if _EXTENTS_MM is None:
    import trimesh
    m = trimesh.load(DEFAULT_MESH, force='mesh', process=False)
    _EXTENTS_MM = sorted(float(v) * 1000.0 for v in m.extents)
  return _EXTENTS_MM


def expected_bbox_px(dist_m, K):
  """Largest plausible on-screen bbox of the bottle at `dist_m`.

  Uses the two largest CAD dimensions, since orientation only ever shrinks a
  projection. Meant as a sanity bound, not an exact expectation.
  """
  e = cad_extents_mm()
  long_mm, mid_mm = e[2], e[1]
  d_mm = max(1e-6, dist_m * 1000.0)
  return (long_mm * K[0, 0] / d_mm, mid_mm * K[1, 1] / d_mm)


def _pick_and_guard(mask, depth, seeds, box, area_ref, K=None):
  """Keep the blob at the near surface, then apply the depth sanity guards.

  Shared by the drawn-box and remembered-mask paths so both get identical
  checks. `box` is (x0, y0, x1, y1) used to break ties when there is no depth
  seed; `area_ref` is the pixel area to quote percentages against.
  """
  x0, y0, x1, y1 = box
  n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
  if n > 1:
    best, best_d = 0, 1e18
    for i in range(1, n):
      if seeds is None:
        mx = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2
        my = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2
        d = (mx - (x0 + x1) / 2.0) ** 2 + (my - (y0 + y1) / 2.0) ** 2
      else:
        dv = depth[lab == i]
        dv = dv[dv > 0]
        d = abs(float(np.median(dv)) - seeds[0]) if dv.size else 1e9
        if d < 1e8:
          d += 1e-6 * stats[i, cv2.CC_STAT_AREA]  # prefer the larger blob on ties
      if d < best_d:
        best, best_d = i, d
    mask = lab == best

  area = int(mask.sum())
  if area < 400:
    return None, f'mask too small ({area} px) - draw a tighter box'

  # FoundationPose needs depth, so check the mask really has usable depth
  # before spending time on registration.
  dv = depth[mask]
  dv = dv[dv > 0]
  cov = dv.size / float(area)
  if cov < 0.5:
    return None, (f'no depth on the object ({100 * cov:.0f}% of mask has depth) '
                  '- move it closer, or the surface is too shiny/glass for the '
                  'stereo camera')
  med = float(np.median(dv))
  if med > 0.60:
    return None, (f'object is {med:.2f} m away - too far for reliable stereo '
                  'depth, move it to 0.15-0.30 m')

  # GrabCut sometimes latches onto a small near object (a hand, the far side of
  # the cap) instead of the bottle. The bottle's projected size is known from
  # the CAD, so a wildly wrong blob is caught here rather than being handed to
  # registration, which would return a confident but useless pose.
  if K is None:
    return mask, (f'mask {area} px '
                  f'({100.0 * area / max(1, area_ref):.0f}% of box), '
                  f'object at {med * 1000:.0f} mm')
  ys, xs = np.where(mask)
  bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
  ew, eh = expected_bbox_px(med, K)
  ratio = min(bw / max(1.0, ew), bh / max(1.0, eh))
  if ratio < 0.25:
    return None, (f'that blob is only {bw}x{bh} px at {med*1000:.0f} mm, but a '
                  f'153 mm bottle at that distance spans about '
                  f'{ew:.0f}x{eh:.0f} px.\n  GrabCut caught the wrong thing - '
                  f'draw a box tightly around the whole bottle, with nothing '
                  f'else inside it')
  return mask, (f'mask {area} px '
                f'({100.0 * area / max(1, area_ref):.0f}% of box), '
                f'object at {med * 1000:.0f} mm, {bw}x{bh} px')


def mask_from_box(rgb, depth, rect, iters=5, init_mask=None, K=None):
  """Turn a user-drawn rectangle into an object mask with GrabCut.

  GrabCut is seeded with the rectangle itself (its standard rectangle init),
  which is what actually separates the bottle from the table here: the two
  differ far more in brightness than in depth, so a depth-driven trimap only
  adds failure modes. Depth is used afterwards, to pick the component that
  sits at the near surface, so a second object inside the same box cannot
  win the mask.

  With `init_mask` the rectangle is ignored and GrabCut is seeded from that
  mask instead (GC_INIT_WITH_MASK), which is how a remembered registration is
  reused without the user redrawing anything.
  """
  H, W = depth.shape
  if init_mask is not None and init_mask.shape == depth.shape:
    # A remembered mask: seed GrabCut from it, with a dilated band of unknown
    # pixels around the edge so a small move can still be absorbed.
    seed = init_mask.astype(np.uint8)
    sure_fg = cv2.erode(seed, np.ones((5, 5), np.uint8), iterations=2)
    sure_bg = (cv2.dilate(seed, np.ones((15, 15), np.uint8), iterations=3) == 0)
    gc = np.full((H, W), cv2.GC_PR_BGD, np.uint8)
    gc[seed > 0] = cv2.GC_PR_FGD
    gc[sure_fg > 0] = cv2.GC_FGD
    gc[sure_bg] = cv2.GC_BGD
    if (gc == cv2.GC_FGD).sum() < 50:
      return None, 'cached mask too small'
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
      cv2.grabCut(bgr, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error as e:
      return None, f'grabCut failed: {str(e)[:60]}'
    mask = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    ys, xs = np.where(mask)
    if xs.size == 0:
      return None, 'cached mask produced nothing'
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return _pick_and_guard(mask, depth, None, box, int(seed.sum()), K)

  x, y, w, h = rect
  x0, y0 = max(0, x), max(0, y)
  x1, y1 = min(W, x + w), min(H, y + h)
  if x1 - x0 < 8 or y1 - y0 < 8:
    return None, 'box too small'

  bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
  gc = np.zeros((H, W), np.uint8)
  bgd = np.zeros((1, 65), np.float64)
  fgd = np.zeros((1, 65), np.float64)
  try:
    cv2.grabCut(bgr, gc, (x0, y0, x1 - x0, y1 - y0), bgd, fgd, iters,
                cv2.GC_INIT_WITH_RECT)
  except cv2.error as e:
    return None, f'grabCut failed: {str(e)[:60]}'

  mask = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
  mask[:y0, :] = False
  mask[y1:, :] = False
  mask[:, :x0] = False
  mask[:, x1:] = False

  seeds = _depth_seeds(depth, x0, y0, x1, y1)
  return _pick_and_guard(mask, depth, seeds, (x0, y0, x1, y1),
                         (x1 - x0) * (y1 - y0), K)


def draw_overlay(vis, pose, bbox, K, line, extra=""):
  vis = draw_posed_3d_box(K, img=vis, ob_in_cam=pose, bbox=bbox)
  vis = draw_xyz_axis(vis, ob_in_cam=pose, scale=0.05, K=K, thickness=3,
                      transparency=0, is_input_rgb=True)
  for i, l in enumerate(line.split('\n')):
    cv2.putText(vis, l, (10, 24 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, l, (10, 24 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0), 1, cv2.LINE_AA)
  if extra:
    cv2.putText(vis, extra, (10, vis.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, extra, (10, vis.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 255), 1, cv2.LINE_AA)
  return vis


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--mesh_file', type=str, default=DEFAULT_MESH)
  p.add_argument('--width', type=int, default=640)
  p.add_argument('--height', type=int, default=480)
  p.add_argument('--fps', type=int, default=30)
  p.add_argument('--warmup', type=int, default=15,
                 help='frames to discard after start, for auto-exposure')
  p.add_argument('--est_refine_iter', type=int, default=5)
  p.add_argument('--track_refine_iter', type=int, default=2)
  p.add_argument('--out', type=str, default=os.path.join(PROJ_DIR, 'live_poses.jsonl'))
  p.add_argument('--snap_dir', type=str, default=os.path.join(PROJ_DIR, 'live_snaps'))
  args = p.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = load_bottle_mesh(args.mesh_file)
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

  est = build_estimator(mesh, debug=0, debug_dir=os.path.join(PROJ_DIR, 'debug_live'))

  pipeline, align, depth_scale = start_pipeline(args.width, args.height, args.fps,
                                               warmup=args.warmup)
  print(f'RealSense started: depth {args.width}x{args.height}@{args.fps}, '
        f'depth scale {depth_scale:.6f} m/unit')

  os.makedirs(args.snap_dir, exist_ok=True)
  pose = None
  mask = None
  frame_id = 0
  n_track = 0
  fps_ema = 0.0
  selecting = False
  bad_frames = 0
  last_inlier = 1.0
  last = None  # last rendered frame, reused as the selectROI background

  print('\nlive feed running. press b to draw a box around the bottle, '
        'then Enter.')
  print('keys: b = draw box (register), r = re-register, s = snapshot, '
        'q = quit\n')

  def draw_hint(img, lines):
    for j, text in enumerate(lines):
      cv2.putText(img, text, (10, 24 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                  (0, 0, 0), 3, cv2.LINE_AA)
      cv2.putText(img, text, (10, 24 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                  (0, 255, 255), 1, cv2.LINE_AA)
    return img

  def do_select(rgb, depth, K):
    """Block for a rectangle on the current frame, then register.

    Returns (pose, mask), or (None, None) if cancelled or the mask is bad.
    """
    rect = cv2.selectROI(WINDOW, last, showCrosshair=True, fromCenter=False)
    if rect == (0, 0, 0, 0):
      print('no box selected (Esc), still live')
      return None, None
    mask, msg = mask_from_box(rgb, depth, rect, K=K)
    if mask is None:
      print(msg)
      return None, None
    print(f'registering ({msg}) ...')
    t0 = time.time()
    pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                        iteration=args.est_refine_iter)
    print(f'registered in {time.time() - t0:.2f} s ->\n{format_pose(pose)}')

    # Do not start tracking a pose that does not actually explain the frame.
    # Without this the tracker happily flies off the object and you only
    # notice when the box is somewhere else entirely.
    #
    # Thresholds are calibrated on the 180 recorded frames scored with their
    # offline poses: known-good fits are min 43.7% / median 58.9%, while a
    # deliberately wrong pose scores 0%. 30% therefore sits in a wide empty
    # gap, and a 50% gate would have rejected 21 of the 180 good frames.
    _, inlier, iou = score_pose(est, pose, K, depth, mask=mask, mesh=mesh)
    print(f'check: mask IoU {iou:.2f}, depth agreement {inlier * 100:.0f}%')
    if inlier < MIN_FIT or (iou is not None and iou < MIN_IOU):
      print('  WARNING: that pose does not fit the frame well. It is probably '
            'wrong.\n  Press b to draw a tighter box on the bottle, making '
            'sure it fills the box.')
      return None, None
    print('  pose looks consistent with the frame.\n')
    return pose, mask

  while True:
    rgb, depth, K = grab_frames(pipeline, align, depth_scale)
    K = np.asarray(K, dtype=np.float64)

    # While the box is being drawn the loop is blocked inside selectROI, so the
    # camera queue fills up; drain it so tracking resumes on fresh frames.
    if selecting:
      for _ in range(5):
        pipeline.wait_for_frames()
      selecting = False

    frame_id += 1
    vis = rgb.copy()

    if pose is None:
      vis = draw_hint(vis, [
        f'live  |  frame {frame_id}',
        'press b to draw a box around the bottle, then Enter',
        f'valid depth: {100.0 * (depth > 0).mean():.0f}% of frame',
      ])
    else:
      t0 = time.time()
      pose = est.track_one(rgb=rgb, depth=depth, K=K,
                           iteration=args.track_refine_iter)
      dt = time.time() - t0
      n_track += 1
      fps_ema = 1.0 / dt if fps_ema == 0 else 0.9 * fps_ema + 0.1 / dt
      with open(args.out, 'a') as f:
        f.write(json.dumps({'frame': frame_id, 'track': n_track,
                            'pose': np.asarray(pose).reshape(4, 4).tolist()}) + '\n')

      # Show how well the pose still explains the frame, so a tracker that is
      # sliding off the object is obvious instead of silently drifting.
      if n_track % 5 == 1 or bad_frames:
        _, inlier, _ = score_pose(est, pose, K, depth, mesh=mesh)
        if inlier < MIN_FIT:
          bad_frames += 1
          if bad_frames == 5:
            print(f'\nWARNING: the pose no longer fits the frame (depth '
                  f'agreement below {MIN_FIT * 100:.0f}%). The tracker has '
                  f'probably lost the object.\n  Press r, then b to '
                  f're-register on the bottle.')
        else:
          bad_frames = 0
      else:
        inlier = last_inlier

      center_pose = np.asarray(pose) @ np.linalg.inv(to_origin)
      last_inlier = inlier
      flag = '  LOST' if inlier < MIN_FIT else ''
      extra = (f'track {n_track}  |  {fps_ema:.1f} Hz  |  fit {inlier*100:3.0f}%'
               f'{flag}  |  r=re-register s=snap q=quit')
      vis = draw_overlay(vis, center_pose, bbox, K, format_pose(pose), extra)
      if mask is not None and n_track == 0:
        vis[mask] = (0.75 * vis[mask] + 0.25 * np.array([0, 255, 255])).astype(np.uint8)

    last = vis[..., ::-1].copy()
    cv2.imshow(WINDOW, last)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
      break
    if key in (ord('b'), ord(' ')):
      selecting = True
      pose, mask = do_select(rgb, depth, K)
      if pose is not None:
        n_track = 0
        bad_frames = 0
        last_inlier = 1.0
        print('tracking live now. r = re-register.\n')
    if key == ord('r'):
      pose = None
      print('released. press b to draw a new box.\n')
    if key == ord('s') and pose is not None:
      fn = os.path.join(args.snap_dir, f'frame{frame_id:06d}.png')
      cv2.imwrite(fn, vis[:, :, ::-1])
      np.savetxt(fn.replace('.png', '.txt'), np.asarray(pose).reshape(4, 4))
      print(f'saved {fn}')

  pipeline.stop()
  cv2.destroyAllWindows()
  print(f'\nstopped after {frame_id} frames, {n_track} tracked. '
        f'poses appended to {args.out}')


if __name__ == '__main__':
  main()
