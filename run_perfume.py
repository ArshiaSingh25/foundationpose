# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to any modifications thereto.  Any use,
# reproduction, disclosure or distribution of this software and any
# modifications thereto is strictly prohibited.

"""Run FoundationPose on the recorded perfume bottle sequence.

Estimates the 6D pose of the bottle on frame 0 (which carries the mask) and
then tracks it through the rest of the sequence, mirroring run_demo.py.

Poses are written to <debug_dir>/ob_in_cam/<frame>.txt as 4x4 matrices, and
printed as translation (base center, mm) + roll/pitch/yaw (deg).
"""

import argparse
import json
import logging
import os

import cv2
import imageio
import numpy as np
import trimesh

from perfume_common import (DEFAULT_DATASET, DEFAULT_MESH, FP_DIR, PROJ_DIR,
                            build_estimator, draw_posed_3d_box, draw_xyz_axis,
                            format_pose, load_bottle_mesh, set_logging_format,
                            set_seed)
import sys
sys.path.insert(0, FP_DIR)
from datareader import PerfumerReader  # noqa: E402


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--mesh_file', type=str, default=DEFAULT_MESH)
  parser.add_argument('--test_scene_dir', type=str, default=DEFAULT_DATASET)
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=os.path.join(PROJ_DIR, 'debug_perfume'))
  parser.add_argument('--max_frames', type=int, default=0,
                      help='0 = all frames')
  parser.add_argument('--no_show', action='store_true',
                      help='do not open an OpenCV window')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  mesh = load_bottle_mesh(args.mesh_file)
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
  logging.info(f'mesh: {len(mesh.vertices)} verts, extents {np.round(mesh.extents*1000,1)} mm')

  debug_dir = args.debug_dir
  os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
  os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)

  est = build_estimator(mesh, debug=args.debug, debug_dir=debug_dir)
  logging.info('estimator initialization done')

  reader = PerfumerReader(args.test_scene_dir, shorter_side=None, zfar=np.inf)
  n = len(reader) if args.max_frames <= 0 else min(args.max_frames, len(reader))
  logging.info(f'running on {n} frames from {args.test_scene_dir}')

  poses = {}
  pose = None
  for i in range(n):
    color = reader.get_color(i)
    depth = reader.get_depth(i)

    if i == 0:
      mask = reader.get_mask(0).astype(bool)
      logging.info(f'mask covers {mask.sum()} px')
      if mask.sum() == 0:
        raise RuntimeError('frame 0 has no mask; register() cannot run')
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask,
                          iteration=args.est_refine_iter)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K,
                          iteration=args.track_refine_iter)

    poses[reader.id_strs[i]] = np.asarray(pose).reshape(4, 4).tolist()
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4, 4))

    if i % 20 == 0 or i == n - 1:
      logging.info(format_pose(pose, f'frame {i:4d}'))

    if args.debug >= 1:
      center_pose = pose @ np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.05, K=reader.K,
                          thickness=3, transparency=0, is_input_rgb=True)
      for j, text in enumerate(format_pose(pose, f'{i}').split('\n')):
        cv2.putText(vis, text, (10, 24 + 18 * j),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
      if args.debug >= 2:
        imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)
      if not args.no_show:
        cv2.imshow('perfume', vis[..., ::-1])
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
          logging.info('stopped by user')
          break

  if not args.no_show:
    cv2.destroyAllWindows()

  out_json = os.path.join(debug_dir, 'poses.json')
  with open(out_json, 'w') as f:
    json.dump(poses, f, indent=1)
  logging.info(f'wrote {len(poses)} poses to {out_json} and per-frame 4x4 matrices '
               f'to {debug_dir}/ob_in_cam/')


if __name__ == '__main__':
  main()
