"""Sanity-check an estimated pose by reprojecting the CAD into the frame.

Independent of FoundationPose's internals: transform the base-centered CAD
by the estimated 4x4 pose, project its corners with the scene intrinsics, and
compare against the annotated mask. Also reports where the bottle's own axes
end up in camera coordinates, which is the quickest way to catch a flipped or
transposed rotation.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import trimesh

from perfume_common import DEFAULT_DATASET, DEFAULT_MESH, load_bottle_mesh


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--pose', type=str, required=True, help='4x4 .txt')
  parser.add_argument('--frame', type=str, default='000000')
  parser.add_argument('--mesh_file', type=str, default=DEFAULT_MESH)
  parser.add_argument('--scene_dir', type=str, default=DEFAULT_DATASET)
  args = parser.parse_args()

  pose = np.loadtxt(args.pose).reshape(4, 4)
  K = np.loadtxt(os.path.join(args.scene_dir, 'cam_K.txt')).reshape(3, 3)
  rgb = cv2.imread(os.path.join(args.scene_dir, 'rgb', f'{args.frame}.png'))[..., ::-1]
  mask = cv2.imread(os.path.join(args.scene_dir, 'masks', f'{args.frame}.png'), -1) > 0

  mesh = load_bottle_mesh(args.mesh_file)
  v = (pose[:3, :3] @ mesh.vertices.T).T + pose[:3, 3]

  print('pose translation (m) :', np.round(pose[:3, 3], 4))
  print('pose rotation matrix:')
  print(np.round(pose[:3, :3], 3))

  # Object axes expressed in camera coordinates.
  for name, ax in [('+X', [1, 0, 0]), ('+Y', [0, 1, 0]), ('+Z', [0, 0, 1])]:
    d = pose[:3, :3] @ np.array(ax, float)
    print(f'  object {name} -> cam {np.round(d, 3)}')

  # Project every vertex and compare the silhouette bounds with the mask.
  z = np.clip(v[:, 2], 1e-6, None)
  u = K[0, 0] * v[:, 0] / z + K[0, 2]
  vv = K[1, 1] * v[:, 1] / z + K[1, 2]
  inside = (v[:, 2] > 0)
  pu, pv = u[inside], vv[inside]

  ys, xs = np.nonzero(mask)
  print()
  print('           %-22s %-22s' % ('projected CAD', 'annotated mask'))
  print('  x range  %-22s %-22s' % (f'{pu.min():.0f} .. {pu.max():.0f}',
                                    f'{xs.min()} .. {xs.max()}'))
  print('  y range  %-22s %-22s' % (f'{pv.min():.0f} .. {pv.max():.0f}',
                                    f'{ys.min()} .. {ys.max()}'))
  print('  height   %-22s %-22s' % (f'{pv.max()-pv.min():.0f} px',
                                    f'{ys.max()-ys.min()} px'))
  print('  width    %-22s %-22s' % (f'{pu.max()-pu.min():.0f} px',
                                    f'{xs.max()-xs.min()} px'))

  # Depth agreement: CAD surface depth vs depth map inside the mask.
  dpath = os.path.join(args.scene_dir, 'depth', f'{args.frame}.png')
  depth = cv2.imread(dpath, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
  H, W = depth.shape
  ui, vi = np.round(pu).astype(int), np.round(pv).astype(int)
  ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
  ui, vi = ui[ok], vi[ok]
  obs = depth[vi, ui]
  valid = obs > 0.001
  if valid.any():
    err = np.abs(v[inside][ok][valid, 2] - obs[valid])
    print()
    print(f'  depth samples on CAD outline: {valid.sum()}')
    print(f'  |CAD z - measured z|: mean {err.mean()*1000:.1f} mm, '
          f'median {np.median(err)*1000:.1f} mm, max {err.max()*1000:.1f} mm')

  # Draw the projected silhouette outline over the RGB frame.
  vis = rgb.copy()
  order = np.argsort(vv)
  for a, b in zip(order[:-1], order[1:]):
    if abs(vv[a] - vv[b]) < 8:
      cv2.line(vis, (int(pu[a]), int(pv[a])), (int(pu[b]), int(pv[b])),
               (0, 255, 0), 1, cv2.LINE_AA)
  out = os.path.join('/tmp/opencode', f'verify_{args.frame}.png')
  cv2.imwrite(out, vis[:, :, ::-1])
  print(f'\n  overlay written to {out}')


if __name__ == '__main__':
  main()
