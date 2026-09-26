"""Tracking quality without ground truth: depth consistency per frame.

For each frame, render the CAD at the estimated pose and compare the rendered
depth against the measured depth map over the rendered silhouette. A correct
pose puts the CAD surface on the real surface, so the inlier fraction (pixels
agreeing within --tol) stays high. A drifted pose makes the CAD float in air
or sink into the table, and the fraction collapses.

This needs no mask or ground-truth pose, so it works on the live stream too.
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

PROJ = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(PROJ, 'FoundationPose'))

from Utils import make_mesh_tensors, nvdiffrast_render  # noqa: E402
import nvdiffrast.torch as dr  # noqa: E402

from perfume_common import DEFAULT_DATASET, DEFAULT_MESH, load_bottle_mesh  # noqa: E402


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--poses', type=str, default=os.path.join(PROJ, 'debug_perfume', 'poses.json'))
  p.add_argument('--scene_dir', type=str, default=DEFAULT_DATASET)
  p.add_argument('--mesh_file', type=str, default=DEFAULT_MESH)
  p.add_argument('--tol', type=float, default=0.010, help='depth tolerance, metres')
  args = p.parse_args()

  with open(args.poses) as f:
    poses = json.load(f)
  K = np.loadtxt(os.path.join(args.scene_dir, 'cam_K.txt')).reshape(3, 3)
  mesh = load_bottle_mesh(args.mesh_file)
  glctx = dr.RasterizeCudaContext()
  mt = make_mesh_tensors(mesh)
  faces = torch.as_tensor(mesh.faces.astype(np.int32), device='cuda')

  frames = sorted(poses.keys())
  print(f'{"frame":>8} {"sil px":>7} {"inlier %":>9} {"med |dz|":>9}   verdict')
  scores = []
  for fid in frames:
    rgb = cv2.imread(os.path.join(args.scene_dir, 'rgb', f'{fid}.png'))
    dm = cv2.imread(os.path.join(args.scene_dir, 'depth', f'{fid}.png'),
                    cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    H, W = dm.shape
    pose = np.asarray(poses[fid], np.float32).reshape(1, 4, 4)
    color, depth, _ = nvdiffrast_render(
      K=K, H=H, W=W, ob_in_cams=torch.as_tensor(pose, device='cuda', dtype=torch.float),
      glctx=glctx, mesh_tensors=mt, mesh=mesh)
    sil = (color[0].sum(-1) > 0).cpu().numpy()
    rd = depth[0].cpu().numpy()
    if not sil.any():
      print(f'{fid:>8} {0:>7} {"-":>9} {"-":>9}   EMPTY (pose off-screen)')
      scores.append(0.0)
      continue
    both = sil & (rd > 0.001) & (dm > 0.001)
    if both.sum() < 50:
      print(f'{fid:>8} {int(sil.sum()):>7} {"-":>9} {"-":>9}   no depth overlap')
      scores.append(0.0)
      continue
    err = np.abs(rd[both] - dm[both])
    inlier = float((err < args.tol).mean())
    scores.append(inlier)
    verdict = 'good' if inlier > 0.8 else ('fair' if inlier > 0.6 else 'POOR')
    print(f'{fid:>8} {int(sil.sum()):>7} {inlier*100:>8.1f}% {np.median(err)*1000:>8.1f}mm   {verdict}')

  a = np.array(scores)
  print()
  print(f'frames: {len(a)}   mean inlier {a.mean()*100:.1f}%   '
        f'min {a.min()*100:.1f}%   frames below 60%: {(a < 0.6).sum()}')


if __name__ == '__main__':
  main()
