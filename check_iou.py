"""Silhouette IoU between the CAD rendered at the estimated pose and the mask.

This is the decisive check for a model-based pose: it renders the mesh with
FoundationPose's own renderer and the same intrinsics used for estimation,
then compares coverage against the annotated mask. 0.90+ is a very good pose,
0.80-0.90 is solid, below ~0.70 means the pose or the mask is wrong.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

PROJ = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(PROJ, 'FoundationPose'))

import nvdiffrast.torch as dr  # noqa: E402
from Utils import make_mesh_tensors, nvdiffrast_render  # noqa: E402

from perfume_common import DEFAULT_DATASET, DEFAULT_MESH, load_bottle_mesh  # noqa: E402


def render_silhouette(mesh, pose, K, H, W):
  glctx = dr.RasterizeCudaContext()
  mesh_tensors = make_mesh_tensors(mesh)
  color, depth, _ = nvdiffrast_render(
    K=K, H=H, W=W,
    ob_in_cams=torch.as_tensor(np.asarray(pose, np.float32).reshape(1, 4, 4),
                               device='cuda', dtype=torch.float),
    glctx=glctx, mesh_tensors=mesh_tensors, mesh=mesh)
  # Background renders as black, so any non-black pixel is object coverage.
  sil = (color[0].sum(-1) > 0).cpu().numpy()
  return sil, depth[0].cpu().numpy()


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--pose', type=str, required=True)
  p.add_argument('--frame', type=str, default='000000')
  p.add_argument('--mesh_file', type=str, default=DEFAULT_MESH)
  p.add_argument('--scene_dir', type=str, default=DEFAULT_DATASET)
  args = p.parse_args()

  pose = np.loadtxt(args.pose).reshape(4, 4)
  K = np.loadtxt(os.path.join(args.scene_dir, 'cam_K.txt')).reshape(3, 3)
  mask = cv2.imread(os.path.join(args.scene_dir, 'masks', f'{args.frame}.png'), -1) > 0
  rgb = cv2.imread(os.path.join(args.scene_dir, 'rgb', f'{args.frame}.png'))[..., ::-1]

  mesh = load_bottle_mesh(args.mesh_file)
  H, W = mask.shape
  sil, rdepth = render_silhouette(mesh, pose, K, H, W)

  if not sil.any():
    print('ERROR: renderer produced an empty silhouette - check pose/K')
    sys.exit(1)

  inter = (sil & mask).sum()
  union = (sil | mask).sum()
  iou = inter / union if union else 0.0
  print(f'frame {args.frame}: silhouette IoU = {iou:.4f}')
  print(f'  CAD   coverage: {sil.sum():6d} px')
  print(f'  mask  coverage: {mask.sum():6d} px')
  print(f'  only in CAD  : {(sil & ~mask).sum():6d} px  (CAD bigger / offset)')
  print(f'  only in mask : {(mask & ~sil).sum():6d} px  (CAD smaller / offset)')

  # Rendered depth vs measured depth, sampled where the two silhouettes agree.
  dm = cv2.imread(os.path.join(args.scene_dir, 'depth', f'{args.frame}.png'),
                 cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
  both = sil & mask & (rdepth > 0.001) & (dm > 0.001)
  if both.any():
    err = np.abs(rdepth[both] - dm[both])
    print(f'  depth err on {both.sum()} shared px: mean {err.mean()*1000:.1f} mm, '
          f'median {np.median(err)*1000:.1f} mm')

  vis = rgb.copy()
  vis[sil & ~mask] = (0.4 * vis[sil & ~mask] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
  vis[mask & ~sil] = (0.4 * vis[mask & ~sil] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
  vis[sil & mask] = (0.4 * vis[sil & mask] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
  out = f'/tmp/opencode/iou_{args.frame}.png'
  cv2.imwrite(out, vis[:, :, ::-1])
  print(f'  green=overlap  red=CAD only  blue=mask only -> {out}')


if __name__ == '__main__':
  main()
