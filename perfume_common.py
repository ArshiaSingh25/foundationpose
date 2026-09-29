# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to any modifications thereto.  Any use,
# reproduction, disclosure or distribution of this software and any
# modifications thereto is strictly prohibited.

"""Shared setup for the perfume bottle 6D pose scripts.

Keeps the FoundationPose import path, the CAD choice and the estimator
construction in one place so the offline and live scripts stay in sync.
"""

import os
import sys

import numpy as np
import trimesh

PROJ_DIR = os.path.dirname(os.path.realpath(__file__))
FP_DIR = os.path.join(PROJ_DIR, 'FoundationPose')

if FP_DIR not in sys.path:
  sys.path.insert(0, FP_DIR)

from estimater import *  # noqa: E402  (needs FP_DIR on sys.path first)

# Base-centered, metric CAD written by prepare_mesh.py
DEFAULT_MESH = os.path.join(PROJ_DIR, 'model', 'perfume_bottle_base.obj')
DEFAULT_DATASET = os.path.join(PROJ_DIR, 'data', 'perfume')
DEFAULT_MASK = os.path.join(DEFAULT_DATASET, 'masks', '000000.png')


def load_bottle_mesh(mesh_file=DEFAULT_MESH):
  """Load the bottle CAD as a single mesh with vertex normals.

  `process=False` keeps the original vertex indexing intact; merge_vertices
  then welds the split vertices that OBJ export leaves behind while keeping
  every face.

  The CAD ships an .mtl that declares a flat diffuse colour but no texture
  image. trimesh still reports that as TextureVisuals, which makes
  `make_mesh_tensors` dereference a None image. Convert to ColorVisuals so
  the mesh renders with the material's flat grey instead.
  """
  mesh = trimesh.load(mesh_file, force='mesh', process=False)
  mesh.merge_vertices()
  kd = np.array([154, 154, 154], dtype=np.uint8)  # .mtl Kd 0.604
  mesh.visual = trimesh.visual.ColorVisuals(
    mesh=mesh, vertex_colors=np.tile(kd, (len(mesh.vertices), 1)))
  return mesh


def build_estimator(mesh, debug=0, debug_dir=None):
  """Instantiate FoundationPose with the pre-trained refiner + scorer."""
  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  debug_dir = debug_dir or os.path.join(PROJ_DIR, 'debug')
  return FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    debug_dir=debug_dir,
    debug=debug,
    glctx=glctx,
  )


def score_pose(est, pose, K, depth, mask=None, mesh=None):
  """Check a pose against the measured depth (and optionally a mask).

  Returns (silhouette, depth_inlier_fraction, mask_iou). The inlier fraction
  is the fraction of the rendered CAD whose depth agrees with the sensor within
  10 mm; it needs no ground truth, so it works on live frames and catches a
  tracker that has drifted off the object. mask_iou compares the rendered
  silhouette with the mask the user drew, which is how a bad initial
  registration is spotted; it is None when no mask is supplied, so "no mask"
  is never mistaken for "no overlap".
  """
  H, W = depth.shape
  if mesh is None:
    mesh = est.mesh_ori
  color, dep, _ = nvdiffrast_render(
    K=K, H=H, W=W,
    ob_in_cams=torch.as_tensor(np.asarray(pose, np.float32).reshape(1, 4, 4),
                               device='cuda'),
    glctx=est.glctx, mesh_tensors=est.mesh_tensors, mesh=mesh)
  sil = (color[0].sum(-1) > 0).cpu().numpy()
  rd = dep[0].cpu().numpy()

  inlier = 0.0
  both = sil & (rd > 1e-3) & (depth > 1e-3)
  if both.sum() > 50:
    inlier = float((np.abs(rd[both] - depth[both]) < 0.010).mean())

  iou = None
  if mask is not None:
    union = (sil | mask).sum()
    iou = float((sil & mask).sum()) / float(union) if union else 0.0
  return sil, inlier, iou


def pose_to_dict(pose):
  """Split a 4x4 pose into a plain dict of human-readable values."""
  pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
  t = pose[:3, 3]
  r = pose[:3, :3]
  # Report rotation as roll/pitch/yaw (deg) and as the rotation vector, both
  # derived from the same matrix so they cannot disagree.
  sy = np.sqrt(r[0, 0]**2 + r[1, 0]**2)
  singular = sy < 1e-6
  if not singular:
    rpy = np.array([np.arctan2(r[2, 1], r[2, 2]),
                    np.arctan2(-r[2, 0], sy),
                    np.arctan2(r[1, 0], r[0, 0])])
  else:
    rpy = np.array([np.arctan2(-r[1, 2], r[1, 1]),
                    np.arctan2(-r[2, 0], sy),
                    0.0])
  rotvec = cv2.Rodrigues(r)[0].ravel()
  # The bottle body is a rectangular prism, so its pose is only defined up to
  # a 180 deg spin about its long (+Y) axis. Report that axis in camera frame
  # too: unlike rpy it is unambiguous.
  return {
    'translation_m': [float(x) for x in t],
    'translation_mm': [float(x * 1000) for x in t],
    'rpy_deg': [float(np.degrees(a)) for a in rpy],
    'rotation_vector_deg': [float(np.degrees(a)) for a in rotvec],
    'long_axis_cam': [float(x) for x in (r @ np.array([0.0, 1.0, 0.0]))],
    'matrix': pose.tolist(),
  }


def format_pose(pose, label=''):
  d = pose_to_dict(pose)
  prefix = f'{label} ' if label else ''
  a = d['long_axis_cam']
  return (f"{prefix}t = [{d['translation_mm'][0]:8.1f}, {d['translation_mm'][1]:8.1f}, "
          f"{d['translation_mm'][2]:8.1f}] mm  (base center)\n"
          f"{' ' * len(prefix)}rpy = [{d['rpy_deg'][0]:7.1f}, {d['rpy_deg'][1]:7.1f}, "
          f"{d['rpy_deg'][2]:7.1f}] deg   (mod 180 about long axis)\n"
          f"{' ' * len(prefix)}axis+ = [{a[0]:5.2f}, {a[1]:5.2f}, {a[2]:5.2f}] in camera frame")
