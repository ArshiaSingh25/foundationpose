#!/usr/bin/env python3
"""
Run NVIDIA FoundationPose on the perfume bottle CAD mesh and captured RGB-D sequence.

Does not modify FoundationPose or existing project files; writes outputs under output/.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import trimesh

from mesh_prep import ensure_prepared_mesh, load_and_prepare_mesh
from paths import DEBUG_DIR, FOUNDATIONPOSE_ROOT, MESH_FILE, MESH_SOURCE, SCENE_DIR


def _bootstrap_foundationpose() -> Path:
  fp_root = FOUNDATIONPOSE_ROOT.resolve()
  if not fp_root.is_dir():
    raise FileNotFoundError(f"FoundationPose not found at {fp_root}")
  os.chdir(fp_root)
  if str(fp_root) not in sys.path:
    sys.path.insert(0, str(fp_root))
  build_dir = fp_root / "mycpp" / "build"
  if build_dir.is_dir() and str(build_dir) not in sys.path:
    sys.path.insert(0, str(build_dir))
  return fp_root


def _rotation_symmetry_y(steps: int = 12) -> np.ndarray:
  """Discrete rotations about +Y (typical upright bottle axis in CAD bounds)."""
  tfs = []
  for i in range(steps):
    angle = 2.0 * np.pi * i / steps
    c, s = np.cos(angle), np.sin(angle)
    tf = np.eye(4, dtype=np.float64)
    tf[0, 0] = c
    tf[0, 2] = s
    tf[2, 0] = -s
    tf[2, 2] = c
    tfs.append(tf)
  return np.stack(tfs, axis=0)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mesh_file", type=Path, default=MESH_FILE)
  parser.add_argument("--scene_dir", type=Path, default=SCENE_DIR)
  parser.add_argument("--debug_dir", type=Path, default=DEBUG_DIR)
  parser.add_argument("--est_refine_iter", type=int, default=5)
  parser.add_argument("--track_refine_iter", type=int, default=2)
  parser.add_argument("--debug", type=int, default=2, help="0=off, 1=live window, 2=+save vis")
  parser.add_argument("--max_frames", type=int, default=-1, help="-1 = all frames")
  parser.add_argument(
    "--symmetry_y",
    action="store_true",
    help="Use discrete Y-axis symmetry (helps symmetric bottles).",
  )
  parser.add_argument("--symmetry_steps", type=int, default=12)
  args = parser.parse_args()

  mesh_file = args.mesh_file.resolve()
  scene_dir = args.scene_dir.resolve()
  debug_dir = args.debug_dir.resolve()

  if mesh_file == MESH_FILE.resolve() and MESH_SOURCE.is_file():
    ensure_prepared_mesh(MESH_SOURCE, MESH_FILE)
  elif not mesh_file.is_file():
    logging.error("Mesh missing: %s", mesh_file)
    return 1
  if not (scene_dir / "cam_K.txt").is_file():
    logging.error("Scene missing cam_K.txt under %s", scene_dir)
    return 1
  if not list((scene_dir / "rgb").glob("*.png")):
    logging.error("No RGB frames in %s/rgb", scene_dir)
    return 1

  _bootstrap_foundationpose()
  from datareader import YcbineoatReader  # noqa: E402
  from estimater import (  # noqa: E402
    FoundationPose,
    PoseRefinePredictor,
    ScorePredictor,
    draw_posed_3d_box,
    draw_xyz_axis,
    set_logging_format,
    set_seed,
  )
  import nvdiffrast.torch as dr  # noqa: E402

  set_logging_format()
  set_seed(0)

  mesh = load_and_prepare_mesh(mesh_file)
  logging.info("Mesh %s: %d verts, extents %s (m)", mesh_file.name, len(mesh.vertices), mesh.extents)

  debug_dir.mkdir(parents=True, exist_ok=True)
  for sub in ("track_vis", "ob_in_cam"):
    (debug_dir / sub).mkdir(parents=True, exist_ok=True)

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

  symmetry_tfs = _rotation_symmetry_y(args.symmetry_steps) if args.symmetry_y else None

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
    symmetry_tfs=symmetry_tfs,
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    debug_dir=str(debug_dir),
    debug=args.debug,
    glctx=glctx,
  )
  logging.info("Estimator ready")

  reader = YcbineoatReader(video_dir=str(scene_dir), shorter_side=None, zfar=np.inf)
  n_frames = len(reader)
  if args.max_frames > 0:
    n_frames = min(n_frames, args.max_frames)
  logging.info("Processing %d / %d frames from %s", n_frames, len(reader), scene_dir)

  mask0_path = scene_dir / "masks" / "000000.png"
  if not mask0_path.is_file():
    logging.error(
      "FoundationPose needs an object mask on frame 0. Create %s (see annotate_first_frame_mask.py).",
      mask0_path,
    )
    return 1

  for i in range(n_frames):
    logging.info("frame %d / %d", i, n_frames - 1)
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    frame_id = reader.id_strs[i]

    if i == 0:
      mask = reader.get_mask(0).astype(bool)
      pose = est.register(
        K=reader.K,
        rgb=color,
        depth=depth,
        ob_mask=mask,
        iteration=args.est_refine_iter,
      )
    else:
      pose = est.track_one(
        rgb=color,
        depth=depth,
        K=reader.K,
        iteration=args.track_refine_iter,
      )

    np.savetxt(debug_dir / "ob_in_cam" / f"{frame_id}.txt", pose.reshape(4, 4))

    if args.debug >= 1:
      center_pose = pose @ np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(
        color,
        ob_in_cam=center_pose,
        scale=0.05,
        K=reader.K,
        thickness=3,
        transparency=0,
        is_input_rgb=True,
      )
      if args.debug >= 3 and os.environ.get("DISPLAY"):
        cv2.imshow("perfume_pose", vis[..., ::-1])
        cv2.waitKey(1)
      if args.debug >= 2:
        imageio.imwrite(str(debug_dir / "track_vis" / f"{frame_id}.png"), vis)

  if args.debug >= 3 and os.environ.get("DISPLAY"):
    cv2.destroyAllWindows()

  logging.info("Done. Poses: %s/ob_in_cam  Visualizations: %s/track_vis", debug_dir, debug_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
