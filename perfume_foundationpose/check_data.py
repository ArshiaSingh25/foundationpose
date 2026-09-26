#!/usr/bin/env python3
"""Validate data/perfume layout before running FoundationPose."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from paths import MESH_SOURCE, SCENE_DIR


def main() -> int:
  errors: list[str] = []
  scene = SCENE_DIR
  rgb = sorted((scene / "rgb").glob("*.png"))
  if not rgb:
    errors.append(f"No RGB frames in {scene / 'rgb'}")
  if not (scene / "cam_K.txt").is_file():
    errors.append(f"Missing {scene / 'cam_K.txt'}")
  if not (scene / "masks" / "000000.png").is_file():
    errors.append(f"Missing first-frame mask {scene / 'masks' / '000000.png'}")
  if not MESH_SOURCE.is_file():
    errors.append(f"Missing CAD mesh {MESH_SOURCE}")

  if rgb:
    i0 = rgb[0].stem
    depth_path = scene / "depth" / f"{i0}.png"
    if not depth_path.is_file():
      errors.append(f"Missing depth for frame 0: {depth_path}")
    else:
      d = cv2.imread(str(depth_path), -1)
      if d is None:
        errors.append(f"Could not read depth {depth_path}")
      else:
        valid = d[d > 0]
        if valid.size == 0:
          errors.append("Frame-0 depth is empty")
        else:
          print(f"Frame 0 depth: {d.shape}, range mm [{valid.min()}, {valid.max()}]")

    im = cv2.imread(str(rgb[0]))
    if im is not None:
      print(f"Frame 0 RGB: {im.shape[1]}x{im.shape[0]}, {len(rgb)} frames total")

    K = np.loadtxt(scene / "cam_K.txt").reshape(3, 3)
    print(f"Intrinsics fx,fy,cx,cy: {K[0,0]:.2f}, {K[1,1]:.2f}, {K[0,2]:.2f}, {K[1,2]:.2f}")

  for e in errors:
    print(f"ERROR: {e}")
  return 1 if errors else 0


if __name__ == "__main__":
  raise SystemExit(main())
