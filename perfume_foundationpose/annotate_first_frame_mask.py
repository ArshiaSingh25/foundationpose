#!/usr/bin/env python3
"""Interactive brush tool to create masks/000000.png for FoundationPose registration."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from paths import SCENE_DIR


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--scene_dir", type=Path, default=SCENE_DIR)
  parser.add_argument("--frame", type=str, default="000000")
  parser.add_argument("--brush", type=int, default=24)
  args = parser.parse_args()

  rgb_path = args.scene_dir / "rgb" / f"{args.frame}.png"
  if not rgb_path.is_file():
    print(f"Missing {rgb_path}")
    return 1

  img = cv2.imread(str(rgb_path))
  mask = np.zeros(img.shape[:2], dtype=np.uint8)
  overlay = img.copy()
  drawing = False

  def refresh() -> None:
    nonlocal overlay
    tint = np.zeros_like(img)
    tint[:, :, 1] = mask
    overlay = cv2.addWeighted(img, 0.7, tint, 0.3, 0)

  def on_mouse(event, x, y, _flags, _param) -> None:
    nonlocal drawing
    if event == cv2.EVENT_LBUTTONDOWN:
      drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
      drawing = False
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
      cv2.circle(mask, (x, y), args.brush, 255, -1)
      refresh()

  refresh()
  win = "Paint object mask (s=save, c=clear, q=quit)"
  cv2.namedWindow(win)
  cv2.setMouseCallback(win, on_mouse)

  while True:
    cv2.imshow(win, overlay)
    key = cv2.waitKey(20) & 0xFF
    if key in (ord("q"), 27):
      break
    if key == ord("c"):
      mask[:] = 0
      refresh()
    if key == ord("s"):
      out_dir = args.scene_dir / "masks"
      out_dir.mkdir(parents=True, exist_ok=True)
      out_path = out_dir / f"{args.frame}.png"
      cv2.imwrite(str(out_path), mask)
      preview = args.scene_dir / "mask_overlay.png"
      cv2.imwrite(str(preview), overlay)
      print(f"Saved {out_path} ({int((mask > 0).sum())} px)")
      break

  cv2.destroyAllWindows()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
