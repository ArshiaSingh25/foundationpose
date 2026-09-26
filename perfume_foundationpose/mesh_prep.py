"""Prepare CAD OBJ for FoundationPose (vertex colors, single watertight mesh)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def load_and_prepare_mesh(obj_path: Path) -> trimesh.Trimesh:
  loaded = trimesh.load(str(obj_path), process=True, force="mesh")
  if isinstance(loaded, trimesh.Scene):
    mesh = loaded.dump(concatenate=True)
  else:
    mesh = loaded

  gray = np.tile(np.array([160, 160, 160], dtype=np.uint8), (len(mesh.vertices), 1))
  if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
    image = None
    if mesh.visual.material is not None:
      image = getattr(mesh.visual.material, "image", None)
    if image is None:
      mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=gray)
  elif mesh.visual.vertex_colors is None:
    mesh.visual.vertex_colors = gray

  mesh.merge_vertices()
  mesh.remove_unreferenced_vertices()
  return mesh


def ensure_prepared_mesh(source: Path, dest: Path) -> Path:
  """Write prepared mesh to dest if missing or older than source."""
  dest.parent.mkdir(parents=True, exist_ok=True)
  if dest.is_file() and dest.stat().st_mtime >= source.stat().st_mtime:
    return dest
  mesh = load_and_prepare_mesh(source)
  mesh.export(str(dest))
  return dest
