"""Default paths for perfume bottle FoundationPose (repo-relative)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FOUNDATIONPOSE_ROOT = REPO_ROOT / "FoundationPose"
MESH_SOURCE = REPO_ROOT / "model" / "perfume_bottle.obj"
MESH_FILE = REPO_ROOT / "output" / "perfume_bottle_fp.obj"
SCENE_DIR = REPO_ROOT / "data" / "perfume"
DEBUG_DIR = REPO_ROOT / "output" / "foundationpose_perfume"
