#!/usr/bin/env bash
# Run from a terminal with the foundationpose conda env (needed for nvdiffrast / torch libs).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${CONDA_PREFIX:-}/etc/conda/activate.d/env_vars.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh"
fi
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"
exec python run_perfume_pose.py "$@"
