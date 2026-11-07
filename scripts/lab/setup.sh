#!/usr/bin/env bash
# One-time setup on a Linux lab machine with an NVIDIA GPU.
#
#   bash scripts/lab/setup.sh             # torch from PyPI (its Linux wheels include CUDA)
#   bash scripts/lab/setup.sh cu126       # or a specific CUDA build from download.pytorch.org
#
# Creates .venv/ in the repository, installs a CUDA build of torch and the package, then
# runs scripts/lab/check_gpu.py. Pick the CUDA tag from `nvidia-smi` and the selector at
# https://pytorch.org/get-started/locally/ if the default does not see the GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."

CUDA_TAG="${1:-}"
PYTHON="${PYTHON:-python3}"

"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "Python >= 3.10 is required"'
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
if [ -n "$CUDA_TAG" ]; then
  python -m pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
else
  python -m pip install torch
fi
python -m pip install -e ".[rl,dev]"

echo
python scripts/lab/check_gpu.py
echo
echo "Setup done. Activate the environment in new shells with: source .venv/bin/activate"
