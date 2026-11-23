#!/bin/bash
# Keep NVIDIA's CUDA library dependencies separate from the training environment.
set -euo pipefail
SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIRECTORY/.."
SETUP_PYTHON="${SETUP_PYTHON:-python3.11}"
"$SETUP_PYTHON" -m venv .venv-gpu-cleanup
.venv-gpu-cleanup/bin/python -m pip install --disable-pip-version-check \
    --only-binary=:all: -r requirements-gpu-cleanup.txt
.venv-gpu-cleanup/bin/python -m pip check
echo "GPU cleanup environment installed. Use scripts/compare_gpu_cleanup.sbatch in a GPU allocation."
