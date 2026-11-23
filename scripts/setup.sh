#!/bin/bash
# Install a checkout into its own Python environment and check the full path.
set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIRECTORY/.."
SETUP_PYTHON="${SETUP_PYTHON:-python3.11}"
MODE="${1:---cuda}"
case "$MODE" in
    --cpu)
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        TORCH_REQUIREMENT="torch==2.5.1+cpu"
        ;;
    --cuda)
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        TORCH_REQUIREMENT="torch==2.5.1+cu124"
        ;;
    *) echo "Usage: bash scripts/setup.sh [--cpu|--cuda]" >&2; exit 2 ;;
esac

"$SETUP_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11"'
"$SETUP_PYTHON" -m venv .venv
PYTHON="$(pwd)/.venv/bin/python"
"$PYTHON" -m pip install --disable-pip-version-check \
    --index-url "$TORCH_INDEX" "$TORCH_REQUIREMENT"
"$PYTHON" -m pip install --disable-pip-version-check -r requirements.txt
TRAINING_PYTHON="$PYTHON" bash scripts/setup_tesseract.sh
"$PYTHON" -m pip check
if [ "$MODE" = --cuda ]; then
    "$PYTHON" -c 'from training_data.planar_488.dependencies import load_noah_builder, load_nvidia_predecoder, load_tesseract; load_noah_builder(); load_nvidia_predecoder(); load_tesseract()'
    echo "Dependencies installed. Submit the GPU check with: sbatch scripts/smoke_test.sbatch"
else
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PYTHON" -B scripts/smoke_test.py
fi
echo "Activate the environment with: source .venv/bin/activate"
