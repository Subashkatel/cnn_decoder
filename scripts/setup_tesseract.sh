#!/bin/bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIRECTORY="$(dirname -- "$SCRIPT_PATH")"
cd -- "$SCRIPT_DIRECTORY/.."
TRAINING_PYTHON="${TRAINING_PYTHON:-python}"

# Install only the compatible decoder wheel into the project, without shared dependencies.
"$TRAINING_PYTHON" -m pip install --no-deps --upgrade \
    --target .runtime/tesseract tesseract-decoder==0.1.1.dev20260822020007
"$TRAINING_PYTHON" -c 'from training_data.planar_488.dependencies import load_tesseract; load_tesseract()'
