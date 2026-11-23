#!/bin/bash
#SBATCH --job-name=planar488-compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --signal=USR1@180

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through scripts/submit_training_loop.py}"
: "${TRAINING_PYTHON:?Activate the prepared environment before submission}"
STUDY_SOURCE="$1"
PREPARED_OUTPUT="$2"
COMPARISON_OUTPUT="$3"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd "$STUDY_SOURCE"
srun --cpu-bind=cores "$TRAINING_PYTHON" -u -B -m training_data.planar_488.jobs.compare \
    --prepared "$PREPARED_OUTPUT" --output "$COMPARISON_OUTPUT" --timeout 60 --seconds 2400
