#!/bin/bash
#SBATCH --job-name=planar488-tune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=verification/tune-%j.out
#SBATCH --error=verification/tune-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${1:?Pass the frozen source directory}"
: "${2:?Pass the training checkpoint directory}"
STUDY_SOURCE="$1"
TRAINING_OUTPUT="$2"
shift 2
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd "$STUDY_SOURCE"
TRAINING_PYTHON="${TRAINING_PYTHON:-python}"
TRAINING_PYTHON="$(command -v "$TRAINING_PYTHON")"
srun --cpu-bind=cores "$TRAINING_PYTHON" -u -B verification/tune_training_study.py \
    --training-output "$TRAINING_OUTPUT" "$@"
