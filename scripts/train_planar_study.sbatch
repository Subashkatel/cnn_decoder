#!/bin/bash
#SBATCH --job-name=planar488-study
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --signal=USR1@180
#SBATCH --output=verification/study-%j.out
#SBATCH --error=verification/study-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${1:?Pass the frozen source directory}"
: "${2:?Pass the training output directory}"
STUDY_SOURCE="$1"
STUDY_OUTPUT="$2"
shift 2

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

cd "$STUDY_SOURCE"
TRAINING_PYTHON="${TRAINING_PYTHON:-python}"
TRAINING_PYTHON="$(command -v "$TRAINING_PYTHON")"
srun --cpu-bind=cores "$TRAINING_PYTHON" -u -B -m training_data.planar_488.jobs.train \
    --output "$STUDY_OUTPUT" "$@"
