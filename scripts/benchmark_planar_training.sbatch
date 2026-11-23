#!/bin/bash
#SBATCH --job-name=planar488-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --output=verification/precision-%j.out
#SBATCH --error=verification/precision-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${1:?Pass the frozen source directory}"
STUDY_SOURCE="$1"
shift
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd "$STUDY_SOURCE"
TRAINING_PYTHON="${TRAINING_PYTHON:-python}"
TRAINING_PYTHON="$(command -v "$TRAINING_PYTHON")"
srun --cpu-bind=cores "$TRAINING_PYTHON" -u -B verification/benchmark_training_precision.py "$@"
