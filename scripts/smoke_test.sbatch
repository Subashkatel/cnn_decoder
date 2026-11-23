#!/bin/bash
#SBATCH --job-name=planar488-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --output=verification/smoke-%j.out
#SBATCH --error=verification/smoke-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
TRAINING_PYTHON="${TRAINING_PYTHON:-$PWD/.venv/bin/python}"
srun --cpu-bind=cores "$TRAINING_PYTHON" -u -B scripts/smoke_test.py --device cuda "$@"
