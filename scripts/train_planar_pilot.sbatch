#!/bin/bash
#SBATCH --job-name=planar488-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --signal=USR1@180
#SBATCH --output=verification/pilot-%j.out
#SBATCH --error=verification/pilot-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"

# Della selects QoS. Slurm owns CUDA_VISIBLE_DEVICES and GPU placement.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

TRAINING_PYTHON="${TRAINING_PYTHON:-python}"
TRAINING_PYTHON="$(command -v "$TRAINING_PYTHON")"

# Run settings default to PilotConfiguration; extra arguments override them.
srun --cpu-bind=cores "$TRAINING_PYTHON" -u -m training_data.planar_488.jobs.train \
    --output "verification/pilot_runs/$SLURM_JOB_ID" "$@"
