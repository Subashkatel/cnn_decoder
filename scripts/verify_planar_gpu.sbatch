#!/bin/bash
#SBATCH --job-name=planar488-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --output=verification/gpu-%j.out
#SBATCH --error=verification/gpu-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"

# Della assigns QoS from wall time. CUDA_VISIBLE_DEVICES belongs to Slurm.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

TRAINING_PYTHON="${TRAINING_PYTHON:-python}"
TRAINING_PYTHON="$(command -v "$TRAINING_PYTHON")"

srun --cpu-bind=cores "$TRAINING_PYTHON" -u -m training_data.planar_488.jobs.check_gpu \
    --steps 20 --shots 8 --model-profile planar_pilot \
    --output "verification/gpu_runs/$SLURM_JOB_ID"

# A fresh-batch smoke run exercises the same checkpoint consumed by CPU evaluation.
srun --cpu-bind=cores "$TRAINING_PYTHON" -u -m training_data.planar_488.jobs.train \
    --steps 12 --training-seconds 120 --validation-interval 4 \
    --batch-size 32 --validation-shots 8 --test-shots 8 \
    --output "verification/pilot_runs/$SLURM_JOB_ID"
