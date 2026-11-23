#!/bin/bash
#SBATCH --job-name=planar488-loop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --signal=USR1@180

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through scripts/submit_training_loop.py}"
: "${TRAINING_PYTHON:?Activate the prepared environment before submission}"
STUDY_SOURCE="$1"
TRAINING_OUTPUT="$2"
PREPARED_OUTPUT="$3"
SEED_BASE="$4"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd "$STUDY_SOURCE"

srun --cpu-bind=cores "$TRAINING_PYTHON" -u -B -m training_data.planar_488.jobs.train \
    --output "$TRAINING_OUTPUT" --distance 7 --rounds 7 --probability .001 \
    --training-probability .003 --model-profile compact --batch-size 128 \
    --steps 12000 --training-seconds 1800 --validation-interval 500 \
    --validation-shots 32 --test-shots 64 --optimizer lion --learning-rate .0001 \
    --learning-rate-schedule warmup_then_decay --ema-decay .999 --tf32 \
    --training-seed "$SEED_BASE" --validation-seed "$((SEED_BASE + 100000))" \
    --calibration-seed "$((SEED_BASE + 200000))" --test-seed "$((SEED_BASE + 300000))" \
    --torch-seed "$SEED_BASE"

srun --cpu-bind=cores "$TRAINING_PYTHON" -u -B -m training_data.planar_488.jobs.prepare_comparison \
    --training-output "$TRAINING_OUTPUT" --output "$PREPARED_OUTPUT" \
    --steps 4000 8000 12000 --validation-shots 8 --test-shots 64 \
    --selection-seed "$((SEED_BASE + 400000))"
