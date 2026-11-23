#!/bin/bash
#SBATCH --job-name=planar488-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=verification/eval-%j.out
#SBATCH --error=verification/eval-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${1:?Pass the completed GPU training output directory}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

TRAINING_DIRECTORY="$1"
shift
TRAINING_PYTHON="${TRAINING_PYTHON:-python}"

# Noah's 63-order Tesseract search takes about 3 s per distance-7 shot, so 2,048
# paired test shots need about 3 hours. Save time for the last atomic progress write.
srun --cpu-bind=cores "$TRAINING_PYTHON" -u -m training_data.planar_488.jobs.evaluate \
    --training-output "$TRAINING_DIRECTORY" --output "$TRAINING_DIRECTORY/evaluation" \
    --evaluation-seconds 14100 "$@"
