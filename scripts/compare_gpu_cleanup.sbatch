#!/bin/bash
#SBATCH --job-name=planar488-gpu-cleanup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=mig
#SBATCH --gres=gpu:1

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${1:?Pass the completed prepared-data directory}"
: "${2:?Pass a new GPU comparison output directory}"
PREPARED_OUTPUT="$1"
COMPARISON_OUTPUT="$2"
shift 2
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
GPU_CLEANUP_PYTHON="${GPU_CLEANUP_PYTHON:-$PWD/.venv-gpu-cleanup/bin/python}"
if [ -n "${PLANAR_SOURCE_DIRECTORY:-}" ]; then
    cd "$PLANAR_SOURCE_DIRECTORY"
fi
srun --cpu-bind=cores "$GPU_CLEANUP_PYTHON" -u -B -m training_data.planar_488.jobs.compare_gpu_cleanup \
    --prepared "$PREPARED_OUTPUT" --output "$COMPARISON_OUTPUT" "$@"
