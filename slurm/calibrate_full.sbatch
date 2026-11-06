#!/bin/bash
# Full-sample calibration, validation and Monte Carlo: one array task
# per screened catchment (data/catalog/screened_catchments.csv, 6,144), all three models, at most 500
# tasks at once. Catchments already done are skipped.
#   sbatch slurm/calibrate_full.sbatch
#SBATCH --job-name=sca-full
#SBATCH --account=ceh_generic
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --array=0-6143%500
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/full_%A_%a.out

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
module load jaspy
source ~/venvs/storage-capacity-analysis/bin/activate
python src/calibrate.py --tag calibration --list data/catalog/screened_catchments.csv \
    --task-index "$SLURM_ARRAY_TASK_ID" --tasks 6144
