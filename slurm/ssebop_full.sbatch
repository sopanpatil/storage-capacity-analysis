#!/bin/bash
# Catchment-mean SSEBop v5 monthly ET for all screened catchments, one task per year (2003-2020).
# Needs the 216 monthly zips from src/download_ssebop.sh. ~40 min per task. Independent of calibration.
#SBATCH --job-name=sca-ssebop
#SBATCH --account=ceh_generic
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --exclude=host1262
#SBATCH --array=2003-2020
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/ssebop_%A_%a.out

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
module load jaspy
source ~/venvs/storage-capacity-analysis/bin/activate
python -W ignore src/benchmark_ssebop.py --list data/catalog/screened_catchments.csv --year "$SLURM_ARRAY_TASK_ID"
