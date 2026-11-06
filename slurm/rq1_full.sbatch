#!/bin/bash
# RQ1 identifiability for all screened catchments (src/identifiability.py), 50 tasks.
# Submit after calibrate_full.sbatch has finished (e.g. --dependency=afterok:<its job ID>); afterwards run
#   python src/identifiability.py --list data/catalog/screened_catchments.csv --combine
#SBATCH --job-name=sca-rq1
#SBATCH --account=ceh_generic
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --exclude=host1262
#SBATCH --array=0-49
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/rq1_%A_%a.out

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
module load jaspy
source ~/venvs/storage-capacity-analysis/bin/activate
python -W ignore src/identifiability.py --list data/catalog/screened_catchments.csv \
    --task-index "$SLURM_ARRAY_TASK_ID" --tasks 50
