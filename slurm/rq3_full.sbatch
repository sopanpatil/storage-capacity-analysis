#!/bin/bash
# RQ3 model-state mass curves for all screened catchments (src/state_capacity.py; optimum + all
# behavioural sets), 200 tasks of ~31 catchments. Submit after calibrate_full.sbatch has finished
# (e.g. --dependency=afterok:<its job ID>); afterwards run  python src/state_capacity.py --list data/catalog/screened_catchments.csv --combine
# (parts and the full table go to <runs>/rq3/; the per-catchment summary to data/rq3/)
#SBATCH --job-name=sca-rq3
#SBATCH --account=ceh_generic
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --exclude=host1262
#SBATCH --array=0-199%100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/rq3_%A_%a.out

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
module load jaspy
source ~/venvs/storage-capacity-analysis/bin/activate
python -W ignore src/state_capacity.py --list data/catalog/screened_catchments.csv \
    --task-index "$SLURM_ARRAY_TASK_ID" --tasks 200
