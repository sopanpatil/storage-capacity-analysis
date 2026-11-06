#!/bin/bash
# RQ1 identifiability at the sensitivity behavioural thresholds, 50 tasks per threshold.
# Submit with --export=ALL,BEH_KGE=0.3 (or 0.7); afterwards run
#   python src/identifiability.py --list data/catalog/screened_catchments.csv --behavioural-kge <T> --combine
#SBATCH --job-name=sca-rq1-kge
#SBATCH --account=ceh_generic
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --exclude=host1262
#SBATCH --array=0-49
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/rq1_kge_%A_%a.out

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
module load jaspy
source ~/venvs/storage-capacity-analysis/bin/activate
python -W ignore src/identifiability.py --list data/catalog/screened_catchments.csv \
    --behavioural-kge "$BEH_KGE" --task-index "$SLURM_ARRAY_TASK_ID" --tasks 50
