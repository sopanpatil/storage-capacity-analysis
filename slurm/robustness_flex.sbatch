#!/bin/bash
# Robustness item J (src/robustness_flex.py): FLEX on the fixed subset, one array task per catchment (robustness/subset.csv,
# 397). Outputs in <workspace>/robustness/flex/; catchments already done are skipped, so the array can be resubmitted.
#   sbatch slurm/robustness_flex.sbatch
#SBATCH --job-name=sca-flex
#SBATCH --account=ceh_generic
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --array=0-396%200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/robustness_flex_%A_%a.out

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
module load jaspy
source ~/venvs/storage-capacity-analysis/bin/activate
hostname
python src/robustness_flex.py --samples 40000 --list robustness/subset.csv \
    --task-index "$SLURM_ARRAY_TASK_ID" --tasks 397
