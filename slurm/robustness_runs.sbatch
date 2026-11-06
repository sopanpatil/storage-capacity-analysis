#!/bin/bash
# Robustness runs A-D (src/robustness_calibrate.py): one array task per subset
# catchment (robustness/subset.csv, 397), runs in priority order (A and its baseline, B, D, C, then the
# optional SAC-SMA optimiser check). Outputs in <workspace>/robustness/<run>/; catchments already done are
# skipped, so the array can be resubmitted.
#   sbatch slurm/robustness_runs.sbatch
#SBATCH --job-name=sca-rob
#SBATCH --account=ceh_generic
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --array=0-396%200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/robustness_%A_%a.out

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
module load jaspy
source ~/venvs/storage-capacity-analysis/bin/activate
hostname
for run in hbv_mo hbv_so hbv_150k gr4j gr6j_sce sacsma_so; do
    extra=""
    [ "$run" = gr4j ] && extra="--samples 100000"
    python src/robustness_calibrate.py --run "$run" $extra --list robustness/subset.csv \
        --task-index "$SLURM_ARRAY_TASK_ID" --tasks 397
done
