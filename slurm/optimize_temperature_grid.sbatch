#!/bin/bash
# Account: set SBATCH_ACCOUNT in your environment (e.g. export SBATCH_ACCOUNT=<project>_g for GPU jobs).
# Written for NERSC Perlmutter (module load python; constraint gpu/cpu); adjust for other clusters.
# Coarse per-size temperature search under the plain N(0,1) prior -- see
# scripts/tuning/optimize_temperature_grid.py's module docstring. One job per training size:
#   for n in 50 100 500 1000; do sbatch slurm/optimize_temperature_grid.sbatch $n; done
# Optional second argument --full evaluates every grid point instead of bisecting.
#SBATCH -J temp_grid
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH -o logs/temp_grid_%j.out
#SBATCH -e logs/temp_grid_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load python
conda activate "${RBF_SBI_CONDA_ENV:-rbf-sbi}"
SCRATCH="${RBF_SBI_SCRATCH:-${PSCRATCH:-$PWD/scratch}}"
export PRETUNED_HP=1   # the paper's search: pretraining-grid lr/wd
export TQDM_DISABLE=1   # no progress bars: per-step MCLMC bars made .err logs 200-370 MB each and filled $HOME (2026-10-29)

# See slurm/bde_correlated_perchain_test.sbatch's comment -- forces single-GPU visibility so
# bde's pmap-based MCMC warmup never sees more than one device, regardless of QOS quirks.
export CUDA_VISIBLE_DEVICES=0

python scripts/tuning/optimize_temperature_grid.py "$1" ${2:-}
