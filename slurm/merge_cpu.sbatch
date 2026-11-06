#!/bin/bash
# Account: set SBATCH_ACCOUNT in your environment (e.g. export SBATCH_ACCOUNT=<project>_g for GPU jobs).
# Written for NERSC Perlmutter (module load python; constraint gpu/cpu); adjust for other clusters.
#SBATCH -J merge
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:15:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# Merge member files into ensemble result files, e.g.
#   sbatch slurm/merge_cpu.sbatch scripts/fit/npe_long_merge.py 10000 0 1 2 3 4
#   sbatch slurm/merge_cpu.sbatch scripts/fit/npe_variant_merge.py <arguments>
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python
conda activate "${RBF_SBI_CONDA_ENV:-rbf-sbi}"
SCRATCH="${RBF_SBI_SCRATCH:-${PSCRATCH:-$PWD/scratch}}"
export JAX_PLATFORMS=cpu
python "$@"
