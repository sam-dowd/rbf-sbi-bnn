#!/bin/bash
# Account: set SBATCH_ACCOUNT in your environment (e.g. export SBATCH_ACCOUNT=<project>_g for GPU jobs).
# Written for NERSC Perlmutter (module load python; constraint gpu/cpu); adjust for other clusters.
#SBATCH -J optimize_npe
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 01:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# NPE learning rate and weight decay per size -> results/npe_params.npz (Appendix C).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python
conda activate "${RBF_SBI_CONDA_ENV:-rbf-sbi}"
export TQDM_DISABLE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
python scripts/tuning/optimize_npe.py
