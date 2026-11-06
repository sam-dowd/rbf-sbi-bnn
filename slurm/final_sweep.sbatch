#!/bin/bash
# Account: set SBATCH_ACCOUNT in your environment (e.g. export SBATCH_ACCOUNT=<project>_g for GPU jobs).
# Written for NERSC Perlmutter (module load python; constraint gpu/cpu); adjust for other clusters.
# FINAL PRODUCTION SWEEP (2026-10-29): one job per (model, training size), looping over the given
# repeats. Models and settings: scripts/fit/run_experiment.py, "FINAL SWEEP" block; plan in
# the paper (Section 4) ("FINAL PRODUCTION SWEEP"). Submit through slurm/submit_final_sweep.sh,
# which sets each job's time limit.
#
#   sbatch -t 00:30:00 slurm/final_sweep.sbatch <final_model> <n_train> [repeats, e.g. "0" or "0,1,2,3,4"]
#
# Output: results/raw/final/{model}_n{n}_r{repeat}.npz (+ final_npe / final_gaussian from the ensemble
# jobs), per-draw BNN outputs in $SCRATCH/sbi_final_draws/. Resumable: existing files are skipped.
#SBATCH -J final_sweep
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 01:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load python
conda activate "${RBF_SBI_CONDA_ENV:-rbf-sbi}"
SCRATCH="${RBF_SBI_SCRATCH:-${PSCRATCH:-$PWD/scratch}}"
export TQDM_DISABLE=1   # no progress bars: per-step MCLMC bars made .err logs 200-370 MB each and filled $HOME (2026-10-29)

# Single-GPU visibility for bde's pmap-based warmup (see slurm/run_experiment.sbatch).
export CUDA_VISIBLE_DEVICES=0
# NPE jobs: stop JAX pre-allocating the GPU so torch can initialise cuBLAS (see slurm/run_experiment.sbatch).
if [ "$1" = "final_npe_ens12" ]; then
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python scripts/fit/run_experiment.py "$1" "$2" "" "${3:-0}"
