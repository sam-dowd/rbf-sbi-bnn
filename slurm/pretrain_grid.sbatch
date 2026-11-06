#!/bin/bash
# Account: set SBATCH_ACCOUNT in your environment (e.g. export SBATCH_ACCOUNT=<project>_g for GPU jobs).
# Written for NERSC Perlmutter (module load python; constraint gpu/cpu); adjust for other clusters.
# Pretraining-only (lr, wd) grid for the BNN's network (2026-11-02): scripts/tuning/optimize_pretrain_grid.py, one size per
# job. 17 lr x 15 wd + the BNN's used point, 3 draws, one network per fit; ~5.5 s per fit on an A100 after compile (timed from
# pretrained_ens job 60696836), so ~75-80 min per size, up to ~2.5 h if each new lr recompiles. Output
# results/hp_search/pretrain_grid/pretrain_grid_n{n}.npz; a size whose file exists is skipped.
# Usage: for n in 50 100 500 1000 5000 10000; do sbatch -J pretrain_grid_n$n slurm/pretrain_grid.sbatch $n; done
#SBATCH -J pretrain_grid
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 04:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python
conda activate "${RBF_SBI_CONDA_ENV:-rbf-sbi}"
SCRATCH="${RBF_SBI_SCRATCH:-${PSCRATCH:-$PWD/scratch}}"
export TQDM_DISABLE=1
export CUDA_VISIBLE_DEVICES=0
echo "=== n=${1:?size}  $(date)"
python scripts/tuning/optimize_pretrain_grid.py "$1"
echo "=== exit $?  $(date)"
