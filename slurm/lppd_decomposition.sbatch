#!/bin/bash
# Account: set SBATCH_ACCOUNT in your environment (e.g. export SBATCH_ACCOUNT=<project>_g for GPU jobs).
# Written for NERSC Perlmutter (module load python; constraint gpu/cpu); adjust for other clusters.
#SBATCH -J lppd_decomposition
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# Decomposition of the BNN's NLL (Tables 8-9) from the per-draw outputs in $SCRATCH/sbi_final_draws/,
# one process per model, then merged into results/lppd_decomposition.npz.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python
conda activate "${RBF_SBI_CONDA_ENV:-rbf-sbi}"
SCRATCH="${RBF_SBI_SCRATCH:-${PSCRATCH:-$PWD/scratch}}"
export JAX_PLATFORMS=cpu RBF_SBI_SCRATCH="$SCRATCH"
for m in final_bnn_stdnormal_pretuned final_bnn_gaussprior_pretuned final_bnn_temp_pretuned; do
    sizes="50 100 500 1000 5000 10000"; [ "$m" = final_bnn_temp_pretuned ] && sizes="50 100 500 1000"
    python scripts/analysis/lppd_decomposition.py "$m" $sizes > "logs/lppd_${m}.log" 2>&1 &
done
wait
python scripts/analysis/lppd_decomposition.py --merge
