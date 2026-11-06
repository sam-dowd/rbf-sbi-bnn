#!/bin/bash
# Reproduction check (run after any change to src/): re-fit the N(0,1) BNN at n=100, repeat 0, into scratch and
# compare with the released result. GPU runs are not bit-reproducible; expect agreement within the repeat-to-repeat
# spread (test NLL sd 0.06 over repeats at n=100).
#   export SBATCH_ACCOUNT=<project>_g; bash slurm/verify_reproduction.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
SCRATCH="${RBF_SBI_SCRATCH:-${PSCRATCH:-$PWD/scratch}}"
OUT="$SCRATCH/rbf_sbi_verify"
mkdir -p "$OUT"
sbatch -J verify_reproduction -C gpu -q debug -N 1 -n 1 -c 32 --gpus-per-task=1 -t 00:20:00 -o logs/%x_%j.out -e logs/%x_%j.err \
    --wrap "module load python; conda activate \"${RBF_SBI_CONDA_ENV:-rbf-sbi}\"; export CUDA_VISIBLE_DEVICES=0 TQDM_DISABLE=1 \
RUN_EXPERIMENT_OUT_DIR=$OUT RBF_SBI_SCRATCH=$OUT; python scripts/fit/run_experiment.py final_bnn_stdnormal_pretuned 100 '' 0"
echo "When it finishes:"
echo "  python -c \"import numpy as np; a=np.load('results/raw/final/final_bnn_stdnormal_pretuned_n100_r0.npz'); \\"
echo "    b=np.load('$OUT/final_bnn_stdnormal_pretuned_n100_r0.npz'); print(-a['log_prob_true'].mean(), -b['log_prob_true'].mean())\""
