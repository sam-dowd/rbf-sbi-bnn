#!/bin/bash
# Account: set SBATCH_ACCOUNT in your environment (e.g. export SBATCH_ACCOUNT=<project>_g for GPU jobs).
# Written for NERSC Perlmutter (module load python; constraint gpu/cpu); adjust for other clusters.
# Gaussian predictor with long pretraining (2026-11-04; scripts/fit/gaussian_predictor_long.py), packed on one full GPU
# debug node. Items "n:rspec" (rspec e.g. 0-1 or 3) are dealt round-robin over 4 lanes (one JAX process per GPU);
# each lane runs its items in turn. Outputs go straight to results/raw/final/ and existing ones are skipped, so a
# time-limit kill loses only the repeat in flight -- resubmit the same items (the queue feeder's last job does this).
# CHECK=1: lane 0 first reruns n=50 r0 with the OLD budget (2,000 / 5) into $SCRATCH/sbi_gp_long_check and compares
# with the saved final_bnn_pretrained_ens12_pretuned file.
#SBATCH -J lx_gp_long
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH --gpus-per-node=4
#SBATCH -t 00:30:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python
conda activate "${RBF_SBI_CONDA_ENV:-rbf-sbi}"
SCRATCH="${RBF_SBI_SCRATCH:-${PSCRATCH:-$PWD/scratch}}"
export TQDM_DISABLE=1
items=("$@")
nvidia-smi --query-gpu=index,name --format=csv,noheader || true
pids=()
for g in 0 1 2 3; do
    mine=(); for ((i = g; i < ${#items[@]}; i += 4)); do mine+=("${items[$i]}"); done
    [ ${#mine[@]} -eq 0 ] && [ "$g" != 0 ] && continue
    ( export CUDA_VISIBLE_DEVICES=$g
      if [ "$g" = 0 ] && [ "${CHECK:-0}" = 1 ]; then
          python scripts/fit/gaussian_predictor_long.py 50 0 --default-budget --out-dir "$SCRATCH/sbi_gp_long_check" \
              > "logs/lx_gp_long_check_${SLURM_JOB_ID}.log" 2>&1 \
              && echo "check done: $(grep -c IDENTICAL "logs/lx_gp_long_check_${SLURM_JOB_ID}.log")/4 IDENTICAL" || echo "check FAIL"
      fi
      for it in "${mine[@]}"; do
          IFS=: read -r n rs <<<"$it"
          lg="logs/lx_gp_long_n${n}_r${rs}_${SLURM_JOB_ID}.log"
          python scripts/fit/gaussian_predictor_long.py "$n" "$rs" > "$lg" 2>&1 || { echo "FAIL $it (gpu $g)"; continue; }
          echo "done $it (gpu $g): $(grep -c '^done' "$lg") fitted, $(grep -c '^skip' "$lg") skipped"
      done ) &
    pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "exit status $rc"; exit $rc
