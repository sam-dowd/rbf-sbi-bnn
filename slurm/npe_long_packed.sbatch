#!/bin/bash
# Account: set SBATCH_ACCOUNT in your environment (e.g. export SBATCH_ACCOUNT=<project>_g for GPU jobs).
# Written for NERSC Perlmutter (module load python; constraint gpu/cpu); adjust for other clusters.
# Long-training NPE members, packed on one full GPU debug node (2026-11-04). Items "n:r:k" are dealt round-robin over
# the 4 GPUs; each GPU runs its items in turn, ONE PROCESS PER MEMBER (scripts/fit/npe_long_members.py writes each member
# file as soon as it is done and skips existing ones, so a time-limit kill loses only the members in flight -- resubmit
# the same items). Members go to $SCRATCH/sbi_npe_long/members; merge with scripts/fit/npe_long_merge.py.
# CHECK=1: GPU 0 first reruns n=10000 r0 member 0 under the DEFAULT budget (steps 20000, patience 20) into
# $SCRATCH/sbi_npe_long/check and compares it with the saved final_npe_ens12 member (GPU reproduction check).
#   CHECK=1 sbatch slurm/npe_long_packed.sbatch 10000:0:0 10000:0:1 ...
#SBATCH -J lx_npe_long
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
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # data/metrics import JAX; keep it from reserving the GPU torch needs
items=("$@")
nvidia-smi --query-gpu=index,name --format=csv,noheader || true
pids=()
for g in 0 1 2 3; do
    mine=(); for ((i = g; i < ${#items[@]}; i += 4)); do mine+=("${items[$i]}"); done
    ( export CUDA_VISIBLE_DEVICES=$g
      if [ "$g" = 0 ] && [ "${CHECK:-0}" = 1 ]; then
          python scripts/fit/npe_long_members.py 10000 0 0 --steps 20000 --patience 20 \
              --out-dir "$SCRATCH/sbi_npe_long/check" --compare > "logs/lx_npe_long_check_${SLURM_JOB_ID}.log" 2>&1 \
              && echo "check done (gpu 0)" || echo "check FAIL (gpu 0)"
      fi
      for it in "${mine[@]}"; do
          IFS=: read -r n r k <<<"$it"
          python scripts/fit/npe_long_members.py "$n" "$r" "$k" > "logs/lx_npe_long_n${n}_r${r}_k${k}_${SLURM_JOB_ID}.log" 2>&1 \
              || { echo "FAIL $it (gpu $g)"; continue; }
          echo "done $it (gpu $g): $(tail -1 "logs/lx_npe_long_n${n}_r${r}_k${k}_${SLURM_JOB_ID}.log")"
      done ) &
    pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "exit status $rc"; exit $rc
