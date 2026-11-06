#!/bin/bash
# Account: set SBATCH_ACCOUNT in your environment (e.g. export SBATCH_ACCOUNT=<project>_g for GPU jobs).
# Written for NERSC Perlmutter (module load python; constraint gpu/cpu); adjust for other clusters.
# NPE architecture variants (head x body comparison, 2026-11-04), packed on one full GPU debug node. Items
# "variant:n:r:kspec" (kspec e.g. 0-11 or 3) are dealt round-robin over LANES lanes (default 8 = 2 processes per GPU;
# ASSUMPTION: the tiny NPE networks leave a GPU under-used, so 2 per GPU raises throughput -- unverified); each lane
# runs its items in turn, one process per item. Members go to $SCRATCH/sbi_npe_variants/members (existing ones are
# skipped, so a time-limit kill loses only the members in flight -- resubmit the same items).
# CHECK=1: lane 0 first reruns final_npe_ens12 n=10000 r0 member 0 (variant base8, default budget) and compares it with
# the saved member (GPU reproduction check of the npe.fit context_dim change).
#   CHECK=1 sbatch slurm/npe_variant_packed.sbatch flow16:50:0:0-11 gauss16:50:0:0-11 ...
#SBATCH -J lx_npe_var
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
LANES=${LANES:-8}
items=("$@")
nvidia-smi --query-gpu=index,name --format=csv,noheader || true
pids=()
for ((l = 0; l < LANES; l++)); do
    mine=(); for ((i = l; i < ${#items[@]}; i += LANES)); do mine+=("${items[$i]}"); done
    [ ${#mine[@]} -eq 0 ] && [ "$l" != 0 ] && continue
    ( export CUDA_VISIBLE_DEVICES=$((l % 4))
      if [ "$l" = 0 ] && [ "${CHECK:-0}" = 1 ]; then
          python scripts/fit/npe_variant_members.py base8 10000 0 0 --budget default \
              --out-dir "$SCRATCH/sbi_npe_variants/check" --compare > "logs/lx_npe_var_check_${SLURM_JOB_ID}.log" 2>&1 \
              && echo "check done: $(grep COMPARE "logs/lx_npe_var_check_${SLURM_JOB_ID}.log")" || echo "check FAIL"
      fi
      for it in "${mine[@]}"; do
          IFS=: read -r v n r ks <<<"$it"
          lg="logs/lx_npe_var_${v}_n${n}_r${r}_k${ks}_${SLURM_JOB_ID}.log"
          python scripts/fit/npe_variant_members.py "$v" "$n" "$r" "$ks" > "$lg" 2>&1 \
              || { echo "FAIL $it (lane $l)"; continue; }
          echo "done $it (lane $l): $(grep -c '^done' "$lg") members, $(grep -c '^skip' "$lg") skipped"
      done ) &
    pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "exit status $rc"; exit $rc
