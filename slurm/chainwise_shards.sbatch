#!/bin/bash
# Account: set SBATCH_ACCOUNT in your environment (e.g. export SBATCH_ACCOUNT=<project>_g for GPU jobs).
# Written for NERSC Perlmutter (module load python; constraint gpu/cpu); adjust for other clusters.
# Chain-wise diagnostics (scripts/analysis/chainwise_diagnostics.py) split by model: one process per model, in parallel on
# one CPU node, each writing its own shard to $SCRATCH/chainwise_shards/; merge afterwards into
# results/rbf_chainwise_diagnostics.npz. Usage: sbatch slurm/chainwise_shards.sbatch model1 model2 ...
#SBATCH -J chainwise_shards
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
module load python
conda activate "${RBF_SBI_CONDA_ENV:-rbf-sbi}"
SCRATCH="${RBF_SBI_SCRATCH:-${PSCRATCH:-$PWD/scratch}}"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
SHARDS="$SCRATCH/chainwise_shards"; mkdir -p "$SHARDS"
for m in "$@"; do
    CHAINWISE_OUT="$SHARDS/$m.npz" python scripts/analysis/chainwise_diagnostics.py "$SCRATCH/sbi_final_draws/${m}_n*_r?_draws.npz" \
        > "$SHARDS/$m.log" 2>&1 &
done
wait
echo "done: $(ls $SHARDS/*.npz | wc -l) shards"
