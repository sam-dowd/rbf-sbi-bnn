#!/bin/bash
# Submit the final runs of the paper: the five BNNs (N(0,1), data-free scale, tempered, N(0,1) + re-tune, empirical
# Bayes) and the 12-flow NPE ensemble; repeats 0-4 at every n and 5-19 at n <= 100 (tempered: n <= 1,000 only).
# One job per (model, n, repeat); existing result files are skipped. The N(0,1) job also writes the capped Gaussian
# predictor (final_bnn_pretrained*_pretuned). Merges the research repository's submit_final_sweep.sh and
# submit_lowdata_extra.sh.
#   export SBATCH_ACCOUNT=<project>_g
#   bash slurm/submit_final_sweep.sh [list|submit]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
MODE="${1:-list}"
MODELS="final_bnn_stdnormal_pretuned final_bnn_gaussprior_pretuned final_bnn_temp_pretuned \
final_bnn_stdnormal_retune_pretuned final_bnn_ebblock_pretuned final_npe_ens12"
minutes() {   # rough wall time per fit, for the time limit
    case "$1" in
        *retune*|*ebblock*) echo 15 ;;
        final_npe_ens12) if [ "$2" -ge 5000 ]; then echo 30; else echo 10; fi ;;
        *) if [ "$2" -ge 5000 ]; then echo 8; else echo 5; fi ;;
    esac
}
count=0; skipped=0
for m in $MODELS; do
    for n in 50 100 500 1000 5000 10000; do
        if [ "$m" = final_bnn_temp_pretuned ] && [ "$n" -gt 1000 ]; then continue; fi
        last=4; if [ "$n" -le 100 ]; then last=19; fi
        tl=$(( $(minutes "$m" "$n") * 2 + 10 ))
        for r in $(seq 0 $last); do
            if [ -f "results/raw/final/${m}_n${n}_r${r}.npz" ]; then skipped=$((skipped + 1)); continue; fi
            cmd="sbatch -J ${m#final_}_n${n}_r${r} -t $tl slurm/final_sweep.sbatch $m $n $r"
            if [ "$MODE" = submit ]; then $cmd; else echo "$cmd"; fi
            count=$((count + 1))
        done
    done
done
echo "$count jobs (mode=$MODE), $skipped already done" >&2
