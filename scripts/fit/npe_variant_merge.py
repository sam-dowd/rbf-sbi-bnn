"""Merge the 12 members per (variant, n, repeat) from scripts/fit/npe_variant_members.py into ensemble + single-network files
with the final_npe_ens12 / final_npe schema (2026-11-04):
  flow16  -> results/raw/final/final_npe_ctx16_ens12_n{n}_r{r}.npz       + final_npe_ctx16_n{n}_r{r}.npz
  gauss16 -> results/raw/final/final_npe_gauss_ctx16_ens12_n{n}_r{r}.npz + final_npe_gauss_ctx16_n{n}_r{r}.npz
plus npe_steps, npe_patience, npe_eval_every, head, context_dim, member_steps_run, member_best_step,
member_stopped_early, member_fit_sec. Aggregation = scripts/npe_long_merge.aggregate (copied from
run_experiment.fit_npe_ensemble, selfcheck-verified); TARP/SBC from an equal share of each member's own draws.
Refuses to overwrite and to merge incomplete / mixed-budget repeats. One experiment-log entry per (variant, n).
Usage: python scripts/fit/npe_variant_merge.py VARIANT N R [R ...] [--budget auto|default|long] [--member-dir DIR]
  R may be a range "0-19".
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # this repository's src/
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling scripts in this folder
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401
from rbf_sbi import data                     # noqa: E402
from rbf_sbi import metrics                  # noqa: E402
from rbf_sbi import experiment_log           # noqa: E402
from npe_long_merge import aggregate                                    # noqa: E402
from npe_variant_members import (VARIANTS, budget_for, member_path, parse_ks, DEFAULT_OUT)   # noqa: E402

FINAL = RESULTS / "raw" / "final"
K = 12
TARP_N_DRAWS, TARP_SEEDS = 1000, list(range(20))
NAMES = {"flow16": ("final_npe_ctx16_ens12", "final_npe_ctx16"),
         "gauss16": ("final_npe_gauss_ctx16_ens12", "final_npe_gauss_ctx16")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=sorted(NAMES))
    ap.add_argument("n", type=int)
    ap.add_argument("repeats", nargs="+")
    ap.add_argument("--budget", choices=["auto", "default", "long"], default="auto")
    ap.add_argument("--member-dir", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    steps, patience = budget_for(a.n, a.budget)
    ens_name, one_name = NAMES[a.variant]
    theta_np = np.asarray(data.load_dataset()[3])
    merged = []
    for r in parse_ks(a.repeats):
        ens_out, one_out = FINAL / f"{ens_name}_n{a.n}_r{r}.npz", FINAL / f"{one_name}_n{a.n}_r{r}.npz"
        if ens_out.exists() or one_out.exists():
            print(f"skip r={r}: output exists"); continue
        files = [member_path(a.member_dir, a.variant, a.n, r, k, steps, patience) for k in range(K)]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            print(f"skip r={r}: {len(missing)} member(s) missing"); continue
        ms = [dict(np.load(f, allow_pickle=True)) for f in files]
        ref = np.load(FINAL / f"final_npe_ens12_n{a.n}_r{r}.npz", allow_pickle=True)
        for k, m in enumerate(ms):
            assert int(m["member"]) == k and int(m["repeat"]) == r and int(m["n_train"]) == a.n
            assert str(m["variant"]) == a.variant and int(m["steps"]) == steps and int(m["patience"]) == patience
            assert int(m["member_seed"]) == int(ref["member_seeds"][k])
            assert float(m["lr"]) == float(ref["lr"]) and float(m["wd"]) == float(ref["wd"])
        lp_k = np.stack([m["log_prob_true"] for m in ms]).astype(np.float64)
        m_mean, m_std = np.stack([m["mean"] for m in ms]), np.stack([m["std"] for m in ms])
        per = -(-TARP_N_DRAWS // K)
        samples = np.concatenate([np.asarray(m["theta_samples"])[:per] for m in ms], 0)[:TARP_N_DRAWS]
        ens = aggregate(lp_k, m_mean, m_std)
        ens.update(sbc_rank=metrics.sbc_marginal_rank(samples, theta_np), member_mean=m_mean, member_std=m_std,
                   member_log_prob_true=lp_k)
        ens_f = metrics.compute_tarp_f_stats(samples, theta_np, data.TARP_LO, data.TARP_HI, TARP_SEEDS)
        extra = dict(npe_steps=steps, npe_patience=patience, npe_eval_every=int(ms[0]["eval_every"]),
                     head=str(ms[0]["head"]), context_dim=int(ms[0]["context_dim"]))
        common = dict(n_train=a.n, repeat=r, seed=int(ref["seed"]), lr=float(ref["lr"]), wd=float(ref["wd"]),
                      X_test=ref["X_test"], theta_test=ref["theta_test"], X_train=ref["X_train"],
                      theta_train=ref["theta_train"], hp_file=str(ref["hp_file"]), gpu_name=str(ms[0]["gpu_name"]),
                      fit_sec_ensemble_total=float(sum(float(m["fit_sec"]) for m in ms)), **extra)
        steps_run = np.array([int(m["steps_run"]) for m in ms]); best = np.array([int(m["best_step"]) for m in ms])
        early = np.array([bool(m["stopped_early"]) for m in ms])
        np.savez_compressed(ens_out.with_suffix(".tmp.npz"), **common, **ens, f=ens_f, n_ensemble=K,
                            member_seeds=np.asarray(ref["member_seeds"]), member_steps_run=steps_run,
                            member_best_step=best, member_stopped_early=early,
                            member_fit_sec=np.array([float(m["fit_sec"]) for m in ms]))
        os.replace(ens_out.with_suffix(".tmp.npz"), ens_out)
        m0 = ms[0]
        np.savez_compressed(one_out.with_suffix(".tmp.npz"), **common, mean=m0["mean"], std=m0["std"], f=m0["f"],
                            log_prob_true=m0["log_prob_true"], sbc_rank=m0["sbc_rank"], steps_run=int(m0["steps_run"]),
                            best_step=int(m0["best_step"]), stopped_early=bool(m0["stopped_early"]),
                            val_history=m0["val_history"])
        os.replace(one_out.with_suffix(".tmp.npz"), one_out)
        nll = -float(ens["log_prob_true"].mean())
        merged.append((r, nll, -float(np.mean(m0["log_prob_true"])), int(np.median(steps_run)), int(early.sum())))
        print(f"r={r}: {ens_name} NLL {nll:+.4f} (NPE ens12 {-float(np.mean(ref['log_prob_true'])):+.4f}); "
              f"steps_run median {int(np.median(steps_run))}, stopped early {int(early.sum())}/{K}")
    if merged:
        v = np.array([m[1] for m in merged]); v1 = np.array([m[2] for m in merged])
        experiment_log.log_run(
            "scripts/fit/npe_variant_members.py + scripts/fit/npe_variant_merge.py",
            config=dict(model=ens_name, single_model=one_name, n_train=a.n, repeats=[m[0] for m in merged],
                        n_members=K, steps=steps, patience_evals=patience, **{k: v_ for k, v_ in VARIANTS[a.variant].items()},
                        lr_wd="NPE's (final_npe_ens12, not re-tuned)"),
            results=dict(test_nll_ens12_mean=round(float(v.mean()), 4),
                         test_nll_ens12_sd=round(float(v.std(ddof=1)), 4) if len(v) > 1 else None,
                         test_nll_single_mean=round(float(v1.mean()), 4),
                         median_steps_run=[m[3] for m in merged], members_stopped_early=[m[4] for m in merged]),
            output_path=f"results/raw/final/{ens_name}_n{a.n}_r*.npz",
            notes="NPE head x body comparison (2026-11-04): one architectural change vs final_npe_ens12, everything "
                  "else identical (subsample, split, seeds, lr/wd, optimiser, budget of the size's NPE comparator). "
                  "Members fit on gpu_debug via slurm/npe_variant_packed.sbatch.")


if __name__ == "__main__":
    main()
