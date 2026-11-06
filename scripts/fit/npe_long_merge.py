"""Merge the 12 long-training NPE members of each repeat (scripts/fit/npe_long_members.py) into ensemble + single-flow
files with the final_npe_ens12 / final_npe schema (2026-11-04):
  results/raw/final/final_npe_ens12_long_n{n}_r{r}.npz, results/raw/final/final_npe_long_n{n}_r{r}.npz
plus extra keys npe_steps, npe_patience, npe_eval_every, member_steps_run, member_best_step, member_stopped_early,
member_fit_sec (ensemble file) and steps_run / best_step / stopped_early / val_history (single file).

Aggregation copied verbatim from scripts/fit/run_experiment.py::fit_npe_ensemble (equal-weight mixture: log density by
log-mean-exp, TARP/SBC from an equal share of each flow's own draws, mean/std from the mixture's first two moments).
The script checks the copy on the default-budget file before using it: --selfcheck recomputes final_npe_ens12's
saved log_prob_true / mean / std from its saved member arrays (sbc_rank / f need per-member draws, not saved there).

Writes one experiment-log entry per merged repeat. Refuses to overwrite an existing output and to merge members with
mixed budgets / seeds.
Usage: python scripts/fit/npe_long_merge.py N R [R ...] [--steps 200000] [--patience 100] [--member-dir DIR] [--selfcheck]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # this repository's src/
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401
from rbf_sbi import data            # noqa: E402
from rbf_sbi import metrics         # noqa: E402
from rbf_sbi import experiment_log  # noqa: E402

FINAL = RESULTS / "raw" / "final"
K = 12
TARP_N_DRAWS, TARP_SEEDS = 1000, list(range(20))   # as scripts/fit/run_experiment.py
DEFAULT_MEMBERS = SCRATCH / "sbi_npe_long" / "members"


def _logsumexp0(a):   # verbatim from run_experiment.py
    m = np.max(a, axis=0)
    return m + np.log(np.sum(np.exp(a - m), axis=0))


def aggregate(lp_k, m_mean, m_std):
    return dict(mean=m_mean.mean(0), std=np.sqrt((m_std ** 2).mean(0) + m_mean.var(0)),
                log_prob_true=_logsumexp0(lp_k) - np.log(lp_k.shape[0]))


def selfcheck(n, r):
    d = np.load(FINAL / f"final_npe_ens12_n{n}_r{r}.npz", allow_pickle=True)
    agg = aggregate(np.asarray(d["member_log_prob_true"], np.float64), d["member_mean"], d["member_std"])
    for key in ("log_prob_true", "mean", "std"):
        ok = np.allclose(agg[key], d[key], rtol=0, atol=1e-6)
        print(f"selfcheck n={n} r={r} {key}: {'OK' if ok else 'MISMATCH'} (max|d| {np.max(np.abs(agg[key] - d[key])):.2e})")
        assert ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("repeats", type=int, nargs="+")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--member-dir", type=Path, default=DEFAULT_MEMBERS)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    X_test, theta_test = data.load_dataset()[2:]
    theta_np = np.asarray(theta_test)

    for r in a.repeats:
        if a.selfcheck:
            selfcheck(a.n, r)
        ens_out = FINAL / f"final_npe_ens12_long_n{a.n}_r{r}.npz"
        one_out = FINAL / f"final_npe_long_n{a.n}_r{r}.npz"
        if ens_out.exists() or one_out.exists():
            print(f"skip r={r}: output exists"); continue
        files = [a.member_dir / f"npe_long_s{a.steps}_p{a.patience}_n{a.n}_r{r}_k{k}.npz" for k in range(K)]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            print(f"skip r={r}: {len(missing)} member(s) missing: {missing}"); continue
        ms = [dict(np.load(f, allow_pickle=True)) for f in files]
        ref = np.load(FINAL / f"final_npe_ens12_n{a.n}_r{r}.npz", allow_pickle=True)
        for k, m in enumerate(ms):
            assert int(m["member"]) == k and int(m["repeat"]) == r and int(m["n_train"]) == a.n
            assert int(m["steps"]) == a.steps and int(m["patience"]) == a.patience
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

        budget = dict(npe_steps=a.steps, npe_patience=a.patience, npe_eval_every=int(ms[0]["eval_every"]))
        common = dict(n_train=a.n, repeat=r, seed=int(ref["seed"]), lr=float(ref["lr"]), wd=float(ref["wd"]),
                      X_test=ref["X_test"], theta_test=ref["theta_test"], X_train=ref["X_train"],
                      theta_train=ref["theta_train"], hp_file=str(ref["hp_file"]), gpu_name=str(ms[0]["gpu_name"]),
                      fit_sec_ensemble_total=float(sum(float(m["fit_sec"]) for m in ms)), **budget)
        steps_run = np.array([int(m["steps_run"]) for m in ms])
        best_step = np.array([int(m["best_step"]) for m in ms])
        early = np.array([bool(m["stopped_early"]) for m in ms])
        np.savez_compressed(ens_out.with_suffix(".tmp.npz"), **common, **ens, f=ens_f, n_ensemble=K,
                            member_seeds=np.asarray(ref["member_seeds"]), member_steps_run=steps_run,
                            member_best_step=best_step, member_stopped_early=early,
                            member_fit_sec=np.array([float(m["fit_sec"]) for m in ms]))
        os.replace(ens_out.with_suffix(".tmp.npz"), ens_out)
        m0 = ms[0]
        np.savez_compressed(one_out.with_suffix(".tmp.npz"), **common, mean=m0["mean"], std=m0["std"], f=m0["f"],
                            log_prob_true=m0["log_prob_true"], sbc_rank=m0["sbc_rank"], steps_run=int(m0["steps_run"]),
                            best_step=int(m0["best_step"]), stopped_early=bool(m0["stopped_early"]),
                            val_history=m0["val_history"])
        os.replace(one_out.with_suffix(".tmp.npz"), one_out)

        nll_long, nll_ref = -float(ens["log_prob_true"].mean()), -float(np.mean(ref["log_prob_true"]))
        nll_1, nll_1ref = -float(np.mean(m0["log_prob_true"])), -float(np.mean(ref["member_log_prob_true"][0]))
        print(f"r={r}: ens12 NLL long {nll_long:+.4f} vs default {nll_ref:+.4f}; single {nll_1:+.4f} vs {nll_1ref:+.4f}; "
              f"steps_run median {int(np.median(steps_run))} (max {steps_run.max()}), best_step median "
              f"{int(np.median(best_step))}, stopped early {int(early.sum())}/{K} -> {ens_out.name}")
        experiment_log.log_run(
            "scripts/fit/npe_long_members.py + scripts/fit/npe_long_merge.py",
            config=dict(model="final_npe_ens12_long", n_train=a.n, repeat=r, n_members=K, seed=int(ref["seed"]),
                        lr=float(ref["lr"]), wd=float(ref["wd"]), steps=a.steps, patience_evals=a.patience,
                        eval_every=budget["npe_eval_every"], default_budget="steps 20000, patience 20"),
            results=dict(test_nll_ens12=round(nll_long, 4), test_nll_ens12_default_budget=round(nll_ref, 4),
                         test_nll_single=round(nll_1, 4), test_nll_single_default_budget=round(nll_1ref, 4),
                         member_steps_run=steps_run.tolist(), member_best_step=best_step.tolist(),
                         members_stopped_early=int(early.sum()),
                         fit_sec_total=round(common["fit_sec_ensemble_total"], 1)),
            output_path=str(ens_out.relative_to(ROOT)),
            notes="Long-training NPE ensemble (does NPE's large-n plateau come from its training budget?). Members fit "
                  "on gpu_debug, packed, via slurm/npe_long_packed.sbatch; same subsample / split / seeds / lr / wd as "
                  "final_npe_ens12; only steps and patience changed (ASSUMPTION: 10x / 5x chosen, not from a source).")


if __name__ == "__main__":
    main()
