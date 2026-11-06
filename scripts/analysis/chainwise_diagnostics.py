"""Chain-wise convergence diagnostics (Sommer et al. 2024, App. B; Vehtari et al. 2021) on the RBF-kernel task, from
the final sweep's per-draw files. Analysis only: reads existing files, fits nothing, so it writes no experiment-log entry.

The RBF counterpart of scripts/chainwise_diagnostics.py (GP surface), whose statistics it imports unchanged
(rank_normal_rhat, split, eq9_stop, q; kappa = 4, Eq. 9 window 100 / eps 0.01). Differences for the two-target task:
  * quantities per test dataset: the predicted mean mu (sigma and rho), the predicted width s (aleatoric, per draw;
    sigma and rho) and the per-draw log density at the truth, log N2(theta* | mu, s, corr), computed here in float64
    (metrics.bivariate_normal_logpdf in float32 gives impossible values once |corr| rounds to 1);
  * LPPD_l per member is the expanding-window mean of that bivariate density (Eq. 8);
  * the pipeline's own classic R-hat (rhat_max / rhat_mean over 2,000 (test dataset, parameter) pairs, blackjax) is
    read from the matching results/raw/final/ file where one exists.
Input: $PSCRATCH/sbi_final_draws/{model}_n{n}_r{r}_draws.npz (mu, s, corr: members x kept x n_test (x 2)), saved
in chain order (bde positions_eT_; lag-1 autocorrelation of mu 0.5-0.9 vs. 0.003 shuffled, checked 2026-10-29), and the
shared test set of data/reference_posterior/test_n100_ntest1000_seed0.npz. Draw files are overwritten when a sweep
job is re-run; each file's modification time is recorded.

Usage: python scripts/analysis/chainwise_diagnostics.py [glob]   (default: every final-sweep draw file)
Output: results/rbf_chainwise_diagnostics.npz (one row per file; env CHAINWISE_OUT overrides, e.g. for per-model shards
merged afterwards) and a printed table.
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # this repository's src/
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401
from rbf_sbi.diagnostics import rank_normal_rhat, split, eq9_stop, q, KAPPA, WINDOW, EPS, logpdf2   # noqa: E402

DRAWS = SCRATCH / "sbi_final_draws"
PATTERN = sys.argv[1] if len(sys.argv) > 1 else str(DRAWS / "final_*_n*_r?_draws.npz")
REF = DATA / "reference_posterior" / "test_n100_ntest1000_seed0.npz"
FINAL = RESULTS / "raw" / "final"
OUT = Path(os.environ.get("CHAINWISE_OUT", str(RESULTS / "rbf_chainwise_diagnostics.npz")))   # override: per-shard runs


def main():
    th = np.load(REF)["theta"].astype(np.float64)
    rows = []
    for f in sorted(glob.glob(PATTERN)):
        name = Path(f).name.replace("_draws.npz", "")
        model, rest = name.rsplit("_n", 1)
        n, r = int(rest.split("_r")[0]), int(rest.split("_r")[1])
        try:
            with np.load(f) as d:
                mu, s, c = (d[k].astype(np.float64) for k in ("mu", "s", "corr"))
        except Exception as e:   # partially written / unreadable: skip, say so
            print(f"skip {name}: {e!r}")
            continue
        M, S = mu.shape[:2]
        logp = logpdf2(th, mu, s, c)                                     # (M, S, n_test)
        flat = lambda x: x.reshape(M, S, -1)                             # (M, S, n_test*2) for mu, s
        rh = {}
        for qn, x in (("mu", flat(mu)), ("s", flat(s)), ("logp", logp)):
            rh[qn] = (rank_normal_rhat(split(x, 2)), [rank_normal_rhat(split(x[m:m + 1], KAPPA)) for m in range(M)])
        row = dict(model=model, n=n, repeat=r, members=M, kept=S, mtime=os.path.getmtime(f))
        for qn in rh:
            pq, mq = q(rh[qn][0]), [q(a) for a in rh[qn][1]]
            row.update({f"{qn}_pooled_med": pq["med"], f"{qn}_pooled_p95": pq["p95"], f"{qn}_pooled_f101": pq["f101"],
                        f"{qn}_pooled_f11": pq["f11"], f"{qn}_crhat_med": np.array([x["med"] for x in mq]),
                        f"{qn}_crhat_f101": np.array([x["f101"] for x in mq]), f"{qn}_crhat_f11": np.array([x["f11"] for x in mq])})
        # all three quantities pass at once, per test dataset (mu, s: worse of sigma / rho)
        per_ds = lambda a: a.reshape(-1, 2).max(1) if a.size == 2 * th.shape[0] else a
        row["all_pooled_f101"] = float(np.mean(np.maximum.reduce([per_ds(rh[k][0]) for k in rh]) <= 1.01))
        row["all_crhat_f101"] = np.array([float(np.mean(np.maximum.reduce([per_ds(rh[k][1][m]) for k in rh]) <= 1.01)) for m in range(M)])
        cum = np.logaddexp.accumulate(logp, axis=1) - np.log(np.arange(1, S + 1))[None, :, None]
        paths = cum.mean(2)                                               # (M, S): LPPD_l per member
        row.update(lppd_final=paths[:, -1], lppd_drift_last10=paths[:, -1] - paths[:, int(0.9 * S) - 1],
                   eq9_stop=np.array([eq9_stop(p) for p in paths]),
                   pooled_lppd=float(np.mean(logsumexp(logp.reshape(M * S, -1), axis=0) - np.log(M * S))),
                   between_mu=float(np.sqrt((mu.mean(1).var(0)).mean())), within_mu=float(np.sqrt((mu.var(1)).mean(0).mean())))
        fin = FINAL / f"{name}.npz"
        if fin.exists():
            with np.load(fin, allow_pickle=True) as d:
                row.update(pipe_rhat_max=float(d["rhat_max"]), pipe_rhat_mean=float(d["rhat_mean"]),
                           pipe_nll=-float(np.mean(d["log_prob_true"])))
        else:
            row.update(pipe_rhat_max=np.nan, pipe_rhat_mean=np.nan, pipe_nll=np.nan)
        rows.append(row)
        print(f"{name}: mu pooled {row['mu_pooled_med']:.3f} cw {row['mu_crhat_med'].mean():.3f} | s pooled {row['s_pooled_med']:.3f} "
              f"cw {row['s_crhat_med'].mean():.3f} | logp pooled {row['logp_pooled_med']:.3f} | test NLL {-row['pooled_lppd']:+.3f} "
              f"(saved {row['pipe_nll']:+.3f}) | pipeline Rhat max {row['pipe_rhat_max']:.2f}", flush=True)

    print(f"\n{len(rows)} fits. cR-hat: kappa={KAPPA} per member; pooled: members x 2 halves; Eq.9 window {WINDOW}, eps {EPS}")
    print("model | n | fits | R-hat median pooled / chain-wise, frac<=1.01 pooled / chain-wise: mu ; s ; logp | all pass pooled / cw | "
          "member LPPD spread | ensemble gain | max |drift last 10%| | pipeline Rhat max")
    for model in sorted({x["model"] for x in rows}):
        for n in sorted({x["n"] for x in rows if x["model"] == model}):
            g = [x for x in rows if x["model"] == model and x["n"] == n]
            m_ = lambda k: np.mean([x[k] for x in g]); c_ = lambda k: np.mean([np.mean(x[k]) for x in g])
            cell = lambda p: f"{m_(p + '_pooled_med'):.3f}/{c_(p + '_crhat_med'):.3f} {m_(p + '_pooled_f101'):.2f}/{c_(p + '_crhat_f101'):.2f}"
            spread = np.mean([x["lppd_final"].max() - x["lppd_final"].min() for x in g])
            gain = np.mean([x["pooled_lppd"] - x["lppd_final"].mean() for x in g])
            drift = np.max([np.abs(x["lppd_drift_last10"]).max() for x in g])
            pr = [x["pipe_rhat_max"] for x in g if np.isfinite(x["pipe_rhat_max"])]
            print(f"{model:24s} {n:6d} {len(g)} | {cell('mu')} ; {cell('s')} ; {cell('logp')} | {m_('all_pooled_f101'):.2f}/{c_('all_crhat_f101'):.2f} | "
                  f"{spread:.3f} | {gain:+.3f} | {drift:.4f} | {(f'{min(pr):.2f}-{max(pr):.2f}' if pr else '--')}")

    def stack(v):
        """Scalars -> (rows,); per-member arrays -> (rows, max members), NaN-padded where a fit kept fewer members
        (2026-10-30: five final-sweep fits kept 11 of 12 and np.array() of the ragged list failed). Readers of the
        per-member keys must use nan-aware reductions."""
        if all(np.ndim(x) == 0 for x in v):
            return np.array(v)
        L = max(len(x) for x in v)
        return np.array([np.pad(np.asarray(x, float), (0, L - len(x)), constant_values=np.nan) for x in v])

    keys = [k for k in rows[0] if k != "model"]
    np.savez_compressed(OUT, model=np.array([x["model"] for x in rows]), **{k: stack([x[k] for x in rows]) for k in keys},
                        kappa=KAPPA, eq9_window=WINDOW, eq9_eps=EPS)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
