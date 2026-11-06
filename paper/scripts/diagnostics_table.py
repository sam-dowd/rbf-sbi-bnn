"""Table for paper/paper.tex, Appendix H.2 (Table 21), from results/rbf_chainwise_diagnostics.npz
(scripts/analysis/chainwise_diagnostics.py on the per-draw files). 2026-11-06: switched to the FINAL runs (the
pretraining-tuned 12-member models, *_pretuned, repeats 0-4 as stored in the file); the earlier 12-member re-run's
rows (same model names without the suffix) are no longer tabulated. Analysis only; no experiment-log
entry. Per (model, n): number of fits; pipeline max R-hat (classic PSRF on mu, range over fits); pooled rank-normalized
R-hat median for mu / s / log p and the fraction of test datasets <= 1.01 for mu / s; chain-wise cR-hat(4) median for
mu / s (mean over members and fits); member LPPD spread (max - min over members), ensemble gain (pooled LPPD minus mean
member LPPD) and the largest |LPPD change over the last 10%| of any member chain. Per-member arrays are NaN-padded to
12 for the fits that kept 11 members, so every per-member statistic is NaN-aware.

Output: paper/tables/results5_chainwise.tex.   Usage: python paper/scripts/diagnostics_table.py
"""
from pathlib import Path

import numpy as np

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # this repository's src/
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401
IN = RESULTS / "rbf_chainwise_diagnostics.npz"
TAB = ROOT / "paper" / "tables"
LABEL = {"final_bnn_stdnormal_pretuned": r"$\mathcal{N}(0,1)$", "final_bnn_temp_pretuned": "tempered",
         "final_bnn_gaussprior_pretuned": "data-free width",
         "final_bnn_stdnormal_retune_pretuned": r"$\mathcal{N}(0,1)$ + re-tune", "final_bnn_ebblock_pretuned": "empirical Bayes"}


def main():
    z = np.load(IN, allow_pickle=True)
    model, n = z["model"], z["n"]
    num = lambda v: f"{int(v):,}".replace(",", "{,}")
    L = []
    for m in LABEL:
        for size in sorted(set(n[model == m])):
            i = np.where((model == m) & (n == size))[0]
            pr = z["pipe_rhat_max"][i]; pr = pr[np.isfinite(pr)]
            cw = lambda k: np.mean([np.nanmean(z[k][j]) for j in i])
            spread = np.mean([np.nanmax(z["lppd_final"][j]) - np.nanmin(z["lppd_final"][j]) for j in i])
            gain = np.mean([z["pooled_lppd"][j] - np.nanmean(z["lppd_final"][j]) for j in i])
            drift = np.max([np.nanmax(np.abs(z["lppd_drift_last10"][j])) for j in i])
            cells = [LABEL[m] if size == min(n[model == m]) else "", num(size), str(len(i)),
                     f"{pr.min():.2f}--{pr.max():.2f}" if pr.size else "--",
                     " / ".join(f"{z[k][i].mean():.3f}" for k in ("mu_pooled_med", "s_pooled_med", "logp_pooled_med")),
                     " / ".join(f"{z[k][i].mean():.2f}" for k in ("mu_pooled_f101", "s_pooled_f101")),
                     f"{cw('mu_crhat_med'):.3f} / {cw('s_crhat_med'):.3f}",
                     f"{spread:.3f}", f"{gain:+.3f}", f"{drift:.3f}"]
            L.append(" & ".join(cells) + " \\\\")
        L.append("\\midrule")
    (TAB / "results5_chainwise.tex").write_text("\n".join(L[:-1]) + "\n")
    print(f"saved {TAB / 'results5_chainwise.tex'}")


if __name__ == "__main__":
    main()
