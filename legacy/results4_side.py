"""Tables and figure for paper/paper.tex, Section "Results IV", subsection "Side by side".

Analysis only (existing sweep files; no fit, no experiment-log entry). 5 repeats, the hyperparameter composite of
Table 1 (`_hpnew` files at n = 50 / 500 / 1,000); every model paired with N(0,1) by repeat (same seed and training
subsample):
  N(0,1)            bnn_vanilla             stock MILE
  tempered          bnn_temp_T0p*           stock MILE, n <= 1,000 (T = 0.2 / 0.3 / 0.6 / 0.8)
  data-free width   bnn_gaussprior          stock MILE, N(0, 0.3655^2)
  Student-t         bnn_studentt            stock MILE, Student-t(4, 0.7071) (variance 1)
  empirical Bayes   bnn_ebblock_iso_10k     re-tune loop -- its own control is N(0,1) + re-tune (bnn_vanilla_retune_10k)
  NPE               npe
Tables: test NLL (mean +- sd; bold = lowest mean and every model whose paired difference from it is within its sd,
Section 5.3); width / warranted width and z-spread (sigma / rho; bold = closest to 1 by the same rule).
Figure: (a-f) paired test-NLL difference of each model from its control (N(0,1) + re-tune vs. N(0,1) included: the
re-tune alone is a sampler change, and the control for empirical Bayes).

Outputs: results/project_summary_results4_side.npz; paper/tables/results4_side_nll.tex, results4_side_calib.tex;
paper/figures/project_summary/results4/side_by_side.png.

Usage: python legacy/results4_side.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # this repository's src/
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling scripts in this folder
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401
from rbf_sbi import data                                     # noqa: E402
import old_runs_common as R1      # noqa: E402

SIZES = R1.SIZES
KEYS = {"N(0,1)": "bnn_vanilla", "data-free width": "bnn_gaussprior", "Student-t": "bnn_studentt",
        "N(0,1) + retune": "bnn_vanilla_retune_10k", "empirical Bayes": "bnn_ebblock_iso_10k", "NPE": "npe"}
TEMP = {50: "bnn_temp_T0p2", 100: "bnn_temp_T0p3", 500: "bnn_temp_T0p6", 1000: "bnn_temp_T0p8"}
COLS = ["N(0,1)", "tempered", "data-free width", "Student-t", "N(0,1) + retune", "empirical Bayes", "NPE"]   # table columns
CONTROL = {"tempered": "N(0,1)", "data-free width": "N(0,1)", "Student-t": "N(0,1)", "N(0,1) + retune": "N(0,1)",
           "empirical Bayes": "N(0,1) + retune", "NPE": "N(0,1)"}
# one series per panel, so each colour only has to separate from the surface. Sibling-figure hues (results4 scale /
# tempering); Student-t violet and NPE orange from the reference palette (a 6-line single panel failed the all-pairs
# CVD check -- red/orange, blue/violet -- hence the facets).
MCOL = {"N(0,1)": R1.COL["BNN"], "N(0,1) + retune": R1.COL["BNN"], "tempered": "#e34948", "data-free width": "#008300", "Student-t": "#6250d6",
        "empirical Bayes": "#eda100", "NPE": R1.COL["NPE"]}
FIG = ROOT / "paper" / "figures" / "project_summary" / "results4"
TAB = ROOT / "paper" / "tables"
OUT = RESULTS / "project_summary_results4_side.npz"
INK, INK2 = R1.INK, R1.INK2


def main():
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    ref = np.load(R1.REF); th = ref["theta"]
    sg, rg = np.asarray(data.SIGMA_GRID), np.asarray(data.RHO_GRID)
    ps = ref["p_sigma"] / ref["p_sigma"].sum(1, keepdims=True); pr = ref["p_rho"] / ref["p_rho"].sum(1, keepdims=True)
    mx = np.stack([(ps * sg).sum(1), (pr * rg).sum(1)], 1)
    sx = np.stack([np.sqrt((ps * (sg - mx[:, :1]) ** 2).sum(1)), np.sqrt((pr * (rg - mx[:, 1:]) ** 2).sum(1))], 1)
    exact_nll = -float(np.mean(np.asarray(data.compute_reference_log_prob_true(ref["post"], th))))

    def stats(d, key):
        mu, w = d["mean"], R1.width(d, key)
        z = (th - mu) / w
        return dict(nll=-float(np.mean(d["log_prob_true"])), warranted=np.sqrt((w ** 2).mean(0) / (sx ** 2 + (mu - mx) ** 2).mean(0)),
                    zstd=z.std(0), fm=(np.abs(mu - th) / np.maximum(np.abs(th), 1e-3)).mean(0))

    S = {}
    for lab, key in KEYS.items():
        for n in SIZES:
            S[(lab, n)] = [stats(np.load(R1.path(key, n, r), allow_pickle=True), key) for r in range(5)]
    for n, key in TEMP.items():
        S[("tempered", n)] = [stats(np.load(R1.path(key, n, r), allow_pickle=True), key) for r in range(5)]
    A = lambda lab, n, k: np.array([x[k] for x in S[(lab, n)]])
    have = lambda lab, n: (lab, n) in S

    print(f"exact posterior test NLL {exact_nll:.3f}")
    for n in SIZES:
        print(f"=== n={n}")
        for lab in COLS:
            if not have(lab, n):
                continue
            m = lambda k: A(lab, n, k).mean(0)
            line = (f"  {lab:16s} NLL {m('nll'):+.3f}±{A(lab, n, 'nll').std(ddof=1):.3f}  warranted {np.round(m('warranted'), 2)}  "
                    f"zstd {np.round(m('zstd'), 2)}  first-moment {np.round(m('fm'), 3)}")
            if lab in CONTROL:
                d = A(lab, n, "nll") - A(CONTROL[lab], n, "nll")
                line += f"  | vs {CONTROL[lab]}: {d.mean():+.3f}±{d.std(ddof=1):.3f} ({(d < 0).sum()}/5)"
            print(line)
    np.savez_compressed(OUT, exact_nll=exact_nll, **{f"{lab}|{n}|{k}": A(lab, n, k) for (lab, n) in S for k in S[(lab, n)][0]})

    # ---- tables (Section 5.3 bolding: best mean + every model whose paired difference from it is within its sd) ----
    bf = lambda t, on: (f"{{\\boldmath{t}}}" if t.startswith("$") else f"\\textbf{{{t}}}") if on else t
    num = lambda n: f"{n:,}".replace(",", "{,}")

    def best_set(n, crit):
        labs = [l for l in COLS if have(l, n)]
        v = {l: crit(l) for l in labs}
        b = min(labs, key=lambda l: v[l].mean())
        return {l for l in labs if l == b or abs((v[l] - v[b]).mean()) <= (v[l] - v[b]).std(ddof=1)}

    L = []
    for n in SIZES:
        win = best_set(n, lambda l: A(l, n, "nll"))
        cells = [num(n)] + [bf(f"${A(l, n, 'nll').mean():+.3f}\\pm{A(l, n, 'nll').std(ddof=1):.3f}$", l in win) if have(l, n) else "--"
                            for l in COLS]
        L.append(" & ".join(cells) + " \\\\")
    (TAB / "results4_side_nll.tex").write_text("\n".join(L) + "\n")

    L = []
    for k, name in (("warranted", "width / warranted"), ("zstd", "z-spread")):
        L.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{name}}} ($\\sigma$/$\\rho$)}} \\\\")
        for n in SIZES:
            cells = [num(n)]
            wins = [best_set(n, lambda l, j=j: np.abs(A(l, n, k)[:, j] - 1)) for j in range(2)]
            for l in COLS:
                cells.append("/".join(bf(f"{A(l, n, k).mean(0)[j]:.2f}", l in wins[j]) for j in range(2)) if have(l, n) else "--")
            L.append(" & ".join(cells) + " \\\\")
        if k == "warranted":
            L.append("\\midrule")
    (TAB / "results4_side_calib.tex").write_text("\n".join(L) + "\n")

    # ---- figure: 2x3 facets, shared y, one model minus its control per panel ----
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.42, wspace=0.12))
    panels = [(l, CONTROL[l]) for l in ("tempered", "data-free width", "Student-t", "N(0,1) + retune", "empirical Bayes", "NPE")]
    for i, (lab, ctl) in enumerate(panels):
        ax = axes.flat[i]
        ns = [n for n in SIZES if have(lab, n)]
        v = np.array([A(lab, n, "nll") - A(ctl, n, "nll") for n in ns])
        mu, sd = v.mean(1), v.std(1, ddof=1)
        ax.axhline(0, color=INK, lw=1.0)
        ax.fill_between(ns, mu - sd, mu + sd, color=MCOL[lab], alpha=0.15, linewidth=0)
        ax.plot(ns, mu, color=MCOL[lab], marker="o", ms=4, lw=2)
        ax.set_title(f"({'abcdef'[i]}) {lab} $-$ {ctl}", fontsize=8.5)
        ax.set_xscale("log"); ax.tick_params(labelsize=7.5)
        if i >= 3:
            ax.set_xlabel("$n$")
        if i % 3 == 0:
            ax.set_ylabel("test NLL difference", fontsize=8)
    fig.savefig(FIG / "side_by_side.png"); plt.close(fig)
    print(f"saved {OUT}, {TAB / 'results4_side_nll.tex'}, {TAB / 'results4_side_calib.tex'}, {FIG / 'side_by_side.png'}")


if __name__ == "__main__":
    main()
