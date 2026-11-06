"""Numbers and figure for paper/paper.tex, Section "Results I: BNN versus NPE across training-set size".

Analysis only (reads existing sweep files, fits nothing, no experiment-log entry). Models: NPE (`npe`), Gaussian
predictor (`gaussian`), BNN with N(0,1) prior on stock MILE (`bnn_vanilla`; the `_hpnew` file where it exists,
i.e. n = 50 / 500 / 1,000 -- the hyperparameter composite of Table 1). 3-member BNN runs, 5 repeats per size.
Reference: the cached exact posterior of the shared test set.

Prints, per model and size (mean +- sample std over repeats): test NLL, first-moment error, width error, TARP
calibration error and coverage AUC; and paired BNN-minus-baseline NLL differences. Saves
results/project_summary_results1.npz, paper/figures/project_summary/results1/vs_n.png, and the two tables as
paper/tables/results1_nll.tex and paper/tables/results1_moments.tex (\input by paper.tex).

Bolding rule (lower-is-better metrics, per size and per parameter): the method with the lowest mean, and every
method whose paired per-repeat difference from it does not exceed its own sample sd (|mean| <= sd), i.e. the
Section 5.3 rule for "not a finding". Not applied to paired-difference columns or to the signed TARP AUC.

Usage: python legacy/old_runs_common.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # this repository's src/
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401
from rbf_sbi import data      # noqa: E402
from rbf_sbi import metrics   # noqa: E402

SIZES = [50, 100, 500, 1000, 5000, 10000]
REPEATS = range(5)
MODELS = {"BNN": "bnn_vanilla", "NPE": "npe", "Gaussian predictor": "gaussian"}
REF = DATA / "reference_posterior" / "test_n100_ntest1000_seed0.npz"
FIG = ROOT / "paper" / "figures" / "project_summary" / "results1"
TAB = ROOT / "paper" / "tables"
OUT = RESULTS / "project_summary_results1.npz"
COL = {"BNN": "#2a78d6", "NPE": "#eb6834", "Gaussian predictor": "#1baf7a"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 9, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.8,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white", "legend.frameon": False,
})


def path(model, n, r):
    for c in (RESULTS / f"raw/sweep_{model}_hpnew_n{n}_r{r}.npz", RESULTS / f"raw/sweep_{model}_n{n}_r{r}.npz"):
        if c.exists():
            return c
    raise FileNotFoundError(f"{model} n={n} r={r}")


def width(d, key):
    if key == "npe":
        return d["std"]
    if key == "gaussian":
        return d["aleatoric"]
    return np.sqrt(d["aleatoric"] ** 2 + d["epistemic"] ** 2)


def tarp_figures(M):
    """(1) ECP(alpha) curves per n (repeats pooled; band: +-1 bootstrap sd over test datasets, metrics.tarp_curve_from_f);
    (2) summary: TARP calibration error and coverage AUC vs n (bands: +-1 sd over repeats). Exact posterior: 0.005 / +0.005
    (Table of exact-posterior values, Section 5.1)."""
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.35, wspace=0.12))
    for ax, n in zip(axes.ravel(), SIZES):
        ax.plot([0, 1], [0, 1], color=INK, ls="--", lw=1.0)
        for lab, key in MODELS.items():
            fs = [np.load(path(key, n, r), allow_pickle=True)["f"] for r in REPEATS]
            a, ecp, bs = metrics.tarp_curve_from_f(fs)
            ax.plot(a, ecp, color=COL[lab], lw=1.6)
            ax.fill_between(a, ecp - bs, ecp + bs, color=COL[lab], alpha=0.2, linewidth=0)
        ax.set_title(f"$n={n:,}$".replace(",", "{,}"), fontsize=8.5); ax.set_aspect("equal"); ax.tick_params(labelsize=7.5)
    for ax in axes[1]:
        ax.set_xlabel(r"credibility level $\alpha$", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("expected coverage", fontsize=8)
    fig.legend(handles=[Line2D([], [], color=COL[l], lw=2, label=l) for l in MODELS] +
                       [Line2D([], [], color=INK, ls="--", lw=1, label="calibrated")],
               loc="upper center", bbox_to_anchor=(0.5, 0.01), ncol=4, fontsize=8)
    fig.savefig(FIG / "tarp_curves.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), gridspec_kw=dict(wspace=0.32))
    for ax, c_i, title, ref in ((axes[0], 5, "(a) TARP calibration error", 0.005), (axes[1], 6, "(b) TARP coverage AUC", 0.005)):
        for lab in MODELS:
            v = np.array([M[(lab, n)][:, c_i] for n in SIZES]); mu, sd = v.mean(1), v.std(1, ddof=1)
            ax.plot(SIZES, mu, color=COL[lab], marker="o", ms=3.5)
            ax.fill_between(SIZES, mu - sd, mu + sd, color=COL[lab], alpha=0.15, linewidth=0)
        ax.axhline(ref, color=INK, ls="--", lw=1.1)
        if c_i == 6:
            ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xscale("log"); ax.set_xlabel("$n$"); ax.set_title(title, fontsize=8.5); ax.tick_params(labelsize=7.5)
    fig.legend(handles=[Line2D([], [], color=COL[l], marker="o", lw=1.8, label=l) for l in MODELS] +
                       [Line2D([], [], color=INK, ls="--", lw=1.1, label="exact posterior")],
               loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=8)
    fig.savefig(FIG / "tarp_summary.png"); plt.close(fig)


def winners(M, n, c_i):
    """Set of model labels to bold for metric column c_i at size n (module docstring rule)."""
    labs = list(MODELS)
    best = min(labs, key=lambda l: M[(l, n)][:, c_i].mean())
    out = {best}
    for l in labs:
        if l == best:
            continue
        d = M[(l, n)][:, c_i] - M[(best, n)][:, c_i]
        if abs(d.mean()) <= d.std(ddof=1):
            out.add(l)
    return out


def fmt(x, nd=3, sign=False):
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$") if not sign else s.replace("-", "$-$").replace("+", "$+$")


def write_tables(M, exact_nll, exact_fm):
    TAB.mkdir(parents=True, exist_ok=True)
    labs = list(MODELS)
    num = lambda n: f"{n:,}".replace(",", "{,}")
    b = lambda txt, on: (f"{{\\boldmath{txt}}}" if txt.startswith("$") else f"\\textbf{{{txt}}}") if on else txt
    # NLL table
    L = []
    for n in SIZES:
        w = winners(M, n, 0)
        cells = [b(f"${M[(l, n)][:, 0].mean():+.3f}\\pm{M[(l, n)][:, 0].std(ddof=1):.3f}$".replace("+", "+"), l in w) for l in labs]
        diffs = []
        for base in ("NPE", "Gaussian predictor"):
            d = M[("BNN", n)][:, 0] - M[(base, n)][:, 0]
            diffs.append(f"${d.mean():+.3f}\\pm{d.std(ddof=1):.3f}$ ({(d < 0).sum()}/{len(d)})")
        L.append(f"{num(n)} & " + " & ".join(cells) + " & " + " & ".join(diffs) + " \\\\")
    (TAB / "results1_nll.tex").write_text("\n".join(L) + "\n")
    # moments table: first-moment sigma/rho, width sigma/rho, TARP calibration error (AUC)
    L = []
    for n in SIZES:
        row = [num(n)]
        for c_pair in ((1, 2), (3, 4)):
            for l in labs:
                parts = []
                for c_i in c_pair:
                    v = M[(l, n)][:, c_i].mean()
                    txt = f"{v:.3f}".lstrip("0") if c_i in (1, 2) else f"{v:.2f}"
                    parts.append(b(txt, l in winners(M, n, c_i)))
                row.append("/".join(parts))
        w = winners(M, n, 5)
        for l in labs:
            ce, auc = M[(l, n)][:, 5].mean(), M[(l, n)][:, 6].mean()
            row.append(b(f"{ce:.3f}".lstrip("0"), l in w) + f" ({auc:+.3f})".replace("-", "$-$").replace("0.", "."))
        L.append(" & ".join(row) + " \\\\")
    (TAB / "results1_moments.tex").write_text("\n".join(L) + "\n")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    ref = np.load(REF); th = ref["theta"]
    sg, rg = np.asarray(data.SIGMA_GRID), np.asarray(data.RHO_GRID)
    ps = ref["p_sigma"] / ref["p_sigma"].sum(1, keepdims=True); pr = ref["p_rho"] / ref["p_rho"].sum(1, keepdims=True)
    mx = np.stack([(ps * sg).sum(1), (pr * rg).sum(1)], 1)
    sx = np.stack([np.sqrt((ps * (sg - mx[:, :1]) ** 2).sum(1)), np.sqrt((pr * (rg - mx[:, 1:]) ** 2).sum(1))], 1)
    exact_nll = -float(np.mean(np.asarray(data.compute_reference_log_prob_true(ref["post"], th))))
    exact_fm = (np.abs(mx - th) / np.maximum(np.abs(th), 1e-3)).mean(0)

    M = {}   # (label, n) -> dict of per-repeat arrays
    for lab, key in MODELS.items():
        for n in SIZES:
            rows = []
            for r in REPEATS:
                d = np.load(path(key, n, r), allow_pickle=True)
                assert np.allclose(d["theta_test"], th)
                w = width(d, key)
                a, e, _ = metrics.tarp_curve_from_f([d["f"]])
                rows.append([-d["log_prob_true"].mean(),
                             *(np.abs(d["mean"] - th) / np.maximum(np.abs(th), 1e-3)).mean(0),
                             *(np.abs(w - sx) / sx).mean(0),
                             np.mean(np.abs(e - a)), np.trapezoid(e - a, a)])
            M[(lab, n)] = np.array(rows)   # cols: nll, fm_s, fm_r, we_s, we_r, tarp_ce, tarp_auc
    cols = ["nll", "fm_sigma", "fm_rho", "we_sigma", "we_rho", "tarp_ce", "tarp_auc"]

    print(f"exact posterior: test NLL {exact_nll:.3f}; first-moment error {exact_fm.round(3)}")
    for c_i, c in enumerate(cols):
        print(f"\n--- {c} (mean +- std over 5 repeats)")
        for n in SIZES:
            print(f"  n={n:6d}  " + "   ".join(f"{lab}: {M[(lab, n)][:, c_i].mean():+.3f} ± {M[(lab, n)][:, c_i].std(ddof=1):.3f}"
                                             for lab in MODELS))
    print("\n--- paired test-NLL difference BNN minus baseline (mean ± sd, #repeats BNN better)")
    for n in SIZES:
        out = []
        for base in ("NPE", "Gaussian predictor"):
            dd = M[("BNN", n)][:, 0] - M[(base, n)][:, 0]
            out.append(f"vs {base}: {dd.mean():+.3f} ± {dd.std(ddof=1):.3f} ({(dd < 0).sum()}/5)")
        print(f"  n={n:6d}  " + "   ".join(out))
    np.savez_compressed(OUT, sizes=np.array(SIZES), models=np.array(list(MODELS)), columns=np.array(cols),
                        values=np.stack([np.stack([M[(lab, n)] for n in SIZES]) for lab in MODELS]),   # (model, size, repeat, col)
                        exact_nll=exact_nll, exact_first_moment=exact_fm)

    # ---- figure: NLL (top, wide) and first-moment / width error for sigma and rho (bottom) ----
    fig = plt.figure(figsize=(7.4, 5.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.55, wspace=0.38)
    panels = [(fig.add_subplot(gs[0, :]), 0, "test NLL (nats)", None)]
    for k, (c_i, ylab, par) in enumerate([(1, "first-moment error", r"$\sigma$"), (2, "first-moment error", r"$\rho$"),
                                          (3, "width error", r"$\sigma$"), (4, "width error", r"$\rho$")]):
        panels.append((fig.add_subplot(gs[1, k]), c_i, ylab, par))
    for ax, c_i, ylab, par in panels:
        for lab in MODELS:
            v = np.array([M[(lab, n)][:, c_i] for n in SIZES]); mu, sd = v.mean(1), v.std(1, ddof=1)
            ax.plot(SIZES, mu, color=COL[lab], marker="o", ms=4, label=lab)
            ax.fill_between(SIZES, mu - sd, mu + sd, color=COL[lab], alpha=0.15, linewidth=0)
        if c_i == 0:
            ax.axhline(exact_nll, color=INK, ls="--", lw=1.2, label="exact posterior")
            ax.legend(loc="upper right", ncol=4, fontsize=8)
        elif c_i in (1, 2):
            ax.axhline(exact_fm[c_i - 1], color=INK, ls="--", lw=1.2)
        if c_i in (3, 4):
            ax.set_yscale("log")
        ax.set_xscale("log"); ax.set_xlabel("$n$")
        if par is None:
            ax.set_ylabel(ylab)
        else:   # bottom row: label as a title above the panel, so it cannot run into the neighbouring panel
            ax.set_title(f"{ylab}, {par}", fontsize=8.5)
            ax.tick_params(axis="both", labelsize=7.5)
    fig.savefig(FIG / "vs_n.png"); plt.close(fig)
    write_tables(M, exact_nll, exact_fm)
    tarp_figures(M)
    print(f"\nsaved {OUT}, {FIG / 'vs_n.png'} and {TAB}/results1_*.tex")


if __name__ == "__main__":
    main()
