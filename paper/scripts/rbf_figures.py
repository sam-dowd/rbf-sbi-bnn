"""Figures and summary numbers for paper/paper.tex, Section "The RBF-kernel inference task".

Analysis only: reads the canonical dataset (data/dataset.npz) and the cached exact grid posterior of its
1,000 test datasets (data/reference_posterior/test_n100_ntest1000_seed0.npz); fits nothing, so it writes
no experiment-log entry. Prints every number the section quotes.

Figures (paper/figures/project_summary/rbf_task/):
  simulator.png            three parameter settings: N=10 draws of the 10-point process, and the summary S
                           (draws re-simulated here from the generative model at those theta -- the dataset
                           stores only S -- so they are illustrative, not the stored datasets)
  reference_posteriors.png exact posterior of four test datasets, with the prior and the true theta
  reference_summary.png    across the 1,000 test datasets: posterior std of sigma and rho, posterior correlation

Usage: python paper/scripts/rbf_figures.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # this repository's src/
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401
from rbf_sbi import data  # noqa: E402

OUT = ROOT / "paper" / "figures" / "project_summary" / "rbf_task"
REF = DATA / "reference_posterior" / "test_n100_ntest1000_seed0.npz"

# Reference palette (dataviz skill, references/palette.md): categorical slots 1-3, blue ramp, ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                                      "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 9, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.5,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def kernel(sigma, rho):
    T = np.asarray(data.T)
    return sigma ** 2 * np.exp(-(T[:, None] - T[None, :]) ** 2 / (2 * rho ** 2)) + data.ETA ** 2 * np.eye(data.D)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, X_test, theta = data.load_dataset()
    ref = np.load(REF)
    assert np.allclose(ref["X"], X_test) and np.allclose(ref["theta"], theta), "reference cache is for another test set"
    post, ps, pr = ref["post"], ref["p_sigma"], ref["p_rho"]
    sg, rg = np.asarray(data.SIGMA_GRID), np.asarray(data.RHO_GRID)
    ps, pr = ps / ps.sum(1, keepdims=True), pr / pr.sum(1, keepdims=True)
    m = np.stack([(ps * sg).sum(1), (pr * rg).sum(1)], 1)
    sd = np.stack([np.sqrt((ps * (sg - m[:, :1]) ** 2).sum(1)), np.sqrt((pr * (rg - m[:, 1:]) ** 2).sum(1))], 1)
    P = post / post.sum((1, 2), keepdims=True)
    cov = (P * (sg[None, :, None] - m[:, 0, None, None]) * (rg[None, None, :] - m[:, 1, None, None])).sum((1, 2))
    corr = cov / (sd[:, 0] * sd[:, 1])
    z = (theta - m) / sd
    # skewness of each marginal (non-Gaussianity)
    skew = np.stack([(ps * ((sg - m[:, :1]) / sd[:, :1]) ** 3).sum(1), (pr * ((rg - m[:, 1:]) / sd[:, 1:]) ** 3).sum(1)], 1)
    edge = np.maximum.reduce([ps[:, 0], ps[:, -1], pr[:, 0], pr[:, -1]])

    print("=== numbers for the text ===")
    print(f"grid {len(sg)} x {len(rg)}: sigma [{sg[0]}, {sg[-1]}] step {sg[1]-sg[0]:.4f}; rho [{rg[0]}, {rg[-1]}] step {rg[1]-rg[0]:.4f}")
    print(f"posterior std sigma: RMS {np.sqrt((sd[:,0]**2).mean()):.3f}, median {np.median(sd[:,0]):.3f}, 5-95% [{np.percentile(sd[:,0],5):.3f}, {np.percentile(sd[:,0],95):.3f}]")
    print(f"posterior std rho:   RMS {np.sqrt((sd[:,1]**2).mean()):.3f}, median {np.median(sd[:,1]):.3f}, 5-95% [{np.percentile(sd[:,1],5):.3f}, {np.percentile(sd[:,1],95):.3f}]")
    print(f"contraction 1 - post var / prior var (median): sigma {np.median(1 - sd[:,0]**2/data.SIGMA_STD**2):.3f}, rho {np.median(1 - sd[:,1]**2/data.RHO_STD**2):.3f}")
    print(f"grid cells per posterior std (min): sigma {sd[:,0].min()/(sg[1]-sg[0]):.1f}, rho {sd[:,1].min()/(rg[1]-rg[0]):.1f}")
    print(f"posterior corr: median {np.median(corr):.3f}, 5-95% [{np.percentile(corr,5):.3f}, {np.percentile(corr,95):.3f}], median |corr| {np.median(np.abs(corr)):.3f}")
    print(f"marginal skewness |median|: sigma {np.median(skew[:,0]):.3f}, rho {np.median(skew[:,1]):.3f}; 95th pct |skew| {np.percentile(np.abs(skew[:,0]),95):.3f} / {np.percentile(np.abs(skew[:,1]),95):.3f}")
    print(f"corr(posterior std sigma, true sigma) {np.corrcoef(sd[:,0], theta[:,0])[0,1]:.3f}; corr(posterior std rho, true rho) {np.corrcoef(sd[:,1], theta[:,1])[0,1]:.3f}")
    print(f"calibration on the test set: std of z {z.std(0).round(3)}, frac |z|<1 {np.mean(np.abs(z)<1,0).round(3)}, frac |z|<2 {np.mean(np.abs(z)<2,0).round(3)}")
    print(f"max boundary marginal mass over test set {edge.max():.2e}")
    from scipy.stats import norm
    print(f"prior mass removed by truncation at 0: sigma {norm.cdf(-data.SIGMA_MU/data.SIGMA_STD):.1e}, rho {norm.cdf(-data.RHO_MU/data.RHO_STD):.1e}")

    # ---- Figure 1: simulator ----
    rng = np.random.default_rng(0)
    settings = [(1.25, 0.25), (1.25, 0.60), (1.25, 1.00)]
    T = np.asarray(data.T)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), gridspec_kw=dict(height_ratios=[1, 1.15]))
    vmax = max(np.abs(kernel(s, r)).max() for s, r in settings) * 1.3
    for j, (s, r) in enumerate(settings):
        X = rng.multivariate_normal(np.zeros(data.D), kernel(s, r), size=data.N)
        S = X.T @ X / data.N
        ax = axes[0, j]
        for x in X:
            ax.plot(T, x, color=BLUE, lw=0.9, alpha=0.75, marker="o", ms=2.5)
        ax.set_title(rf"$\sigma={s}$, $\rho={r}$")
        ax.set_xlabel("$t$"); ax.set_ylim(-4.2, 4.2)
        if j == 0: ax.set_ylabel("$x_i(t)$")
        ax = axes[1, j]; ax.grid(False)
        im = ax.imshow(S, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks([0, 9]); ax.set_yticks([0, 9]); ax.set_xticklabels(["1", "10"]); ax.set_yticklabels(["1", "10"])
        if j == 0: ax.set_ylabel(r"$S=X^\top X/N$")
    fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.85, pad=0.02)
    fig.savefig(OUT / "simulator.png"); plt.close(fig)

    # ---- Figure 2: four reference posteriors ----
    picks = {
        "median width": int(np.argmin(np.abs(sd[:, 0] * sd[:, 1] - np.median(sd[:, 0] * sd[:, 1])))),
        "widest": int(np.argmax(sd[:, 0] * sd[:, 1])),
        r"strongest $|\mathrm{corr}|$": int(np.argmax(np.abs(corr))),
        "narrowest": int(np.argmin(sd[:, 0] * sd[:, 1])),
    }
    SG, RG = np.meshgrid(sg, rg, indexing="ij")
    prior = np.exp(-0.5 * ((SG - data.SIGMA_MU) / data.SIGMA_STD) ** 2 - 0.5 * ((RG - data.RHO_MU) / data.RHO_STD) ** 2)
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.2), sharey=True)
    for ax, (lab, i) in zip(axes, picks.items()):
        ax.grid(False)
        ax.contourf(SG, RG, P[i] / P[i].max(), levels=np.linspace(0.02, 1, 12), cmap=SEQ)
        ax.contour(SG, RG, prior, levels=[np.exp(-2), np.exp(-0.5)], colors=MUTED, linewidths=0.8, linestyles="--")
        ax.plot(*theta[i], marker="x", color=ORANGE, ms=7, mew=2)
        ax.set_title(f"{lab}\ncorr {corr[i]:+.2f}", fontsize=8)
        ax.set_xlim(0.3, 2.5); ax.set_ylim(0.02, 1.4); ax.set_xlabel(r"$\sigma$")
        print(f"fig2 panel '{lab}': test index {i}, theta {theta[i].round(3)}, post mean {m[i].round(3)}, post std {sd[i].round(3)}, corr {corr[i]:+.3f}")
    axes[0].set_ylabel(r"$\rho$")
    fig.savefig(OUT / "reference_posteriors.png"); plt.close(fig)

    # ---- Figure 3: posterior widths and correlation across the test set ----
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.1))
    for ax, x, prior_sd, lab in ((axes[0], sd[:, 0], data.SIGMA_STD, r"posterior std of $\sigma$"),
                                 (axes[1], sd[:, 1], data.RHO_STD, r"posterior std of $\rho$")):
        ax.hist(x, bins=40, color=BLUE, edgecolor="white", linewidth=0.5)
        ax.axvline(prior_sd, color=INK2, lw=1, ls="--"); ax.text(prior_sd, ax.get_ylim()[1] * 0.92, "prior ", color=INK2, fontsize=7, ha="right")
        ax.set_xlabel(lab); ax.set_xlim(0, prior_sd * 1.1)
    axes[0].set_ylabel("test datasets")
    axes[2].hist(corr, bins=40, color=BLUE, edgecolor="white", linewidth=0.5)
    axes[2].set_xlabel(r"posterior corr$(\sigma,\rho)$")
    fig.savefig(OUT / "reference_summary.png"); plt.close(fig)
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
