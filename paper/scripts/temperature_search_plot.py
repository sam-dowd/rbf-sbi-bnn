"""Validation NLL vs temperature T from the tempered BNN's temperature search, for the bottom of
latex/understanding_results.tex. Analysis only (reads results/temperature_search_n{n}_stdprior_pretuned_grid.npz,
the search behind final_bnn_temp_pretuned's T = 0.7 / 0.7 / 0.6 / 0.7; fits nothing; no log entry).

What the files hold (scripts/tuning/optimize_temperature_grid.py): T grid 0.2..1.0 in steps of 0.1, searched by bisection on
the sign of the local slope with T = 1 always included, so only 5 of the 9 grid points were evaluated per size. Each
evaluated T is a 3-member fit_correlated run (N(0,1) prior, pretraining-tuned lr/wd) on the training sets of repeats
0-2, scored by NLL on the 1,000 pool rows following each repeat's training rows. Plotted: mean over the 3 repeats
(line), each repeat (grey dots), the chosen T (ring); a trial whose max split-R-hat on mu exceeds RHAT_BAD in any
repeat is drawn open (poorly mixed). Values above the panel's range are written at its top edge.

Output: paper/figures/understanding_results/temperature_search.pdf
Usage: python paper/scripts/temperature_search_plot.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # this repository's src/
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401
RES = RESULTS
FIG = ROOT / "paper" / "figures" / "understanding_results"
SIZES = [50, 100, 500, 1000]
RHAT_BAD = 1.5          # ASSUMPTION: display threshold for "poorly mixed", not a project gate
BNN = "#2a78d6"         # BNN entity colour, as in make_understanding_plots.py
INK, INK2, MUTED, GRID, DOT = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#a9a7a0"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 9, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.8,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def load(n):
    with np.load(RES / f"temperature_search_n{n}_stdprior_pretuned_grid.npz", allow_pickle=True) as d:
        assert bool(d["complete"]), n
        o = np.argsort(d["trial_temperatures"])
        return dict(T=d["trial_temperatures"][o], mean=d["trial_mean_nll"][o], rep=d["trial_per_repeat_nll"][o],
                    rhat=d["trial_per_repeat_rhat_max"][o], best=float(d["best_temperature"]),
                    t1=float(d["baseline_t1_val"]), members=int(d["n_members"]))


def panel(ax, n, s):
    bad = (s["rhat"] > RHAT_BAD).any(axis=1)
    good_vals = np.concatenate([s["rep"][~bad].ravel(), s["mean"][~bad]])
    lo, hi = good_vals.min(), good_vals.max()
    # Range covers every value (poorly mixed trials included) within 1.5 spans of the well-mixed range, so a
    # poorly mixed trial is never shown by its one good repeat alone; only catastrophic values go off-scale.
    all_vals = np.concatenate([s["rep"].ravel(), s["mean"]])
    shown = all_vals[all_vals <= hi + 1.5 * (hi - lo)]
    lo, hi = shown.min(), shown.max()
    pad = 0.12 * (hi - lo)
    ylo, yhi = lo - pad, hi + 2.2 * pad
    ax.set_ylim(ylo, yhi)
    over = []                       # (T, value) above the panel
    for T, r in zip(s["T"], s["rep"]):
        inside = r <= yhi
        ax.scatter(np.full(inside.sum(), T), r[inside], s=14, color=DOT, zorder=2, linewidths=0)
        over += [(T, v) for v in r[~inside]]
    m_in = s["mean"] <= yhi
    ax.plot(s["T"][m_in], s["mean"][m_in], color=BNN, zorder=3)
    for T, m, b in zip(s["T"], s["mean"], bad):
        if m > yhi:
            over.append((T, m))
            continue
        ax.scatter([T], [m], s=40, zorder=4, color="white" if b else BNN, edgecolors=BNN, linewidths=1.6)
    ib = int(np.argmin(np.abs(s["T"] - s["best"])))
    ax.scatter([s["best"]], [s["mean"][ib]], s=150, facecolors="none", edgecolors=INK, linewidths=1.0, zorder=5)
    # off-scale means: one label per T at the top edge, staggered so neighbouring labels do not collide
    # (off-scale individual repeats are not drawn; the caption says so)
    off = [(t, v) for t, v in over if t in s["T"][s["mean"] > yhi] and v in s["mean"]]
    for k, (T, m) in enumerate(sorted(off)):
        ax.annotate(f"↑ mean {m:+.0f}" if abs(m) >= 10 else f"↑ mean {m:+.2f}", (T, yhi),
                    xytext=(0, -2 - 9 * (k % 2)), textcoords="offset points", ha="center", va="top",
                    fontsize=6.5, color=INK2)
    gain = s["t1"] - s["mean"][ib]
    ax.set_title(f"$n={n:,}$: chosen $T={s['best']:.1f}$, gain over $T=1$: {gain:.2f} nats", loc="left")
    ax.set_xlim(0.15, 1.05)
    ax.set_xticks(np.round(np.arange(0.2, 1.01, 0.1), 1))
    ax.set_xticklabels([f"{t:.1f}" if i % 2 == 0 else "" for i, t in enumerate(np.arange(0.2, 1.01, 0.1))])
    ax.set_xlabel("temperature $T$")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for ax, n in zip(axes.flat, SIZES):
        s = load(n)
        bad = (s["rhat"] > RHAT_BAD).any(axis=1)
        print(f"n={n:5d} members={s['members']} chosen T={s['best']:.1f} T=1 {s['t1']:+.3f}")
        for T, m, r, rh, b in zip(s["T"], s["mean"], s["rep"], s["rhat"], bad):
            print(f"   T={T:.1f} mean {m:+9.3f}  repeats {np.array2string(r, precision=3)}  "
                  f"rhat_max {np.array2string(rh, precision=2)}{'  POORLY MIXED' if b else ''}")
        panel(ax, n, s)
    for ax in axes[:, 0]:
        ax.set_ylabel("validation NLL (nats; lower is better)")
    out = FIG / "temperature_search.pdf"
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
