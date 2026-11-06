"""Numbers, table and figure for paper/paper.tex, Section "Results IV", subsection "Prior scale".

Analysis only (existing sweep files; no fit, no experiment-log entry). 3-member BNNs, 5 repeats, the hyperparameter
composite of Table 1 (`_hpnew` files at n = 50 / 500 / 1,000):
  N(0,1)             bnn_vanilla             stock MILE, prior N(0,1)
  data-free width    bnn_gaussprior          stock MILE, prior N(0, 0.3655^2) (scripts/tuning/optimize_prior_width.py)
  N(0,1) + retune    bnn_vanilla_retune_10k  retune loop, prior N(0,1)          -- control for empirical Bayes
  empirical Bayes    bnn_ebblock_iso_10k     retune loop, one prior scale for all weights, block update between retunes
  tempered           bnn_temp_T0p*           (reference only; n <= 1,000)
Paired differences: data-free width minus N(0,1); empirical Bayes minus N(0,1)+retune.
Per (model, n, repeat): test NLL, width / warranted width (RMS ratio, 2026-10-29), z-spread, RMS aleatoric / epistemic, RMS error of the mean
(vs. exact posterior mean), TARP calibration error, R-hat max; for empirical Bayes the final prior scale per member and
the number of members at the 1e-3 floor (< 0.01).

Outputs: results/project_summary_results4_scale.npz; paper/tables/results4_scale.tex;
paper/figures/project_summary/results4/scale_decomposition.png.

Usage: python legacy/results4_scale.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # this repository's src/
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling scripts in this folder
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401
from rbf_sbi import data                                     # noqa: E402
from rbf_sbi import metrics                                  # noqa: E402
import old_runs_common as R1      # noqa: E402

SIZES = R1.SIZES
MODELS = {"N(0,1)": "bnn_vanilla", "data-free width": "bnn_gaussprior",
          "N(0,1) + retune": "bnn_vanilla_retune_10k", "empirical Bayes": "bnn_ebblock_iso_10k"}
TEMP = {50: "bnn_temp_T0p2", 100: "bnn_temp_T0p3", 500: "bnn_temp_T0p6", 1000: "bnn_temp_T0p8"}
PAIRS = (("data-free width", "N(0,1)"), ("empirical Bayes", "N(0,1) + retune"))
# colours: N(0,1) blue, tempered red, data-free green, empirical Bayes yellow (validate_palette.js light: passes;
# green/red protan Delta-E 7.2 -> distinct markers as secondary encoding)
MCOL = {"N(0,1)": R1.COL["BNN"], "tempered": "#e34948", "data-free width": "#008300", "empirical Bayes": "#eda100"}
MMK = {"N(0,1)": "o", "tempered": "D", "data-free width": "s", "empirical Bayes": "^"}
FIG = ROOT / "paper" / "figures" / "project_summary" / "results4"
TAB = ROOT / "paper" / "tables"
OUT = RESULTS / "project_summary_results4_scale.npz"
INK, INK2 = R1.INK, R1.INK2


def main():
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    ref = np.load(R1.REF); th = ref["theta"]
    sg, rg = np.asarray(data.SIGMA_GRID), np.asarray(data.RHO_GRID)
    ps = ref["p_sigma"] / ref["p_sigma"].sum(1, keepdims=True); pr = ref["p_rho"] / ref["p_rho"].sum(1, keepdims=True)
    mx = np.stack([(ps * sg).sum(1), (pr * rg).sum(1)], 1)
    sx = np.stack([np.sqrt((ps * (sg - mx[:, :1]) ** 2).sum(1)), np.sqrt((pr * (rg - mx[:, 1:]) ** 2).sum(1))], 1)
    exact_rms = np.sqrt((sx ** 2).mean(0))

    def stats(d):
        mu, al, ep = d["mean"], d["aleatoric"], d["epistemic"]
        w = np.sqrt(al ** 2 + ep ** 2); z = (th - mu) / w
        a, e, _ = metrics.tarp_curve_from_f([d["f"]])
        out = dict(nll=-float(np.mean(d["log_prob_true"])), warranted=np.sqrt((w ** 2).mean(0) / (sx ** 2 + (mu - mx) ** 2).mean(0)),
                   zstd=z.std(0), rms_al=np.sqrt((al ** 2).mean(0)), rms_ep=np.sqrt((ep ** 2).mean(0)),
                   rms_err=np.sqrt(((mu - mx) ** 2).mean(0)), tarp_ce=float(np.mean(np.abs(e - a))), rhat_max=float(d["rhat_max"]))
        if "rasmussen_ebblock_tau_history" in d.files:
            tau = np.asarray(d["rasmussen_ebblock_tau_history"])[:, -1, 0]
            out.update(eb_tau=tau, eb_floor=int(np.sum(tau < 0.01)))
        return out

    S = {}
    for lab, key in MODELS.items():
        for n in SIZES:
            S[(lab, n)] = [stats(np.load(R1.path(key, n, r), allow_pickle=True)) for r in range(5)]
    for n, key in TEMP.items():
        S[("tempered", n)] = [stats(np.load(R1.path(key, n, r), allow_pickle=True)) for r in range(5)]
    A = lambda lab, n, k: np.array([x[k] for x in S[(lab, n)]])

    for n in SIZES:
        print(f"=== n={n}")
        for lab in list(MODELS) + (["tempered"] if n in TEMP else []):
            m = lambda k: A(lab, n, k).mean(0)
            print(f"  {lab:16s} NLL {m('nll'):+.3f}  warranted {np.round(m('warranted'), 2)}  zstd {np.round(m('zstd'), 2)}  "
                  f"RMS al {np.round(m('rms_al'), 3)} ep {np.round(m('rms_ep'), 3)} err {np.round(m('rms_err'), 3)}  "
                  f"TARP {m('tarp_ce'):.3f}  Rhat {A(lab, n, 'rhat_max').min():.2f}-{A(lab, n, 'rhat_max').max():.2f}")
        for a_, b_ in PAIRS:
            d = A(a_, n, "nll") - A(b_, n, "nll")
            print(f"    {a_} - {b_}: NLL {d.mean():+.3f} ± {d.std(ddof=1):.3f} ({(d < 0).sum()}/5)")
        tau = np.concatenate([x["eb_tau"] for x in S[("empirical Bayes", n)]])
        print(f"    EB final scale over 15 members: mean {tau.mean():.3f} median {np.median(tau):.3f} range {tau.min():.3f}-{tau.max():.3f}; "
              f"at floor (<0.01) {int((tau < 0.01).sum())}/15")
        if n in TEMP:
            d = A("data-free width", n, "nll") - A("tempered", n, "nll")
            print(f"    data-free width - tempered (both stock MILE): NLL {d.mean():+.3f} ± {d.std(ddof=1):.3f} ({(d < 0).sum()}/5)")
    np.savez_compressed(OUT, **{f"{lab}|{n}|{k}": A(lab, n, k) for (lab, n) in S for k in S[(lab, n)][0] if k != "eb_tau"},
                        **{f"empirical Bayes|{n}|eb_tau": np.concatenate([x["eb_tau"] for x in S[("empirical Bayes", n)]]) for n in SIZES})

    # ---- table: per n, pairs "control -> variant" with the better bold (Section 5.3 rule), EB scale, R-hat ----
    CRIT = {"nll": lambda lab, n, j: A(lab, n, "nll"),
            "warranted": lambda lab, n, j: np.abs(A(lab, n, "warranted")[:, j] - 1),
            "rms_al": lambda lab, n, j: np.abs(A(lab, n, "rms_al")[:, j] - exact_rms[j])}
    def win(a_, b_, n, k, j=None):
        d = CRIT[k](a_, n, j) - CRIT[k](b_, n, j)
        if abs(d.mean()) <= d.std(ddof=1):
            return {a_, b_}
        return {a_} if d.mean() < 0 else {b_}
    bf = lambda t, on: (f"{{\\boldmath{t}}}" if t.startswith("$") else f"\\textbf{{{t}}}") if on else t
    num = lambda n: f"{n:,}".replace(",", "{,}")
    L = []
    for n in SIZES:
        cells = [num(n)]
        for a_, b_ in PAIRS:
            w = win(a_, b_, n, "nll"); d = A(a_, n, "nll") - A(b_, n, "nll")
            cells.append(bf(f"${A(b_, n, 'nll').mean():+.3f}$", b_ in w) + " $\\to$ " + bf(f"${A(a_, n, 'nll').mean():+.3f}$", a_ in w))
            cells.append(f"${d.mean():+.3f}\\pm{d.std(ddof=1):.3f}$ ({(d < 0).sum()}/5)")
            for k in ("warranted", "rms_al"):
                parts_b, parts_a = [], []
                for j in range(2):
                    ww = win(a_, b_, n, k, j)
                    parts_b.append(bf(f"{A(b_, n, k).mean(0)[j]:.2f}", b_ in ww)); parts_a.append(bf(f"{A(a_, n, k).mean(0)[j]:.2f}", a_ in ww))
                cells.append("/".join(parts_b) + " $\\to$ " + "/".join(parts_a))
        tau = np.concatenate([x["eb_tau"] for x in S[("empirical Bayes", n)]])
        cells.append(f"{np.median(tau):.3f} ({int((tau < 0.01).sum())}/15)")
        L.append(" & ".join(cells) + " \\\\")
    (TAB / "results4_scale.tex").write_text("\n".join(L) + "\n")

    # ---- figure: 2x2, rows aleatoric / epistemic, columns sigma / rho; colour + marker = model ----
    show = ["N(0,1)", "tempered", "data-free width", "empirical Bayes"]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6), sharex=True, gridspec_kw=dict(wspace=0.3, hspace=0.35))
    for r_i, (k, term) in enumerate((("rms_al", "aleatoric"), ("rms_ep", "epistemic"))):
        for c_i, par in enumerate((r"$\sigma$", r"$\rho$")):
            ax = axes[r_i, c_i]
            for lab in show:
                ns = [n for n in SIZES if (lab, n) in S]
                v = np.array([A(lab, n, k)[:, c_i] for n in ns]); mu, sd = v.mean(1), v.std(1, ddof=1)
                ax.plot(ns, mu, color=MCOL[lab], marker=MMK[lab], ms=3.8, lw=1.6)
                ax.fill_between(ns, np.maximum(mu - sd, 1e-4), mu + sd, color=MCOL[lab], alpha=0.12, linewidth=0)
            if k == "rms_al":
                ax.axhline(exact_rms[c_i], color=INK, ls=":", lw=1.2)
            else:
                err = np.array([A("N(0,1)", n, "rms_err")[:, c_i] for n in SIZES]).mean(1)
                ax.plot(SIZES, err, color=INK, ls=":", lw=1.2)
            ax.set_title(f"({'abcd'[2 * r_i + c_i]}) {term}, {par}", fontsize=8.5)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.tick_params(labelsize=7.5)
            if r_i == 1:
                ax.set_xlabel("$n$")
            if c_i == 0:
                ax.set_ylabel("RMS width", fontsize=8)
    fig.legend(handles=[Line2D([], [], color=MCOL[l], marker=MMK[l], lw=1.8, label=l) for l in show] +
                       [Line2D([], [], color=INK, ls=":", lw=1.2, label="target: exact sd (a, b); N(0,1) mean error (c, d)")],
               loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=8)
    fig.savefig(FIG / "scale_decomposition.png"); plt.close(fig)
    print(f"saved {OUT}, {TAB / 'results4_scale.tex'}, {FIG / 'scale_decomposition.png'}")


if __name__ == "__main__":
    main()
