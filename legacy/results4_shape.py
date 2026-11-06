"""Numbers and table for paper/paper.tex, Section "Results IV", subsection "Prior shape".

Analysis only (existing sweep files; no fit, no experiment-log entry). 3-member BNNs, 5 repeats, the hyperparameter
composite of Table 1 (`_hpnew` files at n = 50 / 500 / 1,000), both on stock MILE with the same chain budget:
  N(0,1)       bnn_vanilla    prior N(0,1)
  Student-t    bnn_studentt   prior Student-t(nu=4, scale=0.7071) on every weight -- the marginal of
                              N(0, tau^2), tau^2 ~ InverseGamma(2, 1) (scripts/fit/run_experiment.py, STUDENTT_NU/SCALE)
  tempered     bnn_temp_T0p*  (reference; n <= 1,000)
  data-free width bnn_gaussprior (reference)
Paired differences: Student-t minus N(0,1); also Student-t minus tempered and minus data-free width (all stock MILE).
Per (model, n, repeat): test NLL, width / warranted width (RMS ratio, 2026-10-29), z-spread, RMS aleatoric / epistemic, RMS error of the mean
(vs. exact posterior mean), TARP calibration error, R-hat max.

Outputs: results/project_summary_results4_shape.npz; paper/tables/results4_shape.tex.

Usage: python legacy/results4_shape.py
"""
import sys
from pathlib import Path

import numpy as np

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # this repository's src/
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling scripts in this folder
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401
from rbf_sbi import data                                     # noqa: E402
from rbf_sbi import metrics                                  # noqa: E402
import old_runs_common as R1      # noqa: E402

SIZES = R1.SIZES
MODELS = {"N(0,1)": "bnn_vanilla", "Student-t": "bnn_studentt", "data-free width": "bnn_gaussprior"}
TEMP = {50: "bnn_temp_T0p2", 100: "bnn_temp_T0p3", 500: "bnn_temp_T0p6", 1000: "bnn_temp_T0p8"}
TAB = ROOT / "paper" / "tables"
OUT = RESULTS / "project_summary_results4_shape.npz"


def main():
    TAB.mkdir(parents=True, exist_ok=True)
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
        return dict(nll=-float(np.mean(d["log_prob_true"])), warranted=np.sqrt((w ** 2).mean(0) / (sx ** 2 + (mu - mx) ** 2).mean(0)),
                    zstd=z.std(0), rms_al=np.sqrt((al ** 2).mean(0)), rms_ep=np.sqrt((ep ** 2).mean(0)),
                    rms_err=np.sqrt(((mu - mx) ** 2).mean(0)), tarp_ce=float(np.mean(np.abs(e - a))), rhat_max=float(d["rhat_max"]))

    S = {}
    for lab, key in MODELS.items():
        for n in SIZES:
            files = [np.load(R1.path(key, n, r), allow_pickle=True) for r in range(5)]
            if key == "bnn_studentt":   # the prior actually used, read from the files
                nu = {float(f["studentt_nu"]) for f in files}; sc = {round(float(f["studentt_scale"]), 4) for f in files}
                assert nu == {4.0} and sc == {0.7071}, (n, nu, sc)
            S[(lab, n)] = [stats(f) for f in files]
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
        for b_ in ["N(0,1)", "data-free width"] + (["tempered"] if n in TEMP else []):
            d = A("Student-t", n, "nll") - A(b_, n, "nll")
            print(f"    Student-t - {b_}: NLL {d.mean():+.3f} ± {d.std(ddof=1):.3f} ({(d < 0).sum()}/5 Student-t better)")
        for k in ("warranted", "rms_al", "rms_ep"):
            d = A("Student-t", n, k) - A("N(0,1)", n, k)
            print(f"    Student-t - N(0,1) {k}: {np.round(d.mean(0), 3)} ± {np.round(d.std(0, ddof=1), 3)}")
    np.savez_compressed(OUT, **{f"{lab}|{n}|{k}": A(lab, n, k) for (lab, n) in S for k in S[(lab, n)][0]})

    # ---- table: per n, "N(0,1) -> Student-t" with the better bold (Section 5.3 rule) ----
    CRIT = {"nll": lambda lab, n, j: A(lab, n, "nll"),
            "warranted": lambda lab, n, j: np.abs(A(lab, n, "warranted")[:, j] - 1),
            "rms_al": lambda lab, n, j: np.abs(A(lab, n, "rms_al")[:, j] - exact_rms[j]),
            "rms_ep": lambda lab, n, j: np.abs(A(lab, n, "rms_ep")[:, j] - A(lab, n, "rms_err")[:, j]),
            "rhat_max": lambda lab, n, j: A(lab, n, "rhat_max")}
    def win(a_, b_, n, k, j=None):
        d = CRIT[k](a_, n, j) - CRIT[k](b_, n, j)
        if abs(d.mean()) <= d.std(ddof=1):
            return {a_, b_}
        return {a_} if d.mean() < 0 else {b_}
    bf = lambda t, on: (f"{{\\boldmath{t}}}" if t.startswith("$") else f"\\textbf{{{t}}}") if on else t
    num = lambda n: f"{n:,}".replace(",", "{,}")
    a_, b_ = "Student-t", "N(0,1)"
    L = []
    for n in SIZES:
        w = win(a_, b_, n, "nll"); d = A(a_, n, "nll") - A(b_, n, "nll")
        cells = [num(n), bf(f"${A(b_, n, 'nll').mean():+.3f}$", b_ in w) + " $\\to$ " + bf(f"${A(a_, n, 'nll').mean():+.3f}$", a_ in w),
                 f"${d.mean():+.3f}\\pm{d.std(ddof=1):.3f}$ ({(d < 0).sum()}/5)"]
        for k in ("warranted", "rms_al", "rms_ep"):
            parts_b, parts_a = [], []
            for j in range(2):
                ww = win(a_, b_, n, k, j)
                parts_b.append(bf(f"{A(b_, n, k).mean(0)[j]:.2f}", b_ in ww)); parts_a.append(bf(f"{A(a_, n, k).mean(0)[j]:.2f}", a_ in ww))
            cells.append("/".join(parts_b) + " $\\to$ " + "/".join(parts_a))
        rb, ra = A(b_, n, "rhat_max"), A(a_, n, "rhat_max")
        cells.append(f"{rb.min():.2f}--{rb.max():.2f} $\\to$ {ra.min():.2f}--{ra.max():.2f}")
        L.append(" & ".join(cells) + " \\\\")
    (TAB / "results4_shape.tex").write_text("\n".join(L) + "\n")
    print(f"saved {OUT}, {TAB / 'results4_shape.tex'}")


if __name__ == "__main__":
    main()
