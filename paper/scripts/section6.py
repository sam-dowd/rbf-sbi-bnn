"""Tables and figures for Section 6 of paper/paper.tex (2026-11-05): test NLL only.

Analysis only (reads results/raw/final/*.npz and, if present, results/lppd_decomposition.npz; fits nothing; no
experiment-log entry). Final basis: pretraining-tuned 12-member runs, repeats 0-19 at n <= 100 and 0-4 above, the
converged Gaussian predictor, NPE with its own tuned lr/wd, the long-trained NPE ensemble at n=10,000.

Section 6.1 (sampling and ensembling together): the three BNNs against a SINGLE Gaussian-predictor network and a
SINGLE NPE flow. Section 6.2 (sampling alone): the same BNNs against the 12-member ensembles. Bolding and daggers are
make_understanding_results.py's (imported unchanged): bold = lowest mean and every row not beaten by it under a
two-sided paired t-test at p < 0.05 (rule t05); dagger = at n <= 100 a two-sided Wilcoxon signed-rank test disagrees.
Training subsample per (n, repeat) asserted identical across every row.

Outputs: paper/tables/section6_{single,ensemble}_nll.tex (rows), paper/tables/section6_{lppd,draw}.tex (rows, if the
decomposition has been merged; split into two tables and bolded 2026-11-06), paper/tables/appendix_width.tex (rows: width over warranted width, coherence ratio,
aleatoric share of the excess width -- definitions in Section 5.1 of the write-up),
paper/figures/project_summary/section6/nll_{single,ensemble}.pdf, paper/figures/project_summary/section6/tarp_ensembles.pdf
(appendix TARP figure, 2026-11-06). section6_ensemble_nll.tex also carries a |z| > 3 block below the NLL rows (2026-11-06).
Usage: python paper/scripts/section6.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # this repository's src/
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401
from rbf_sbi import evaluation as AFS          # noqa: E402
from rbf_sbi import evaluation as MUR   # noqa: E402

TABLES = ROOT / "paper" / "tables"
FIG = ROOT / "paper" / "figures" / "project_summary" / "section6"
FINAL = RESULTS / "raw" / "final"
SIZES = [50, 100, 500, 1000, 5000, 10000]
EXACT_NLL = -2.390
BNNS = [("BNN, $\\mathcal N(0,1)$", "final_bnn_stdnormal_pretuned"),
        ("BNN, data-free scale", "final_bnn_gaussprior_pretuned"),
        ("BNN, tempered", "final_bnn_temp_pretuned")]
GROUPS_SINGLE = [("Sampled BNN (12 chains)", BNNS),
                 ("Single network", [("Gaussian predictor, 1 net", "final_bnn_pretrained_pretuned_long"),
                                     ("NPE, 1 flow", "final_npe")])]
GROUPS_ENS = [("Sampled BNN (12 chains)", BNNS),
              ("12-member ensembles", [("Gaussian predictor, 12 nets", "final_bnn_pretrained_ens12_pretuned_long"),
                                       ("NPE, 12 flows", "final_npe_ens12"),
                                       ("NPE, 12 flows, long-trained", {10000: "final_npe_ens12_long"})])]
# entity colours: dataviz categorical slots 1-5 in fixed order (validated light: all checks pass; aqua / yellow /
# magenta < 3:1 contrast on white -> distinct markers, legend and the table as relief)
STYLE = {"BNN, $\\mathcal N(0,1)$": ("#2a78d6", "o", "-"), "BNN, data-free scale": ("#eda100", "D", "-"),
         "BNN, tempered": ("#e87ba4", "v", "-"),
         "Gaussian predictor, 1 net": ("#1baf7a", "^", "--"), "NPE, 1 flow": ("#eb6834", "s", "--"),
         "Gaussian predictor, 12 nets": ("#1baf7a", "^", "--"), "NPE, 12 flows": ("#eb6834", "s", "--"),
         "NPE, 12 flows, long-trained": ("#eb6834", "P", "none")}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 9, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.8,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white", "legend.frameon": False,
})


def table(groups, name, R, ref_x):
    vals = MUR.load_vals(R, 20, groups=groups, sizes=SIZES, ref_x=ref_x)
    bold = MUR.bold_for(vals, "nll", "t05", name, groups=groups, sizes=SIZES)
    mark = MUR.wilcoxon_marks(vals, "nll", bold, name, groups=groups, sizes=SIZES)
    rows = MUR.rows_for(vals, "nll", bold, True, groups=groups, sizes=SIZES, mark=mark)
    out = TABLES / f"section6_{name}_nll.tex"
    out.write_text("\n".join(rows) + "\n")
    print(f"wrote {out}")
    return vals


def panel(ax, vals, groups, sizes, title, ylim, exact=True):
    labs = [lab for _, rows in groups for lab, _ in rows]
    mt = lambda t: t.replace("\\mathcal N", "\\mathcal{N}")   # matplotlib mathtext needs braces
    for lab in labs:
        ns = [n for n in sizes if (lab, n) in vals]
        if not ns:
            continue
        col, mk, ls = STYLE[lab]
        m = np.array([vals[(lab, n)]["nll"].mean() for n in ns])
        s = np.array([vals[(lab, n)]["nll"].std(ddof=1) for n in ns])
        if ls != "none":
            ax.fill_between(ns, m - s, m + s, color=col, alpha=0.15, linewidth=0)
            ax.plot(ns, m, color=col, marker=mk, markersize=5, linestyle=ls, markeredgecolor="white",
                    markeredgewidth=0.8, label=mt(lab))
        else:
            ax.errorbar(ns, m, yerr=s, color=col, marker=mk, markersize=7, linestyle="none", capsize=2,
                        markeredgecolor="white", markeredgewidth=0.8, label=mt(lab))
        for n, mi, si in zip(ns, m, s):   # off-scale means are written, not drawn
            if mi > ylim[1]:
                ax.annotate(f"{mt(lab)}: ${mi:+.2f}\\pm{si:.2f}$",
                            (n, ylim[1]), xytext=(4, -10 - 10 * labs.index(lab) % 30), textcoords="offset points",
                            fontsize=7, color=INK2, arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8))
    if exact:
        ax.axhline(EXACT_NLL, color=MUTED, linestyle=":", linewidth=1.0)
        ax.annotate("exact posterior", (sizes[0], EXACT_NLL), xytext=(2, 3), textcoords="offset points",
                    color=INK2, fontsize=8, va="bottom")
    ax.set_ylim(*ylim)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{n:,}" for n in sizes])
    ax.minorticks_off()
    ax.set_xlabel("training size $n$")
    ax.set_ylabel("test NLL (nats; lower is better)")
    ax.set_title(title, loc="left")


def figure(vals, groups, name, ylim_all, ylim_zoom):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    panel(axes[0], vals, groups, SIZES, "(a) all training sizes", ylim_all)
    panel(axes[1], vals, groups, [n for n in SIZES if n <= 1000], "(b) zoom: $n \\leq 1{,}000$", ylim_zoom,
          exact=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.07), fontsize=8)
    out = FIG / f"nll_{name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def _lppd_series(d, model, key):
    """{n: (repeats, values)} from results/lppd_decomposition.npz, sorted by repeat."""
    out = {}
    for n in SIZES:
        m = (d["model"] == model) & (d["n"] == n)
        if m.any():
            o = np.argsort(d["repeat"][m])
            out[n] = (d["repeat"][m][o], d[key][m][o])
    return out


def _file_series(stem):
    """{n: (repeats, test NLL)} from results/raw/final (same repeats as everywhere in Section 6)."""
    out = {}
    for n in SIZES:
        reps = np.arange(20 if n <= 100 else 5)
        files = [FINAL / f"{stem}_n{n}_r{r}.npz" for r in reps]
        if all(f.exists() for f in files):
            out[n] = (reps, np.array([-float(np.mean(np.load(f)["log_prob_true"])) for f in files]))
    return out


def _bolded(rows):
    """rows: list of (label or None for a group header string, {n: (repeats, values)}). Bold per column: the lowest mean
    and every row not separable from it by a two-sided paired t-test at p < 0.05 (the Section 6 rule t05)."""
    from scipy.stats import ttest_rel, wilcoxon
    bold, dag = {}, {}
    for n in SIZES:
        cand = [(k, ser[n]) for k, (lab, ser) in enumerate(rows) if isinstance(ser, dict) and n in ser]
        if not cand:
            continue
        kb, (rb, vb) = min(cand, key=lambda c: np.mean(c[1][1]))
        for k, (r, v) in cand:
            assert np.array_equal(r, rb), f"repeat mismatch at n={n}"
            if k == kb or ttest_rel(v, vb).pvalue >= 0.05:
                bold[(k, n)] = True
            # dagger, as in the other Section 6 tables: at n <= 100 a two-sided Wilcoxon signed-rank test disagrees
            if k != kb and n <= 100 and (wilcoxon(v, vb).pvalue < 0.05) == bool(bold.get((k, n))):
                dag[(k, n)] = True
    L = []
    for k, (lab, ser) in enumerate(rows):
        if not isinstance(ser, dict):
            L.append(lab)
            continue
        cells = []
        for n in SIZES:
            if n not in ser:
                cells.append("--")
                continue
            v = ser[n][1]
            c = f"{np.mean(v):+.3f}\\pm{np.std(v, ddof=1):.3f}"
            d = "$^\\dagger$" if dag.get((k, n)) else ""
            cells.append((f"{{\\boldmath${c}$}}" if bold.get((k, n)) else f"${c}$") + d)
        L.append(" & ".join([lab] + cells) + r" \\")
    return L


def lppd_rows():
    """Two tables from results/lppd_decomposition.npz (2026-11-06 split of the former single table):
    section6_lppd.tex -- averaging inside the BNN: the starting networks (1, 12) and, per BNN, one chain and 12 chains;
    section6_draw.tex -- a typical single posterior draw (median over the 12 x 1,000 draws of each draw's own test NLL)
    against optimized networks (capped member 0, converged member 0, converged 12-network ensemble).
    Bold per column as in _bolded."""
    f = RESULTS / "lppd_decomposition.npz"
    if not f.exists():
        print(f"{f} not found -- run the decomposition and --merge first; section6_lppd/draw.tex not written")
        return
    d = np.load(f)
    lab = {"final_bnn_stdnormal_pretuned": "BNN, $\\mathcal N(0,1)$", "final_bnn_gaussprior_pretuned": "BNN, data-free scale",
           "final_bnn_temp_pretuned": "BNN, tempered"}
    ref = "final_bnn_stdnormal_pretuned"     # the starting networks are the N(0,1) run's own pretrained networks
    rows = [(r"\multicolumn{7}{l}{\emph{The BNN's starting networks (capped Gaussian predictor)}} \\", None),
            ("1 network (member 0)", _lppd_series(d, ref, "nll_start1")),
            ("12 networks", _lppd_series(d, ref, "nll_start12"))]
    for model, name in lab.items():
        rows += [(r"\midrule", None), (rf"\multicolumn{{7}}{{l}}{{\emph{{{name}}}}} \\", None),
                 ("one chain", _lppd_series(d, model, "nll_chain")),
                 ("12 chains (the BNN)", _lppd_series(d, model, "nll_bnn"))]
    out = TABLES / "section6_lppd.tex"
    out.write_text("\n".join(_bolded(rows)) + "\n")
    print(f"wrote {out}")

    rows = [(r"\multicolumn{7}{l}{\emph{Optimized networks (Gaussian predictor)}} \\", None),
            ("capped, 1 network (MILE's start)", _lppd_series(d, ref, "nll_start1")),
            ("converged, 1 network", _file_series("final_bnn_pretrained_pretuned_long")),
            ("converged, 12 networks", _file_series("final_bnn_pretrained_ens12_pretuned_long")),
            (r"\midrule", None),
            (r"\multicolumn{7}{l}{\emph{A single posterior draw, used alone (median over the $12\times1{,}000$ draws)}} \\", None)]
    rows += [(name, _lppd_series(d, model, "nll_draw")) for model, name in lab.items()]
    out = TABLES / "section6_draw.tex"
    out.write_text("\n".join(_bolded(rows)) + "\n")
    print(f"wrote {out}")


def _ms(v):
    return "--" if len(v) == 0 else f"${np.mean(v):+.3f}\\pm{np.std(v, ddof=1):.3f}$"


WIDTH_MODELS = BNNS + [("Gaussian predictor, 12 nets", "final_bnn_pretrained_ens12_pretuned_long"),
                       ("NPE, 12 flows", "final_npe_ens12")]


def width_rows(R):
    """Per model and n, mean over repeats of (sigma / rho): width over warranted width (RMS ratio), coherence ratio,
    and for the N(0,1) and tempered BNNs the aleatoric share of the excess variance
    (E[al^2] - E[s*^2]) / (E[w^2] - E[s*^2] - E[(mu - mu*)^2]), which is only defined where there is an excess."""
    th, mx, sx = R["th"], R["mx"], R["sx"]
    stats = {}
    for lab, stem in WIDTH_MODELS:
        for n in SIZES:
            files = [FINAL / f"{stem}_n{n}_r{r}.npz" for r in (range(20) if n <= 100 else range(5))]
            if not all(f.exists() for f in files):
                continue
            W, C, A = [], [], []
            for f in files:
                with np.load(f, allow_pickle=True) as d:
                    assert np.allclose(d["theta_test"], th)
                    w, mu = AFS.width(d), d["mean"]
                    err2 = ((mu - mx) ** 2).mean(0)
                    W.append(np.sqrt((w ** 2).mean(0) / ((sx ** 2).mean(0) + err2)))
                    C.append(((w ** 2).mean(0) + mu.var(0)) / th.var(0))
                    if "epistemic" in d.files and stem in ("final_bnn_stdnormal_pretuned", "final_bnn_temp_pretuned"):
                        al = d["aleatoric"]
                        A.append(((al ** 2).mean(0) - (sx ** 2).mean(0)) / ((w ** 2).mean(0) - (sx ** 2).mean(0) - err2))
            stats[(lab, n)] = (np.mean(W, 0), np.mean(C, 0), np.mean(A, 0) if A else None)
    pair = lambda v, f="{:.2f}": "--" if v is None else (f + " / " + f).format(*v)
    L = []
    for bi, (title, idx) in enumerate([("Width over warranted width (1: width explained by the method's own mean error)", 0),
                                       ("Coherence ratio (1: coherent; exact posterior 1.00 / 1.01)", 1),
                                       ("Aleatoric share of the excess variance", 2)]):
        if bi:
            L.append(r"\midrule")
        L.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{title}}}}} \\")
        for lab, stem in WIDTH_MODELS:
            if idx == 2 and stem not in ("final_bnn_stdnormal_pretuned", "final_bnn_temp_pretuned"):
                continue
            cells = []
            for n in SIZES:
                v = stats.get((lab, n))
                if v is None or (idx == 2 and (v[2] is None or n >= 5000)):
                    cells.append("--")   # no excess to split at n >= 5,000 (width / warranted ~ 1)
                else:
                    cells.append(pair(v[idx]))
            L.append(" & ".join([lab] + cells) + r" \\")
    out = TABLES / "appendix_width.tex"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


def _width(d):
    return d["std"] if "std" in d.files else (np.sqrt(d["aleatoric"] ** 2 + d["epistemic"] ** 2)
                                              if "epistemic" in d.files else d["aleatoric"])


def zfrac_rows():
    """Appended to section6_ensemble_nll.tex (2026-11-06): per model and n, the percentage of the 1,000 test datasets
    with |z| > 3 in at least one parameter, z = (predicted mean - theta*) / predicted sd (the model's total predictive
    sd per parameter); median over repeats, worst repeat in parentheses. The exact posterior scores
    0.9% (a Gaussian would give about 0.5%; the posteriors are not exactly Gaussian). Same files and repeats as the NLL rows."""
    L = [r"\midrule",
         r"\multicolumn{7}{l}{\emph{Test datasets with $|z|>3$ (\%): median over repeats (worst repeat)}} \\"]
    for _, members in GROUPS_ENS:
        for lab, stem in members:
            cells = []
            for n in SIZES:
                st = stem.get(n) if isinstance(stem, dict) else stem
                files = None if st is None else [FINAL / f"{st}_n{n}_r{r}.npz" for r in (range(20) if n <= 100 else range(5))]
                if files is None or not all(f.exists() for f in files):
                    cells.append("--")
                    continue
                z = []
                for f in files:
                    with np.load(f, allow_pickle=True) as d:
                        z.append(100 * (np.abs((d["mean"] - d["theta_test"]) / _width(d)) > 3).any(1).mean())
                cells.append(f"${np.median(z):.1f}$ (${np.max(z):.1f}$)")
            L.append(" & ".join([lab] + cells) + r" \\")
    out = TABLES / "section6_ensemble_nll.tex"
    out.write_text(out.read_text() + "\n".join(L) + "\n")
    print(f"appended |z|>3 rows to {out}")


def tarp_figure():
    """Appendix figure (2026-11-06): TARP coverage deviation ECP(alpha) - alpha against alpha for the five ensembled
    models (3 BNNs, Gaussian predictor 12 nets converged, NPE 12 flows), one panel per n. Line: pooled over repeats
    (metrics.tarp_curve_from_f, as for every TARP number in the write-up); band: +-1 sd over the per-repeat curves.
    0 = calibrated, > 0 conservative (regions cover the truth more often than claimed), < 0 over-confident."""
    from rbf_sbi import metrics
    models = BNNS + [("Gaussian predictor, 12 nets", "final_bnn_pretrained_ens12_pretuned_long"),
                     ("NPE, 12 flows", "final_npe_ens12")]
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.8), sharex=True)
    for ax, n in zip(axes.ravel(), SIZES):
        for lab, stem in models:
            files = [FINAL / f"{stem}_n{n}_r{r}.npz" for r in (range(20) if n <= 100 else range(5))]
            if not all(f.exists() for f in files):
                continue
            fs = [np.asarray(np.load(f)["f"]) for f in files]
            a, ecp, _ = metrics.tarp_curve_from_f(fs)
            per = np.stack([metrics.tarp_curve_from_f([x])[1] - a for x in fs])
            col, mk, _ls = STYLE[lab]
            ax.fill_between(a, per.mean(0) - per.std(0, ddof=1), per.mean(0) + per.std(0, ddof=1), color=col,
                            alpha=0.15, linewidth=0)
            ax.plot(a, ecp - a, color=col, label=lab.replace("\\mathcal N", "\\mathcal{N}"),
                    linestyle="-" if lab.startswith("BNN") else "--")
            print(f"TARP n={n:5d} {lab:30s} area {np.trapezoid(ecp - a, a):+.4f}")
        ax.axhline(0, color=MUTED, linewidth=1.0)
        ax.set_title(f"$n={n:,}$", loc="left")
        if ax in axes[:, 0]:
            ax.set_ylabel("ECP$(\\alpha)-\\alpha$")
        if ax in axes[1]:
            ax.set_xlabel("credibility level $\\alpha$")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.04), fontsize=8)
    fig.tight_layout()
    out = FIG / "tarp_ensembles.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    R, ref_x = AFS.reference(), {}
    vs = table(GROUPS_SINGLE, "single", R, ref_x)
    ve = table(GROUPS_ENS, "ensemble", R, ref_x)
    zfrac_rows()
    tarp_figure()
    figure(vs, GROUPS_SINGLE, "single", (-2.45, 0.9), (-2.0, 0.9))
    figure(ve, GROUPS_ENS, "ensemble", (-2.45, -0.1), (-2.0, -0.1))
    lppd_rows()
    width_rows(R)
    verdict_table()


# ---- verdict table (Table 5; merged from make_verdict_table.py) ----
from scipy.stats import ttest_rel  # noqa: E402

VERDICT_OUT = TABLES / "section6_verdict.tex"

V_BNNS = [("$\\mathcal N(0,1)$", "final_bnn_stdnormal_pretuned"),
        ("data-free scale", "final_bnn_gaussprior_pretuned"),
        ("tempered", "final_bnn_temp_pretuned")]

V_COMPS = [("NPE, 12 flows", {n: "final_npe_ens12" for n in SIZES}),
         ("Gaussian predictor, 12 nets", {n: "final_bnn_pretrained_ens12_pretuned_long" for n in SIZES})]

V_NPE_LONG = "final_npe_ens12_long"

V_P = 0.05

def _v_nll(stem, n):
    reps = range(20) if n <= 100 else range(5)
    files = [FINAL / f"{stem}_n{n}_r{r}.npz" for r in reps]
    if not all(f.exists() for f in files):
        return None
    v, xs = [], []
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            v.append(-float(np.mean(d["log_prob_true"])))
            xs.append(np.asarray(d["X_train"]))
    return np.array(v), xs

def _v_cell(a, b):
    (va, xa), (vb, xb) = a, b
    for p, q in zip(xa, xb):
        assert np.array_equal(p, q), "training subsample mismatch"
    d = va - vb
    p = ttest_rel(va, vb).pvalue
    if p >= V_P:
        word = "tie"
    else:
        word = "\\textbf{better}" if d.mean() < 0 else "worse"
    return f"{word} ${d.mean():+.2f}$", d.mean(), p

def verdict_table():
    rows = []
    for clab, cstem in V_COMPS:
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{BNN against {clab}}}}} \\\\")
        for blab, bstem in V_BNNS:
            cells = []
            for n in SIZES:
                a, b = _v_nll(bstem, n), _v_nll(cstem[n], n)
                if a is None or b is None:
                    cells.append("--")
                    continue
                txt, m, p = _v_cell(a, b)
                print(f"{blab:16s} vs {clab:28s} n={n:5d} diff {m:+.3f} p={p:.2g} -> {txt}")
                if clab.startswith("NPE") and n == 10000:
                    c = _v_nll(V_NPE_LONG, n)
                    if c is not None:
                        t2, m2, p2 = _v_cell(a, c)
                        print(f"{blab:16s} vs NPE long                     n={n:5d} diff {m2:+.3f} p={p2:.2g} -> {t2}")
                cells.append(txt)
            rows.append(f"{blab} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    VERDICT_OUT.write_text("% generated by scripts/make_verdict_table.py -- do not edit\n" + "\n".join(rows) + "\n")
    print(f"wrote {VERDICT_OUT}")


if __name__ == "__main__":
    main()
