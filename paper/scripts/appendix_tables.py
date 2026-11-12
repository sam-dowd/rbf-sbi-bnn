"""Appendix tables of the paper, from the final runs (results/raw/final/*.npz). Analysis only; fits nothing.

  Tables 13-15  error of the predicted mean, error of the predictive sd, TARP calibration error for every model
                (paper/tables/understanding_results_{mean,sd,tarp}.tex)
  Tables 6-7    paired differences in test NLL (paper/tables/understanding_results_paired_{a,b}.tex)
  Tables 10-11  NPE head and encoder-width variants (paper/tables/closeout_npe_{nll,mean,sd,tarp,diff}.tex)

Merged, with function bodies unchanged, from the research repository's make_understanding_results.py (main),
make_paired_tests.py and the NPE part of make_closeout_tables.py. Bolding: lowest mean plus every row not
distinguishable from it by a two-sided paired t-test at p < 0.05 (--rule t05); dagger: at n <= 100 a two-sided
Wilcoxon signed-rank test disagrees. Repeats 0-19 at n <= 100 (--low-repeats 20), 0-4 above.

Usage: python paper/scripts/appendix_tables.py [--low-repeats 20] [--rule t05]
"""
import argparse

import numpy as np
from scipy import stats

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # this repository's src/
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401
from rbf_sbi import evaluation as MU  # noqa: E402

TABLES = ROOT / "paper" / "tables"
FINAL = MU.FINAL


# ---- Tables 13-15 ----
def moment_tables(a):
    R = MU.reference()
    vals = MU.load_vals(R, a.low_repeats)

    for table, blocks in MU.TABLE_BLOCKS.items():
        if table == "nll":   # not in the paper
            continue
        L = []
        for bi, (blab, key) in enumerate(blocks):
            bold = MU.bold_for(vals, key, a.rule, key)
            if blab is not None:
                if bi:
                    L += ["\\midrule", "\\midrule"]
                L.append(f"\\multicolumn{{{len(MU.SIZES) + 1}}}{{l}}{{\\textbf{{{blab}}}}} \\\\")
                L.append("\\midrule")
            L += MU.rows_for(vals, key, bold, table in MU.SIGNED, mark=MU.wilcoxon_marks(vals, key, bold, key))
        out = TABLES / f"understanding_results_{table}.tex"
        out.write_text("\n".join(L) + "\n")
        print(f"wrote {out} (rule={a.rule}, low_repeats={a.low_repeats})")


# ---- Tables 6-7 ----
PAIRED_SIZES = (50, 100, 500, 1000, 5000, 10000)


PAIRED_MODELS = {"N01": "final_bnn_stdnormal_pretuned", "datafree": "final_bnn_gaussprior_pretuned",
     "temp": "final_bnn_temp_pretuned", "retune": "final_bnn_stdnormal_retune_pretuned",
     "EB": "final_bnn_ebblock_pretuned",
     "GP12c": "final_bnn_pretrained_ens12_pretuned_long", "GP1c": "final_bnn_pretrained_pretuned_long",
     "GP12": "final_bnn_pretrained_ens12_pretuned", "GP1": "final_bnn_pretrained_pretuned",
     "NPE12": "final_npe_ens12", "NPE1": "final_npe", "NPE12long": "final_npe_ens12_long"}

PAIRED_LABEL = {"N01": r"BNN $\mathcal N(0,1)$", "datafree": "BNN data-free", "temp": "BNN tempered",
         "retune": r"BNN $\mathcal N(0,1)$+retune", "EB": "BNN emp.\\ Bayes",
         "GP12c": "Gaussian predictor 12 conv.", "GP1c": "Gaussian predictor 1 conv.", "GP12": "Gaussian predictor 12 capped", "GP1": "Gaussian predictor 1 capped",
         "NPE12": "NPE 12", "NPE1": "NPE 1", "NPE12long": "NPE 12 long"}

PAIRED_BLOCKS = [
    ("BNN vs NPE ensemble (thesis)", [(a, "NPE12") for a in ("N01", "datafree", "temp", "retune", "EB")]),
    ("BNN vs long-trained NPE ensemble ($n=5{,}000$ and $10{,}000$ only)", [(a, "NPE12long") for a in ("N01", "datafree", "retune", "EB")]),
    ("Sampling vs finished optimization: BNN vs converged Gaussian predictor (12 nets)",
     [(a, "GP12c") for a in ("N01", "datafree", "temp", "retune", "EB")]),
    ("Sampling vs the BNN's own start: BNN vs capped Gaussian predictor (12 nets)",
     [(a, "GP12") for a in ("N01", "datafree", "temp")]),
    ("Ensembling: 12 members vs member 0", [("NPE12", "NPE1"), ("GP12c", "GP1c"), ("GP12", "GP1")]),
    ("Between the unsampled baselines", [("GP12c", "NPE12"), ("GP12c", "GP12")]),
    ("Prior, likelihood and sampler variants", [("datafree", "N01"), ("temp", "N01"), ("retune", "N01"), ("EB", "retune")]),
]


def _paired_load(m, n):
    """{repeat: (nll, X_train or None)} for every existing file of model m at size n."""
    out = {}
    for r in (range(20) if n <= 100 else range(5)):
        f = FINAL / f"{PAIRED_MODELS[m]}_n{n}_r{r}.npz"
        if f.exists():
            with np.load(f, allow_pickle=True) as d:
                out[r] = (-float(np.mean(d["log_prob_true"])), np.asarray(d["X_train"]) if "X_train" in d.files else None)
    return out


def fmt_p(p):
    return "<.001" if p < 0.001 else f"{p:.3f}".lstrip("0")


def _paired_cell(a, b, n):
    A, B = _paired_load(a, n), _paired_load(b, n)
    reps = sorted(set(A) & set(B))
    if len(reps) < 5:
        return "--", None
    for r in reps:
        if A[r][1] is not None and B[r][1] is not None:
            assert np.array_equal(A[r][1], B[r][1]), f"unpaired training rows: {a} vs {b}, n={n}, r={r}"
    x, y = np.array([A[r][0] for r in reps]), np.array([B[r][0] for r in reps])
    d = x - y
    pt = stats.ttest_rel(x, y).pvalue
    k, R = int((d < 0).sum()), len(d)
    top = f"${d.mean():+.3f}\\pm{d.std(ddof=1):.3f}$"
    if pt < 0.05:
        top = r"{\boldmath" + top + "}"
    bot = f"{k}/{R}, $t$ {fmt_p(pt)}"
    if n <= 100:
        pw = stats.wilcoxon(x, y).pvalue
        bot += f", W {fmt_p(pw)}"
        if (pt < 0.05) != (pw < 0.05):
            top += r"$^\dagger$"
    return rf"\pc{{{top}}}{{{bot}}}", (d.mean(), pt)


def paired_tables():
    out_dir = TABLES
    for suffix, blocks in (("a", PAIRED_BLOCKS[:3]), ("b", PAIRED_BLOCKS[3:])):
        lines = []
        for title, pairs in blocks:
            lines.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{title}}}}} \\")
            for a, b in pairs:
                cells = []
                for n in PAIRED_SIZES:
                    c, res = _paired_cell(a, b, n)
                    cells.append(c)
                    if res is not None:
                        print(f"{a:>9} - {b:<9} n={n:>5}: {res[0]:+.3f}  t p {res[1]:.3g}")
                lines.append(f"{PAIRED_LABEL[a]} $-$ {PAIRED_LABEL[b]} & " + " & ".join(cells) + r" \\")
            lines.append(r"\midrule")
        path = out_dir / f"understanding_results_paired_{suffix}.tex"
        path.write_text("\n".join(lines[:-1]) + "\n")
        print(f"wrote {path}")


# ---- Tables 10-11 (NPE variants) ----
N01, DF, TEMP = "BNN, $\\mathcal N(0,1)$", "BNN, data-free scale", "BNN, tempered"

GP12, GP1 = "Gaussian predictor, 12 nets, capped", "Gaussian predictor, 1 net, capped"

GP12C, GP1C = "Gaussian predictor, 12 nets, converged", "Gaussian predictor, 1 net, converged"

NPE12, NPE1 = "NPE, 12 flows", "NPE, 1 flow"

F8, F16, G16 = "Flow, encoder 8 (= NPE)", "Flow, encoder 16", "Gaussian head, encoder 16"

F8_1, F16_1, G16_1 = "Flow, encoder 8 (= NPE), 1 net", "Flow, encoder 16, 1 net", "Gaussian head, encoder 16, 1 net"

RET, EB = "BNN, $\\mathcal N(0,1)$ + retune", "BNN, empirical Bayes"

SAMPLED = ("Sampled BNN", [(N01, "final_bnn_stdnormal_pretuned"), (DF, "final_bnn_gaussprior_pretuned"),
                           (TEMP, "final_bnn_temp_pretuned")])
COMPARISONS = {   # the NPE comparison only (the Gaussian-predictor one is not in the paper)
    "npe": dict(
        sizes=[50, 100, 10000],
        groups=[SAMPLED,
                ("BNN without sampling", [(GP12, "final_bnn_pretrained_ens12_pretuned"),
                                          (GP12C, "final_bnn_pretrained_ens12_pretuned_long")]),
                ("NPE-style, 12 networks", [(F8, {50: "final_npe_ens12", 100: "final_npe_ens12",
                                                  10000: "final_npe_ens12_long"}),
                                            (F16, "final_npe_ctx16_ens12"), (G16, "final_npe_gauss_ctx16_ens12")]),
                ("NPE-style, 1 network (member 0)", [(F8_1, {50: "final_npe", 100: "final_npe",
                                                             10000: "final_npe_long"}),
                                                     (F16_1, "final_npe_ctx16"), (G16_1, "final_npe_gauss_ctx16")])],
        contrasts=[("Flow enc.\\ 16 $-$ flow enc.\\ 8 (capacity)", F16, F8),
                   ("Gauss.\\ head enc.\\ 16 $-$ flow enc.\\ 16 (head)", G16, F16),
                   ("BNN data-free $-$ flow enc.\\ 16", DF, F16),
                   ("BNN $\\mathcal N(0,1)$ $-$ flow enc.\\ 16", N01, F16),
                   ("BNN data-free $-$ Gauss.\\ head enc.\\ 16", DF, G16),
                   ("Gauss.\\ predictor conv.\\ $-$ Gauss.\\ head enc.\\ 16", GP12C, G16)]),
}


DAGGER, wilcoxon_p = MU.DAGGER, MU.wilcoxon_p


DIFF_BLOCKS = [("Test NLL", "nll"), ("Mean error, $\\sigma$", "mean_sigma"), ("Mean error, $\\rho$", "mean_rho"),
               ("sd error, $\\sigma$", "sd_sigma"), ("sd error, $\\rho$", "sd_rho"), ("TARP error", "tarp")]


def diff_rows(vals, contrasts, sizes, key):
    L = []
    for lab, a, b in contrasts:
        cells = []
        for n in sizes:
            if (a, n) not in vals or (b, n) not in vals:
                cells.append("--"); continue
            d = vals[(a, n)][key] - vals[(b, n)][key]
            p = stats.ttest_1samp(d, 0.0).pvalue if np.any(d != 0) else 1.0
            pw = wilcoxon_p(d)
            t = f"$({d.mean():+.3f}\\pm{d.std(ddof=1):.3f})$"
            t = f"{{\\boldmath{t}}}" if p < 0.05 else t
            dag = DAGGER if n <= 100 and (p < 0.05) != (pw < 0.05) else ""
            cells.append(f"{t}{dag}\\,{{\\scriptsize {int((d < 0).sum())}/{len(d)}}}")
            print(f"[diff {key}] n={n:5d} {lab:55s} {d.mean():+.4f}+-{d.std(ddof=1):.4f}  median {np.median(d):+.4f}  "
                  f"A better {int((d < 0).sum())}/{len(d)}  two-sided t p {p:.3g}  Wilcoxon p {pw:.3g}")
        L.append(" & ".join([lab] + cells) + " \\\\")
    return L


def report_training(groups, sizes, low_repeats):
    """Training-length / member-finiteness summary for the new files (printed, for the captions)."""
    for _, rows in groups:
        for lab, stem in rows:
            for n in sizes:
                s = stem.get(n) if isinstance(stem, dict) else stem
                if s is None or not any(k in s for k in ("_long", "ctx16")) or "1 net" in lab:
                    continue
                reps = range(low_repeats if n <= 100 else 5)
                fs = [FINAL / f"{s}_n{n}_r{r}.npz" for r in reps]
                if not all(f.exists() for f in fs):
                    continue
                out = []
                for f in fs:
                    with np.load(f, allow_pickle=True) as d:
                        if "pretrain_stop_epoch" in d.files:
                            out.append((np.atleast_1d(d["pretrain_stop_epoch"]), np.atleast_1d(d["pretrain_last_epoch"]),
                                        np.atleast_1d(d["member_finite"])))
                        elif "member_steps_run" in d.files:
                            out.append((np.where(d["member_stopped_early"], 0, -1), d["member_best_step"],
                                        np.ones(len(d["member_best_step"]), bool)))
                stop = np.concatenate([o[0] for o in out]); steps = np.concatenate([o[1] for o in out])
                fin = np.concatenate([o[2] for o in out])
                print(f"[train] {s:45s} n={n:5d}: members {stop.size}, at cap {(stop == -1).sum()}, finite "
                      f"{fin.sum()}/{fin.size}, steps (GP: last epoch; NPE: best step) median {np.median(steps):.0f} "
                      f"range {steps.min()}-{steps.max()}")


def npe_variant_tables(a):
    R = MU.reference()
    ref_x = {}
    for name, C in COMPARISONS.items():
        groups, sizes = C["groups"], C["sizes"]
        vals = MU.load_vals(R, a.low_repeats, groups, sizes, ref_x)
        missing = [(lab, n) for _, rows in groups for lab, st in rows for n in sizes
                   if (lab, n) not in vals and (not isinstance(st, dict) or n in st) and not (lab == TEMP and n > 1000)]
        assert not missing, f"{name}: missing files for {missing}"
        report_training(groups, sizes, a.low_repeats)
        for lab, n in sorted(vals, key=lambda t: t[1]):
            x = vals[(lab, n)]["nll"]
            print(f"[{name} nll median] n={n:5d} {lab:42s} mean {x.mean():+.3f} median {np.median(x):+.3f} "
                  f"max {x.max():+.3f}")
        for table, blocks in MU.TABLE_BLOCKS.items():
            L = []
            for bi, (blab, key) in enumerate(blocks):
                bold = MU.bold_for(vals, key, a.rule, f"{name} {key}", groups, sizes)
                if blab is not None:
                    if bi:
                        L += ["\\midrule", "\\midrule"]
                    L.append(f"\\multicolumn{{{len(sizes) + 1}}}{{l}}{{\\textbf{{{blab}}}}} \\\\")
                    L.append("\\midrule")
                mark = MU.wilcoxon_marks(vals, key, bold, f"{name} {key}", groups, sizes)
                L += MU.rows_for(vals, key, bold, table in MU.SIGNED, groups, sizes, mark)
            out = TABLES / f"closeout_{name}_{table}.tex"
            out.write_text("\n".join(L) + "\n")
            print(f"wrote {out}")
        L = []
        for bi, (blab, key) in enumerate(DIFF_BLOCKS):
            if bi:
                L.append("\\midrule")
            L.append(f"\\multicolumn{{{len(sizes) + 1}}}{{l}}{{\\textbf{{{blab}}}}} \\\\")
            L += diff_rows(vals, C["contrasts"], sizes, key)
        out = TABLES / f"closeout_{name}_diff.tex"
        out.write_text("\n".join(L) + "\n")
        print(f"wrote {out} (rule={a.rule}, low_repeats={a.low_repeats})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low-repeats", type=int, default=20)
    ap.add_argument("--rule", choices=["strict", "t045", "t05"], default="t05")
    a = ap.parse_args()
    moment_tables(a)
    paired_tables()
    npe_variant_tables(a)


if __name__ == "__main__":
    main()
