"""Shared evaluation helpers for the analysis and paper scripts.

Merged, with function bodies unchanged, from the research repository's analyze_final_sweep.py (reference, width),
make_understanding_results.py (per-fit metrics, bolding, Wilcoxon marks, row formatting, loading of the final runs),
make_project_summary_results1.py (REF, SIZES) and optimize_bde.py (build_repeat_splits, used by the pretraining grid).
"""
import numpy as np
from scipy import stats

from rbf_sbi import data, metrics
from rbf_sbi.paths import DATA, RESULTS

REF = DATA / "reference_posterior" / "test_n100_ntest1000_seed0.npz"   # exact posteriors of the 1,000 test datasets
FINAL = RESULTS / "raw" / "final"
SIZES = [50, 100, 500, 1000, 5000, 10000]


# ---- exact reference and predictive width ----
def reference():
    ref = np.load(REF); th = ref["theta"]
    sg, rg = np.asarray(data.SIGMA_GRID), np.asarray(data.RHO_GRID)
    ps = ref["p_sigma"] / ref["p_sigma"].sum(1, keepdims=True); pr = ref["p_rho"] / ref["p_rho"].sum(1, keepdims=True)
    mx = np.stack([(ps * sg).sum(1), (pr * rg).sum(1)], 1)
    sx = np.stack([np.sqrt((ps * (sg - mx[:, :1]) ** 2).sum(1)), np.sqrt((pr * (rg - mx[:, 1:]) ** 2).sum(1))], 1)
    q = lambda pm, g, a: np.array([np.interp(a, c, g) for c in np.cumsum(pm, 1)])
    hx = np.stack([(q(ps, sg, .84) - q(ps, sg, .16)) / 2, (q(pr, rg, .84) - q(pr, rg, .16)) / 2], 1)
    exact_nll = -float(np.mean(np.asarray(data.compute_reference_log_prob_true(ref["post"], th))))
    return dict(th=th, mx=mx, sx=sx, hx=hx, exact_nll=exact_nll)


def width(d):
    if "std" in d.files:
        return d["std"]
    if "epistemic" in d.files:
        return np.sqrt(d["aleatoric"] ** 2 + d["epistemic"] ** 2)
    return d["aleatoric"]


# ---- per-fit metrics, paired-test bolding and table rows ----
GROUPS = [   # (group label, [(row label, file stem)])
    ("Sampled BNN", [("BNN, $\\mathcal N(0,1)$", "final_bnn_stdnormal_pretuned"),
                     ("BNN, data-free scale", "final_bnn_gaussprior_pretuned"),
                     ("BNN, tempered", "final_bnn_temp_pretuned")]),
    ("BNN without sampling", [("Gaussian predictor, 12 nets, capped", "final_bnn_pretrained_ens12_pretuned"),
                              ("Gaussian predictor, 12 nets, converged", "final_bnn_pretrained_ens12_pretuned_long"),
                              ("Gaussian predictor, 1 net, capped", "final_bnn_pretrained_pretuned"),
                              ("Gaussian predictor, 1 net, converged", "final_bnn_pretrained_pretuned_long")]),
    ("NPE", [("NPE, 12 flows", "final_npe_ens12"), ("NPE, 1 flow", "final_npe")]),
    ("Appendix (retune sampler)", [("BNN, $\\mathcal N(0,1)$ + retune", "final_bnn_stdnormal_retune_pretuned"),
                                   ("BNN, empirical Bayes", "final_bnn_ebblock_pretuned")]),
]

TABLE_BLOCKS = {
    "nll": [(None, "nll")],
    "mean": [("$\\sigma$", "mean_sigma"), ("$\\rho$", "mean_rho")],
    "sd": [("$\\sigma$", "sd_sigma"), ("$\\rho$", "sd_rho")],
    "tarp": [(None, "tarp")],
}

SIGNED = {"nll"}


def fit_metrics(d, R):
    assert np.allclose(d["theta_test"], R["th"])
    mu, w = d["mean"], width(d)
    a, ecp, _ = metrics.tarp_curve_from_f([d["f"]])
    rms = np.sqrt(((mu - R["mx"]) ** 2).mean(0))
    we = (np.abs(w - R["sx"]) / R["sx"]).mean(0)
    return {"nll": -float(np.mean(d["log_prob_true"])), "mean_sigma": float(rms[0]), "mean_rho": float(rms[1]),
            "sd_sigma": float(we[0]), "sd_rho": float(we[1]), "tarp": float(np.mean(np.abs(ecp - a)))}


def beaten(d, rule):
    """d = model - best over repeats (positive = model worse). True if the model is clearly beaten."""
    m, s, R = d.mean(), d.std(ddof=1), len(d)
    if rule == "strict":
        return abs(m) > s and int((d > 0).sum()) == R
    p1 = stats.ttest_1samp(d, 0.0, alternative="greater").pvalue
    return p1 < 0.0445 if rule == "t045" else (m > 0 and 2 * p1 < 0.05)


def bold_for(vals, key, rule, tag, groups=GROUPS, sizes=SIZES):
    """vals: {(label, n): {metric: per-repeat array}} -> {(label, n): bool}; prints every comparison.
    groups / sizes default to this table's; scripts/make_closeout_tables.py passes its own (2026-11-05)."""
    bold = {}
    for n in sizes:
        labs = [lab for _, rows in groups for lab, _ in rows if (lab, n) in vals]
        best = min(labs, key=lambda l: vals[(l, n)][key].mean())
        for l in labs:
            x = vals[(l, n)][key]
            d = x - vals[(best, n)][key]
            if l == best:
                bold[(l, n)] = True
                continue
            bold[(l, n)] = not beaten(d, rule)
            others = {r: not beaten(d, r) for r in ("strict", "t045", "t05")}
            loose = abs(d.mean()) <= d.std(ddof=1)
            p1 = stats.ttest_1samp(d, 0.0, alternative="greater").pvalue
            pw = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0   # two-sided signed-rank check (no normality)
            print(f"[{tag}] n={n:5d} R={len(d):2d} {l:36s} {x.mean():+.3f}+-{x.std(ddof=1):.3f}  vs best "
                  f"({best}): {d.mean():+.3f}+-{d.std(ddof=1):.3f} ({int((d > 0).sum())}/{len(d)} worse, one-sided p "
                  f"{p1:.3g}, two-sided {min(1, 2 * p1) if d.mean() > 0 else 1:.3g}, Wilcoxon {pw:.3g})  bold[strict/t045/t05/loose]={[int(others[r]) for r in ('strict', 't045', 't05')] + [int(loose)]}")
    return bold


def rows_for(vals, key, bold, signed, groups=GROUPS, sizes=SIZES, mark=None):
    """mark: optional {(label, n): suffix} appended after a cell (default none; used by make_closeout_tables.py)."""
    fmt = (lambda v: f"{v:+.3f}") if signed else (lambda v: f"{v:.3f}")
    cell = lambda l, n: ("--" if (l, n) not in vals else
                         (lambda t: f"{{\\boldmath{t}}}" if bold[(l, n)] else t)(
                             f"${fmt(vals[(l, n)][key].mean())}\\pm{vals[(l, n)][key].std(ddof=1):.3f}$")
                         + (mark or {}).get((l, n), ""))
    L = []
    for gi, (g, rows) in enumerate(groups):
        if gi:
            L.append("\\midrule")
        L.append(f"\\multicolumn{{{len(sizes) + 1}}}{{l}}{{\\emph{{{g}}}}} \\\\")
        L += [" & ".join([lab] + [cell(lab, n) for n in sizes]) + " \\\\" for lab, _ in rows]
    return L


DAGGER = "$^\\dagger$"


def wilcoxon_p(d):
    return stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0


def wilcoxon_marks(vals, key, bold, tag, groups=GROUPS, sizes=SIZES):
    """{(label, n): dagger} where a two-sided Wilcoxon signed-rank test on d = row - best (beaten iff p < 0.05 and
    median d > 0) disagrees with the bolding (2026-11-05). n <= 100 only: with R = 5 (n >= 500) the minimum two-sided
    Wilcoxon p is 0.0625, so it can never reject. Same best (lowest mean) as bold_for."""
    mark = {}
    for n in sizes:
        if n > 100:
            continue
        labs = [lab for _, rows in groups for lab, _ in rows if (lab, n) in vals]
        best = min(labs, key=lambda l: vals[(l, n)][key].mean())
        for l in labs:
            if l == best:
                continue
            d = vals[(l, n)][key] - vals[(best, n)][key]
            beaten_w = wilcoxon_p(d) < 0.05 and np.median(d) > 0
            if beaten_w == bold[(l, n)]:     # bold = not beaten; disagreement iff the two booleans are equal
                mark[(l, n)] = DAGGER
                print(f"[dagger {tag}] n={n:5d} {l:42s} bold={bold[(l, n)]} Wilcoxon p {wilcoxon_p(d):.3g} "
                      f"median d {np.median(d):+.3f}")
    return mark


def load_vals(R, low_repeats, groups=GROUPS, sizes=SIZES, ref_x=None):
    """{(label, n): {metric: per-repeat array}} for every (row, size) whose files all exist. A row's stem may be a
    str or a dict {n: stem} (per-size stem; a size missing from the dict is skipped). Training subsample per
    (n, repeat) asserted identical across all rows (and across calls sharing ref_x)."""
    vals, ref_x = {}, ({} if ref_x is None else ref_x)
    for _, rows in groups:
        for lab, stem in rows:
            for n in sizes:
                s = stem.get(n) if isinstance(stem, dict) else stem
                if s is None:
                    continue
                reps = range(low_repeats if n <= 100 else 5)
                files = [FINAL / f"{s}_n{n}_r{r}.npz" for r in reps]
                if not all(f.exists() for f in files):
                    continue
                per = []
                for r, f in zip(reps, files):
                    with np.load(f, allow_pickle=True) as d:
                        x = np.asarray(d["X_train"])
                        assert np.array_equal(ref_x.setdefault((n, r), x), x), f"subsample mismatch {f.name}"
                        per.append(fit_metrics(d, R))
                vals[(lab, n)] = {k: np.array([p[k] for p in per]) for k in per[0]}
    return vals


# ---- training/validation splits of the hyperparameter searches ----
N_REPEATS = 3

SUBSAMPLE_BASE_SEED = 123

VAL_SIZE = 1000


def build_repeat_splits(X_train, theta_train, n):
    """N_REPEATS independent (train, val) splits, precomputed once and reused unchanged across
    every BO trial for this size -- mirrors optimize_temperature.py's build_repeat_splits exactly.
    Xt/yt is the FULL n-sized subsample (bit-identical to data.subsample_train(..., repeat=k,
    base_seed=SUBSAMPLE_BASE_SEED)), Xv/yv is a disjoint VAL_SIZE-point slice of the training
    pool's leftover rows.

    Edge case (2026-09-04): at n == len(X_train) (n_train=10000, the whole pool), there are no
    leftover rows at all -- falls back to an inner 80/20 split of the n-sized subsample itself
    (the OLD, pre-fix behavior) for that size only, printing a warning so this is visible in the
    job log rather than silently happening."""
    pool_size = len(X_train)
    splits = []
    for k in range(N_REPEATS):
        perm = np.random.default_rng(SUBSAMPLE_BASE_SEED + k).permutation(pool_size)
        train_idx = perm[:n]
        leftover = pool_size - n
        if leftover <= 0:
            print(f"  [warn] n_train={n} consumes the entire {pool_size}-row pool -- no leftover "
                  f"rows for a real held-out validation slice. Falling back to an inner 80/20 "
                  f"split of the n_train subsample itself for repeat={k} (the pre-2026-09-04 "
                  f"behavior) -- this repeat's validation set is NOT independent of training data "
                  f"the way the other sizes' repeats are.", flush=True)
            n_val = max(1, int(0.2 * n))
            val_idx = train_idx[-n_val:]
            train_idx = train_idx[:-n_val]
        else:
            val_idx = perm[n:n + min(VAL_SIZE, leftover)]
        Xt, yt = X_train[train_idx], theta_train[train_idx]
        Xv, yv = X_train[val_idx], theta_train[val_idx]
        splits.append((Xt, yt, Xv, yv))
    return splits
