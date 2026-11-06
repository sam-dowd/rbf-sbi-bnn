"""Simulation-free choice of the isotropic weight-prior width, by prior-predictive coherence (2026-11-05).

Question: the data-free width 0.3655 (scripts/tuning/optimize_prior_width.py) was calibrated on 1,000 paired (theta, x)
simulator draws (plus 2,000 for input standardisation) -- more simulations than n=50/100 training sets. Can the
same width be found from the n training INPUTS and the analytic prior moments alone?

Condition (law of total variance): a coherent posterior map satisfies
    E_x[Var(theta | x)] + Var_x(E[theta | x]) = Var_p(theta).
For the UNTRAINED BNN, weights w ~ N(0, width^2) i.i.d., outputs decoded exactly as src/bnn.py does (mu, s =
softplus(a) + 1e-6), in the pipeline's standardised target units (bnn._standardize on the training subsample):
    mixture ratio R_j(width) = [ mean_{w,x} s_j^2 + var_{w,x} mu_j ] / V*_j,   V*_j = Var_p(theta_j) / ystd_j^2,
with Var_p the analytic truncated-normal prior variance. Solved for R_j = 1 per target and jointly
(argmin sum_j log^2 R_j). Also reported: the per-draw version (median over w of V_w / V*, with
V_w = mean_x s^2 + var_x mu), which is what a per-draw coherence prior term would act on.

The training thetas enter only through ystd (unit conversion, as in the pipeline); the target is the analytic prior
variance. No simulator calls beyond the training subsample, no fitting of the network.

ASSUMPTIONS: (1) network = 100 -> 16 -> 16 -> 16 -> 5, soft_sign, via legacy/optimize_prior.py's forward (which
asserts it matches bde's Fnn); (2) every weight AND bias ~ N(0, width^2) (as in optimize_prior_width.py and the
gaussprior runs); (3) the coherence condition is a different objective from optimize_prior_width.py's TARP coverage
AUC -- agreement between the two is the thing being tested, not assumed.

Usage: JAX_PLATFORMS=cpu python scripts/tuning/coherence_width_check.py [--k-weights 2000] [--seed 0] [--smoke scratch.npz]
Output: results/coherence_width_check.npz
"""
import argparse
import sys
from pathlib import Path

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # this repository's src/
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from scipy.stats import truncnorm

from rbf_sbi import data
from rbf_sbi import experiment_log
from rbf_sbi import prior_predictive as op
from rbf_sbi.bnn import _standardize

DATA_FREE_WIDTH = 0.3655326          # results/prior_width_search.npz best_width
SIZES_N = [50, 100, 500, 1000]
REPEATS = range(5)
WIDTHS = np.geomspace(0.05, 2.0, 81)
OUT = RESULTS / "coherence_width_check.npz"


def prior_var():
    v = []
    for m, s in [(data.SIGMA_MU, data.SIGMA_STD), (data.RHO_MU, data.RHO_STD)]:
        v.append(truncnorm((0 - m) / s, np.inf, loc=m, scale=s).var())
    return np.array(v)


def root_log(widths, ratio):
    """Width where ratio crosses 1, by linear interpolation in (log width, log ratio); nan if no crossing."""
    lr = np.log(ratio)
    idx = np.where(np.diff(np.sign(lr)) != 0)[0]
    if len(idx) == 0:
        return np.nan, 0
    i = idx[0]
    lw = np.log(widths)
    t = -lr[i] / (lr[i + 1] - lr[i])
    return float(np.exp(lw[i] + t * (lw[i + 1] - lw[i]))), len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-weights", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=str, default=None,
                    help="write to this scratch .npz instead, n=50 r0 only, 200 weights, no log entry")
    args = ap.parse_args()
    global SIZES_N, REPEATS, OUT
    if args.smoke:
        SIZES_N, REPEATS, OUT, args.k_weights = [50], range(1), Path(args.smoke), 200

    sizes = op.build_sizes()
    assert sizes == [100, 16, 16, 16, 5], sizes
    act = op.ACT_FNS[op.ACTIVATION]
    loc0, _ = op.init_phi(sizes, seed=0)
    flat0, unravel = ravel_pytree(loc0)
    eps = jax.random.normal(jax.random.PRNGKey(args.seed), (args.k_weights, flat0.size))   # common random numbers

    @jax.jit
    def moments(width, Xs):
        def one(e):
            out = op.forward(unravel(width * e), Xs, act)                    # (n, 5)
            mu = out[:, 0:2]
            s = jax.nn.softplus(out[:, 2:4]) + op.MIN_SIGMA
            return (s ** 2).mean(0), mu.mean(0), (mu ** 2).mean(0)
        es2, m1, m2 = jax.lax.map(one, eps, batch_size=200)                  # each (K, 2)
        mix = es2.mean(0) + (m2.mean(0) - m1.mean(0) ** 2)                   # total variance over (w, x)
        per_draw = es2 + (m2 - m1 ** 2)                                      # V_w, (K, 2)
        return mix, jnp.median(per_draw, axis=0)

    pv = prior_var()
    print(f"truncated prior variances: {pv.round(5)}   K={args.k_weights} weight draws, {len(WIDTHS)} widths")
    X_pool, th_pool, _, _ = data.load_dataset()

    res = {}
    rows = []
    for n in SIZES_N:
        for rep in REPEATS:
            Xb, yb = data.subsample_train(X_pool, th_pool, n, repeat=rep)
            Xs, ys, _, _, ymu, ystd = _standardize(np.asarray(Xb, float), np.asarray(yb, float))
            vstar = pv / np.asarray(ystd) ** 2
            Xs = jnp.asarray(Xs, jnp.float32)
            mix = np.zeros((len(WIDTHS), 2)); med = np.zeros((len(WIDTHS), 2))
            for k, w in enumerate(WIDTHS):
                a, b = moments(jnp.float32(w), Xs)
                mix[k], med[k] = np.asarray(a) / vstar, np.asarray(b) / vstar
            r_s, c_s = root_log(WIDTHS, mix[:, 0]); r_r, c_r = root_log(WIDTHS, mix[:, 1])
            joint = float(WIDTHS[np.argmin((np.log(mix) ** 2).sum(1))])
            pd_s, _ = root_log(WIDTHS, med[:, 0]); pd_r, _ = root_log(WIDTHS, med[:, 1])
            at = lambda wq, M: np.array([np.interp(np.log(wq), np.log(WIDTHS), np.log(M[:, j])) for j in range(2)])
            r_df, r_one = np.exp(at(DATA_FREE_WIDTH, mix)), np.exp(at(1.0, mix))
            res[(n, rep)] = dict(mix=mix, med=med, vstar=vstar)
            rows.append((n, rep, r_s, r_r, joint, pd_s, pd_r, *r_df, *r_one, c_s, c_r))
            print(f"n={n:>4} r{rep}: root sigma {r_s:.4f} rho {r_r:.4f} joint {joint:.4f} | per-draw median root "
                  f"{pd_s:.4f}/{pd_r:.4f} | R at 0.3655 {r_df[0]:.3f}/{r_df[1]:.3f} | R at 1.0 {r_one[0]:.2f}/{r_one[1]:.2f}"
                  f" | crossings {c_s}/{c_r}")

    rows = np.array(rows, float)
    print("\nsummary over repeats (mean +- sd):")
    summary = {}
    for n in SIZES_N:
        R = rows[rows[:, 0] == n]
        f = lambda c: f"{np.nanmean(R[:, c]):.4f} +- {np.nanstd(R[:, c]):.4f}"
        line = (f"root sigma {f(2)}, root rho {f(3)}, joint {f(4)}, per-draw root {f(5)} / {f(6)}, "
                f"R(0.3655) {f(7)} / {f(8)}, R(1.0) {f(9)} / {f(10)}")
        print(f"n={n:>4}: {line}")
        summary[f"n{n}"] = line

    np.savez(OUT, widths=WIDTHS, sizes_n=np.array(SIZES_N), rows=rows,
             row_cols=np.array(["n", "repeat", "root_sigma", "root_rho", "joint", "perdraw_root_sigma",
                                "perdraw_root_rho", "R_df_sigma", "R_df_rho", "R_one_sigma", "R_one_rho",
                                "crossings_sigma", "crossings_rho"]),
             mix=np.stack([res[(n, r)]["mix"] for n in SIZES_N for r in REPEATS]),
             med=np.stack([res[(n, r)]["med"] for n in SIZES_N for r in REPEATS]),
             vstar=np.stack([res[(n, r)]["vstar"] for n in SIZES_N for r in REPEATS]),
             prior_var=pv, k_weights=args.k_weights, seed=args.seed, data_free_width=DATA_FREE_WIDTH)
    print(f"wrote {OUT}")
    if args.smoke:
        return

    experiment_log.log_run(
        script="scripts/tuning/coherence_width_check.py",
        config=dict(k_weights=args.k_weights, seed=args.seed, widths="geomspace(0.05, 2.0, 81)",
                    sizes_n=SIZES_N, repeats="0-4", network="100-16-16-16-5 soft_sign, all params N(0, width^2)",
                    standardisation="bnn._standardize on each training subsample", target="truncated-normal prior var"),
        results=summary, output_path=str(OUT.relative_to(ROOT)),
        notes=("Simulation-free width selection by prior-predictive coherence (law of total variance), compared "
               "with the simulator-calibrated data-free width 0.3655. Analysis/selection only, no network fits."))


if __name__ == "__main__":
    main()
