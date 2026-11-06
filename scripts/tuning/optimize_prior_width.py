"""GP-based Bayesian optimization over the scale of an ISOTROPIC Gaussian weight prior
N(0, width^2) (a single scalar `width` shared by every weight, loc fixed at 0), searching for the
width that best calibrates the zero-data prior-predictive coverage -- the same quantity
legacy/prior_coverage_check.py measures for the two fixed priors it already compares (this
project's default N(0,1) and legacy/optimize_prior.py's per-weight Delaunoy-tuned prior).

This generalizes Delaunoy, de la Brassinne Bonardeaux, Mishra-Sharma & Louppe (2024)
(arXiv:2408.15136) Figure 1's right panel -- which sweeps a small FIXED grid of scalar widths
(1, 0.1, 0.05, 0.01, 0.001) against their tuned prior -- into a continuous GP-BO search over width,
using the exact same coverage-AUC objective (Sec 4, p.5): the paper's own Figure 1 motivates this
sweep as "no fixed N(0,sigma^2) width is calibrated across sigma" -- this script asks "what's the
single best such width for THIS network/task," not whether a fixed-width family can match the
tuned (per-weight, non-isotropic) prior's own calibration.

================================ ASSUMPTIONS FLAGGED ============ ================================
  C1. Objective = |coverage_auc| (absolute value of trapz(ecp - alpha, alpha), scripts/
      prior_coverage_check.py's own metric) -- zero is perfectly calibrated, so this treats
      over- and under-confident widths symmetrically. Not a paper-specified BO objective (the
      paper never runs BO over width at all, see module docstring above) -- a natural choice given
      what's already computed elsewhere in this project, not derived from anywhere else.
  C2. Search space Real(1e-3, 3.0, prior="log-uniform") -- covers the paper's own swept grid
      (0.001-1) with headroom above 1 in case this project's network/task differs; log-uniform
      because (like lr/weight_decay elsewhere in this project) the relevant scale spans orders of
      magnitude, not a paper-derived range.
  C3. Reuses legacy/prior_coverage_check.py's exact standardization/evaluation-set reconstruction
      (B3/B4/B5 in that script's own docstring) so all three priors in the eventual comparison
      (N(0,1), Delaunoy-tuned, this script's optimized width) are scored on IDENTICAL (X,theta)
      evaluation draws -- requires results/prior_coverage.npz to already exist (run
      legacy/prior_coverage_check.py first) for its saved opt_seed/opt_m_size/m_eval/k_weights.
  C4. No repeats per BO trial (unlike scripts/optimize_temperature.py's N_REPEATS=3) -- TARP's own
      internal bootstrap (metrics.tarp_curve_from_f, n_boot=200) already estimates uncertainty on
      one evaluation, and prior_coverage_check.py itself only ever computes one draw per prior;
      matching that convention rather than adding new noise-averaging machinery.
===================================================================================================

Usage: JAX_PLATFORMS=cpu python scripts/tuning/optimize_prior_width.py [--n-trials 40] [--seed 0]
Requires results/prior_coverage.npz (legacy/prior_coverage_check.py's output) to already exist.
Writes results/prior_width_search.npz (BO trial history, best_width, and a final TARP curve at
best_width computed with the same m_eval/k_weights as prior_coverage_check.py, ready to plot
alongside that script's own saved curves).
"""
import argparse
import sys
import time
from pathlib import Path

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # this repository's src/
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from skopt import Optimizer
from skopt.space import Real

from rbf_sbi import data
from rbf_sbi import metrics
from rbf_sbi import experiment_log
from rbf_sbi import prior_predictive as op  # forward/decode_head/build_sizes/ACT_FNS/init_phi/build_measurement_set
from rbf_sbi import prior_predictive as pcc  # reconstruct_standardization/sample_predictive/coverage_curve

RESULTS_DIR = RESULTS
COVERAGE_PATH = RESULTS_DIR / "prior_coverage.npz"
OUT_PATH = RESULTS_DIR / "prior_width_search.npz"

WIDTH_BOUNDS = (1e-3, 3.0)  # C2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not COVERAGE_PATH.exists():
        raise FileNotFoundError(
            f"{COVERAGE_PATH} not found -- run legacy/prior_coverage_check.py first "
            "(this script reuses its saved standardization/evaluation-set config, C3)."
        )
    cov = np.load(COVERAGE_PATH)
    opt_seed, opt_m_size = int(cov["opt_seed"]), int(cov["opt_m_size"])
    m_eval, k_weights = int(cov["m_eval"]), int(cov["k_weights"])
    sizes = [int(s) for s in cov["sizes"]]
    activation = cov["activation"].item()
    act_fn = op.ACT_FNS[activation]

    print(f"Reusing prior_coverage_check.py's config: opt_seed={opt_seed} opt_m_size={opt_m_size} "
          f"m_eval={m_eval} k_weights={k_weights} sizes={sizes} activation={activation!r}")

    # C3: identical standardization + evaluation set to prior_coverage_check.py's, so all three
    # priors' TARP curves are scored on the same (X, theta) draws.
    xmu, xstd = pcc.reconstruct_standardization(opt_seed, opt_m_size)
    ymu = jnp.array([data.SIGMA_MU, data.RHO_MU])
    ystd = jnp.array([data.SIGMA_STD, data.RHO_STD])
    eval_key = jax.random.PRNGKey(opt_seed + 999_983)
    X_eval, theta_eval_np = op.build_measurement_set(eval_key, m_eval)
    X_eval_std = jnp.asarray((X_eval - xmu) / xstd)
    theta_eval = np.asarray(theta_eval_np)

    tarp_lo, tarp_hi = data.TARP_LO, data.TARP_HI
    tarp_seeds = list(range(20))  # matches prior_coverage_check.py's own TARP_SEEDS convention

    dummy_loc, _ = op.init_phi(sizes, seed=0)
    flat_dummy, unravel = ravel_pytree(dummy_loc)

    def coverage_auc_at_width(width, key):
        loc_flat = jnp.zeros_like(flat_dummy)
        scale_flat = jnp.full_like(flat_dummy, width)
        theta_samples = pcc.sample_predictive(
            loc_flat, scale_flat, unravel, X_eval_std, key, act_fn, ymu, ystd, k_weights
        )
        alpha, ecp, boot_std = pcc.coverage_curve(theta_samples, theta_eval, tarp_lo, tarp_hi, tarp_seeds)
        auc = float(np.trapezoid(ecp - alpha, alpha))
        return auc, (alpha, ecp, boot_std)

    space = [Real(*WIDTH_BOUNDS, prior="log-uniform", name="width")]
    opt = Optimizer(space, base_estimator="GP", acq_func="EI", random_state=args.seed)

    trials = []  # (width, auc, abs_auc, elapsed_sec)
    t0 = time.time()
    for trial_idx in range(args.n_trials):
        (width,) = opt.ask()
        trial_t0 = time.time()
        key = jax.random.PRNGKey(args.seed + 1000 + trial_idx)  # C4: one draw, no repeats
        auc, _ = coverage_auc_at_width(width, key)
        obj = abs(auc)  # C1
        opt.tell([width], obj)
        elapsed = time.time() - trial_t0
        trials.append((width, auc, obj, elapsed))
        print(f"trial={trial_idx + 1:>3} width={width:.5f} -> coverage_auc={auc:+.4f} "
              f"|auc|={obj:.4f} ({elapsed:.1f}s)", flush=True)

    best_idx = int(np.argmin(opt.yi))
    best_width = float(opt.Xi[best_idx][0])
    best_obj = float(opt.yi[best_idx])
    total_elapsed = time.time() - t0
    print(f"\nDone: {args.n_trials} trials in {total_elapsed:.0f}s | "
          f"best width={best_width:.5f} |coverage_auc|={best_obj:.4f}")

    # Final clean curve at best_width, using the SAME per-prior key convention prior_coverage_
    # check.py uses (seed+2, since that script's own two priors use seed+0/seed+1) -- so this curve
    # slots in as the "third prior" under that file's own indexing scheme.
    final_key = jax.random.PRNGKey(int(cov["seed"]) + 2)
    best_auc, (alpha, ecp, boot_std) = coverage_auc_at_width(best_width, final_key)
    print(f"Final curve at best width: coverage_auc={best_auc:+.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        width_bounds=np.array(WIDTH_BOUNDS),
        trial_widths=np.array([t[0] for t in trials]),
        trial_auc=np.array([t[1] for t in trials]),
        trial_abs_auc=np.array([t[2] for t in trials]),
        trial_elapsed_sec=np.array([t[3] for t in trials]),
        n_trials=args.n_trials, elapsed_sec=total_elapsed, seed=args.seed,
        best_width=best_width, best_abs_auc=best_obj,
        alpha=alpha, ecp=ecp, boot_std=boot_std, coverage_auc=best_auc,
        m_eval=m_eval, k_weights=k_weights,
        opt_seed=opt_seed, opt_m_size=opt_m_size, sizes=np.array(sizes), activation=activation,
    )
    print(f"Saved: {OUT_PATH}")

    experiment_log.log_run(
        script="scripts/tuning/optimize_prior_width.py",
        config=dict(
            n_trials=args.n_trials, seed=args.seed, width_bounds=WIDTH_BOUNDS,
            m_eval=m_eval, k_weights=k_weights, opt_seed=opt_seed, opt_m_size=opt_m_size,
        ),
        results=dict(best_width=best_width, best_abs_auc=best_obj, coverage_auc=best_auc),
        output_path=str(OUT_PATH),
        notes="GP-BO search over an isotropic N(0,width^2) weight-prior width, generalizing "
              "Delaunoy et al. (2024, arXiv:2408.15136) Figure 1's fixed-grid width sweep -- "
              "see this script's own module docstring for assumptions flagged (C1-C4).",
    )


if __name__ == "__main__":
    main()
