"""Recreates the right-hand panel of Figure 1 in Delaunoy, de la Brassinne Bonardeaux,
Mishra-Sharma & Louppe (2024), "Low-Budget Simulation-Based Inference with Bayesian Neural
Networks" (arXiv:2408.15136) -- "expected coverage of the prior Bayesian model average with the
tuned prior and normal priors for varying standard deviations" -- for THIS project's own network
and optimized prior (legacy/optimize_prior.py's output), instead of their SLCP benchmark.

Only two prior curves are drawn, not the paper's full 5-way sigma sweep
(N(0,1^2), N(0,0.1^2), N(0,0.01^2), N(0,0.05^2), N(0,0.001^2), tuned): the tuned prior p(w|phi) and
this project's own actual default prior_family='standardnormal' == N(0, 1^2), since that's the one
baseline directly relevant to this project's own default BdeRegressor configuration -- not an
omission, a deliberate scope match to what this project actually compares against elsewhere.

METHODOLOGY -- verified directly against the paper's own PDF text (Section 2 Eq (1),
Section 3 pages 3-5):
    EC(p_hat, alpha) = E_{p(theta,x)}[ 1(theta in Theta_{p_hat}(alpha)) ]
i.e. the fraction of the time the true theta falls inside the level-alpha highest-posterior-density
credible region of the approximate posterior p_hat(theta|x), averaged over (theta,x) drawn from the
TRUE joint (prior x simulator) -- calibrated iff EC(alpha)=alpha for all alpha. Estimated here via
this project's existing metrics.compute_tarp_f_stats/tarp_curve_from_f (Lemos et al. 2023's TARP
rank statistic), already used throughout this project's other calibration plots.

================================ ASSUMPTIONS FLAGGED ============ ================================
  B1. TARP is a DIFFERENT ESTIMATOR of the same EC quantity above (a random-reference-point rank
      statistic rather than an explicit HPD-region computation) -- the pages of the paper read
      don't state which estimator THEY used for Figure 1 (may be in an unread appendix), so this
      plot targets the same quantity but is not guaranteed numerically identical to theirs.
  B2. Number of weight samples per prior (K, default 1000) and number of fresh evaluation (theta,x)
      pairs (M_EVAL, default 1000): not stated in the paper's main text (Appendix C, not read,
      may have exact values) -- chosen to match this project's own TARP_N_DRAWS/N_TEST conventions
      used elsewhere (see scripts/bde_correlated_perchain_test.py).
  B3. Input standardization (xmu, xstd): NOT saved by legacy/optimize_prior.py's current output
      schema. Reconstructed here deterministically from the saved seed/m_size by rebuilding the
      IDENTICAL measurement set optimize_phi() built internally (same jax.random.PRNGKey(seed) ->
      split -> build_measurement_set call, confirmed against optimize_prior.py's own source -- not
      a fresh re-estimate). This matters: evaluating the tuned weights against inputs standardized
      differently from what the KL objective fit them against would give a misleading coverage
      curve for the tuned prior specifically (the N(0,1) baseline doesn't depend on this choice at
      all, since it isn't fit to anything).
  B4. The M_EVAL evaluation set is drawn fresh, from a key disjoint from B3's measurement-set
      reconstruction -- avoids evaluating calibration on the same draws the prior was optimized
      against.
  B5. theta standardization (ymu, ystd): fixed analytic population moments (data.SIGMA_MU/STD,
      RHO_MU/STD), matching optimize_prior.py's own A3 exactly -- these are exact population
      constants, not a per-run estimate, so there's no reconstruction-vs-save mismatch risk here.
===================================================================================================

This script COMPUTES only -- matches this project's existing compute-in-script/plot-in-notebook
split (e.g. scripts/bde_correlated_perchain_test.py + notebooks/correlated_bde_perchain_analysis
.ipynb). The actual figure is the first plot in notebooks/correlated_bde_perchain_analysis_optprior
.ipynb (2026-08-28, per explicit request -- placed before that notebook's other per-chain analysis
cells, which read the optimized-prior BDE run's own separate _optprior_perchain.npz file).

Usage: JAX_PLATFORMS=cpu python legacy/prior_coverage_check.py [--m-eval 1000] [--k-weights 1000]
    [--seed 0]
Requires results/bnn_prior_phi.npz (legacy/optimize_prior.py's output) to already exist.
Writes results/prior_coverage.npz.
"""
import argparse
import sys
from pathlib import Path

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # this repository's src/
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling scripts in this folder
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from rbf_sbi import data
from rbf_sbi import metrics
from rbf_sbi import experiment_log
import optimize_prior as op  # forward/decode_head/build_sizes/ACT_FNS/init_phi/build_measurement_set

RESULTS_DIR = RESULTS
PHI_PATH = RESULTS_DIR / "bnn_prior_phi.npz"


def reconstruct_standardization(seed, m_size):
    """B3: exactly reproduces optimize_phi()'s internal xmu/xstd by rebuilding the identical
    measurement set from the saved seed/m_size (same PRNGKey derivation as optimize_prior.py's
    optimize_phi(): key = PRNGKey(seed); mkey, okey = split(key))."""
    key = jax.random.PRNGKey(int(seed))
    mkey, _ = jax.random.split(key)
    X_M, _ = op.build_measurement_set(mkey, int(m_size))
    xmu, xstd = X_M.mean(0), X_M.std(0) + 1e-8
    return xmu, xstd


def sample_predictive(loc_flat, scale_flat, unravel, X_std, key, act_fn, ymu, ystd, k_weights):
    """Draws k_weights weight samples w ~ N(loc_flat, scale_flat) (flat, ravel_pytree order),
    forward-passes each through the network on X_std, decodes to (mu, scale, corr), and draws one
    posterior-predictive theta sample per weight draw per point -- (K, M, 2)/(K, M, 2)/(K, M), the
    same convention bnn.fit_correlated() uses for its pooled MCMC draws, so
    metrics.sample_posterior_predictive applies unchanged."""
    wkey, dkey = jax.random.split(key)
    wkeys = jax.random.split(wkey, k_weights)

    def one_weight(wk):
        eps = jax.random.normal(wk, loc_flat.shape)
        w = unravel(loc_flat + scale_flat * eps)
        raw = op.forward(w, X_std, act_fn)  # (M, 5)
        return op.decode_head(raw, ymu, ystd)  # (mu, scale, corr)

    mu, scale, corr = jax.vmap(one_weight)(wkeys)  # (K, M, 2), (K, M, 2), (K, M)
    theta_samples = metrics.sample_posterior_predictive(dkey, mu, scale, corr)  # (K, M, 2)
    return np.asarray(theta_samples)


def coverage_curve(theta_samples, theta_true, tarp_lo, tarp_hi, tarp_seeds):
    f = metrics.compute_tarp_f_stats(theta_samples, theta_true, tarp_lo, tarp_hi, tarp_seeds)
    return metrics.tarp_curve_from_f([f])  # (alpha, ecp, boot_std)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m-eval", type=int, default=1000)     # B2
    p.add_argument("--k-weights", type=int, default=1000)  # B2
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not PHI_PATH.exists():
        raise FileNotFoundError(f"{PHI_PATH} not found -- run legacy/optimize_prior.py first.")
    d = np.load(PHI_PATH)
    tuned_loc, tuned_scale = np.asarray(d["loc"]), np.asarray(d["scale"])
    sizes = [int(s) for s in d["sizes"]]
    activation = d["activation"].item()
    act_fn = op.ACT_FNS[activation]
    opt_seed, opt_m_size = int(d["seed"]), int(d["m_size"])

    print(f"Loaded tuned prior from {PHI_PATH}: sizes={sizes} activation={activation!r} "
          f"opt_seed={opt_seed} opt_m_size={opt_m_size}")

    xmu, xstd = reconstruct_standardization(opt_seed, opt_m_size)  # B3
    ymu = jnp.array([data.SIGMA_MU, data.RHO_MU])                  # B5
    ystd = jnp.array([data.SIGMA_STD, data.RHO_STD])

    eval_key = jax.random.PRNGKey(opt_seed + 999_983)  # B4: disjoint from the measurement-set key
    X_eval, theta_eval_np = op.build_measurement_set(eval_key, args.m_eval)
    X_eval_std = jnp.asarray((X_eval - xmu) / xstd)
    theta_eval = np.asarray(theta_eval_np)

    tarp_lo, tarp_hi = data.TARP_LO, data.TARP_HI
    tarp_seeds = list(range(20))  # matches TARP_SEEDS elsewhere in this project

    dummy_loc, _ = op.init_phi(sizes, seed=0)  # structure-only; values unused by ravel_pytree's unravel
    flat_dummy, unravel = ravel_pytree(dummy_loc)
    assert flat_dummy.shape == tuned_loc.shape, (
        f"tuned prior's flat shape {tuned_loc.shape} doesn't match architecture sizes={sizes}'s "
        f"flat shape {flat_dummy.shape} -- BNN_HIDDEN_LAYERS mismatch since the prior was optimized?"
    )

    priors = {
        "tuned": (jnp.asarray(tuned_loc), jnp.asarray(tuned_scale)),
        "standard_normal": (jnp.zeros_like(flat_dummy), jnp.ones_like(flat_dummy)),
    }

    results = {}
    for i, (name, (loc_flat, scale_flat)) in enumerate(priors.items()):
        print(f"Sampling {args.k_weights} weight draws from prior={name!r}, "
              f"forward-passing over {args.m_eval} fresh evaluation points...")
        key = jax.random.PRNGKey(args.seed + i)
        theta_samples = sample_predictive(
            loc_flat, scale_flat, unravel, X_eval_std, key, act_fn, ymu, ystd, args.k_weights
        )
        alpha, ecp, boot_std = coverage_curve(theta_samples, theta_eval, tarp_lo, tarp_hi, tarp_seeds)
        coverage_auc = float(np.trapezoid(ecp - alpha, alpha))  # matches the paper's own coverage-AUC (Sec 4, p.5)
        print(f"  {name}: coverage AUC={coverage_auc:+.4f} "
              f"(0=calibrated, +=conservative/underconfident, -=overconfident)")
        results[name] = dict(alpha=alpha, ecp=ecp, boot_std=boot_std, coverage_auc=coverage_auc)

    out_npz = RESULTS_DIR / "prior_coverage.npz"
    np.savez_compressed(
        out_npz,
        alpha=results["tuned"]["alpha"],
        ecp_tuned=results["tuned"]["ecp"], boot_std_tuned=results["tuned"]["boot_std"],
        coverage_auc_tuned=results["tuned"]["coverage_auc"],
        ecp_standard_normal=results["standard_normal"]["ecp"],
        boot_std_standard_normal=results["standard_normal"]["boot_std"],
        coverage_auc_standard_normal=results["standard_normal"]["coverage_auc"],
        m_eval=args.m_eval, k_weights=args.k_weights, seed=args.seed,
        opt_seed=opt_seed, opt_m_size=opt_m_size, sizes=np.array(sizes), activation=activation,
    )
    print(f"Saved: {out_npz}")
    print("Plot lives in notebooks/correlated_bde_perchain_analysis_optprior.ipynb's first "
          "plot cell, not this script -- see this module's docstring.")

    experiment_log.log_run(
        script="legacy/prior_coverage_check.py",
        config=vars(args),
        results={
            "coverage_auc_tuned": results["tuned"]["coverage_auc"],
            "coverage_auc_standard_normal": results["standard_normal"]["coverage_auc"],
        },
        output_path=str(out_npz),
        notes="Recreation of Delaunoy et al. (2024, arXiv:2408.15136) Figure 1's right panel "
              "(expected coverage of the prior BMA) for this project's tuned prior vs. N(0,1) -- "
              "see this script's own module docstring for assumptions flagged (B1-B5).",
    )


if __name__ == "__main__":
    main()
