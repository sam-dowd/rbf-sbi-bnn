"""Offline optimization of a per-weight Gaussian prior (loc, scale) for CorrelatedBdeRegressor's
network, following the functional-prior-matching method of Delaunoy, de la Brassinne Bonardeaux,
Mishra-Sharma & Louppe (2024), "Low-Budget Simulation-Based Inference with Bayesian Neural
Networks" (arXiv:2408.15136). NOT YET RUN (written 2026-08-27 while NERSC's shared partitions were down
system-wide) -- validated only via the tiny toy smoke test at the bottom of this
file, on the login node, CPU only.

GOAL: bde's own weight prior (`bde.sampler.prior`) is a single fixed-scale Normal(0,1) applied to
every weight. This script instead
computes a per-weight (loc, scale) vector such that the network's INDUCED OUTPUT DISTRIBUTION,
before any training, matches -- in KL divergence -- a Gaussian process centered at the TRUE prior
over (sigma, rho) (data.SIGMA_MU/STD, data.RHO_MU/STD). Per the paper's Proposition 1: "The
Bayesian model average of a Gaussian process centered around the prior on the simulator's
parameters is calibrated" -- a zero-data boundary-condition argument, the same self-consistency
idea behind Simulation-Based Calibration (Talts et al. 2018). Note that it is
the OUTPUT/function-space prior that is matched to p(theta); the per-weight (loc, scale) is only
the tunable knob used to reach that target, not itself equal to p(theta) in any direct sense.

Output loc/scale vectors are meant to be passed straight to
`CorrelatedBdeRegressor(prior_family="normal", prior_kwargs={"loc": ..., "scale": ...})` --
confirmed via direct source read of bde.sampler.prior.log_prior_normal that it broadcasts loc/scale
over the FULL flattened weight vector (`ravel_pytree` order) via ordinary array broadcasting, so no
bde subclassing is needed to use an array-valued (not scalar) prior (an earlier claim that this
needed subclass surgery was wrong).

================================ ASSUMPTIONS FLAGGED ============ ================================
arXiv:2408.15136 was consulted only through paraphrased summaries, not a direct read of
the PDF's own equations -- every equation-level claim below rests on that paraphrase, not on having
actually seen the raw math. Re-verify against the paper directly before trusting a real run's
numbers. None of A1-A8 below are confirmed against the paper's own Appendix unless stated:

  A1. f(theta,x) := log N(theta; mu_w(x), Sigma_w(x)), using CorrelatedBdeRegressor's own fitted
      bivariate-Gaussian head (metrics.bivariate_normal_logpdf). The paper's own f is a
      normalizing-flow's log_prob evaluated at an arbitrary queried theta; our regression head only
      emits one Gaussian's moments per x, so this is our own adaptation, not something the paper itself needed or verifies.
  A2. Measurement set M is drawn from the TRUE joint simulator (data.simulate_dataset) -- i.e.
      (theta_i, x_i) pairs are correlated draws from the real generative process, not the paper's
      stated "i.i.d. uniform over the marginal supports of theta and x independently". Chosen
      because x's support (valid empirical-covariance matrices) has no simple product measure to
      sample from directly, unlike theta's analytically-known support. Not verified to be harmless.
  A3. Standardization: real fits (src/bnn.py's CorrelatedBdeRegressor path, via `bnn._standardize`)
      standardize X and y using a specific training subsample's realized mean/std. This script runs
      before any specific n_train subsample exists, so it standardizes theta using the FIXED
      analytic population moments (data.SIGMA_MU/STD, RHO_MU/STD -- exact, since those ARE the true
      prior, not a sample estimate) and standardizes X using THIS script's own measurement set M's
      empirical mean/std as a stand-in, since X's marginal distribution doesn't depend on n_train
      (fixed simulator). May not exactly match a specific real run's realized subsample statistics.
  A4. GP kernel lengthscales (theta-space and x-space RBF): 0.1-quantile of pairwise squared
      distances within M, per the paper's Appendix A as read. Implemented directly as read.
  A5. GP kernel AMPLITUDE ("sigma = C/2 where C is described only as 'the prior constant'" in the
      fetched Appendix A excerpt) was never resolved to a computable definition .
      Substituted here: the empirical variance of the target log-density log p(theta_i) over M,
      used as the GP's diagonal variance. A genuine guess, not a paper-derived number.
  A6. SSGE hyperparameters: see src/ssge.py's own assumption flags (bandwidth/eigenfunction-count
      defaults were tuned against a local toy Gaussian check, not the paper's Appendix).
  A7. Weight-prior family: independent per-weight Gaussians, reparameterized as
      w = loc + softplus(raw_scale) * epsilon, epsilon ~ N(0,1) -- matches the paper's stated
      variational family. Optimized with Adam (optax), matching this project's own convention
      elsewhere (e.g. bnn.py's pretraining optimizer) -- the paper's own optimizer/LR for THIS
      prior-fitting stage specifically (as opposed to their downstream VI training stage, which WAS
      confirmed to use temperature-scaled ELBO + Adam-style reparameterized gradients) was not
      resolved from what was read.
  A8. Network architecture (sizes = [n_features] + hidden_layers + [output_dim], list of (W,b)
      tuples, matching bde.models.models.Fnn/bde_builder.py's `_build_full_sizes`/`get_model`
      exactly) is manually replicated here, not imported from bde directly, so that this script has
      no import-time dependency on bde's internals -- confirmed correct via direct source read
      (2026-08-27), not an assumption, but a manual reproduction rather than an import means a
      future bde version bump could silently desync it (same caveat as CorrelatedBdeRegressor's
      other overrides in src/bnn.py -- check bde_builder.py directly if this ever misbehaves).
  A9. `MIN_SIGMA = 1e-6` below is read directly from `BivariateGaussianNLL.__init__`'s default in
      src/bnn.py (confirmed, not assumed) -- would need updating here if that default ever changes.
  A10. n_weight_samples=256 (was 64), n_query=4 (was 16) -- 2026-09-03 fix, raised directly after
      the entropy term's loss history (results/bnn_prior_phi.npz's saved `history`, the last real
      run) showed no visible convergence over 2000 steps (bouncing ~0.2-11.6 throughout, final
      value not near the run's own minimum). Root-caused to src/ssge.py: kl_loss's entropy term
      calls ssge.estimate_score(f_samples) where f_samples has shape (n_weight_samples, n_query) --
      i.e. SSGE was asked to estimate a `n_query`-dimensional score from only `n_weight_samples`
      samples. ssge.py's OWN module docstring (self-tested against a known-truth toy Gaussian)
      reports relative error ~5-15% at d<=4 vs. ~20-35% at d in [8,16], with its tested sample-count
      range starting at n=128 -- the old (64, 16) call was simultaneously BELOW the lowest tested n
      and in the worst-error d band. New (256, 4): n now well above the tested floor (and n/d went
      from 4x to 64x), d now in the low-error band. This is a reasoned fix based on ssge.py's own
      documented error characterization, not a paper-derived choice (the paper's own SSGE
      hyperparameters for this stage were never resolved -- see A6/ssge.py's docstring); re-run
      ssge.py's own `if __name__ == "__main__"` self-test at (256, 4) specifically before trusting
      this further, and check whether the loss history actually settles down before treating the
      resulting prior as good. Reducing n_query 16->4 also shrinks the cross-entropy term's own
      per-step Gaussian-likelihood batch (fewer (theta,x) query pairs scored per step) -- a
      real tradeoff against SSGE's lower error, not a free change; not expected to matter much given
      n_steps=2000 draws a fresh random n_query subset every step, but flagged since it wasn't true
      before this change.
===================================================================================================

Usage: JAX_PLATFORMS=cpu python legacy/optimize_prior.py [--m-size 2000] [--n-steps 2000]
    [--n-weight-samples 256] [--n-query 4] [--lr 1e-2] [--seed 0]
Writes results/bnn_prior_phi.npz (loc, scale, sizes, activation, config, final_loss).
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
import optax
from jax.flatten_util import ravel_pytree

from rbf_sbi import data
from rbf_sbi import metrics
import ssge
from rbf_sbi import experiment_log

# ---- Network architecture -- MUST match src/bnn.py's CorrelatedBdeRegressor config exactly
# (A8 above). If BNN_HIDDEN_LAYERS or the correlated head's output width ever change in
# scripts/bde_correlated_perchain_test.py / src/bnn.py, update these to match. ----
N_FEATURES = data.D * data.D            # 100 -- flattened D x D empirical covariance
HIDDEN_LAYERS = [16, 16, 16]            # matches BNN_HIDDEN_LAYERS in the bde_correlated_*.py scripts
OUTPUT_DIM = 5                          # mu_sigma, mu_rho, log_s_sigma, log_s_rho, raw_corr
ACTIVATION = "soft_sign"                # matches CorrelatedBdeRegressor.fit()'s bde_kwargs["activation"]
MIN_SIGMA = 1e-6                        # A9: BivariateGaussianNLL.__init__'s default, src/bnn.py

ACT_FNS = {"soft_sign": jax.nn.soft_sign, "tanh": jnp.tanh, "relu": jax.nn.relu}


def build_sizes():
    return [N_FEATURES] + HIDDEN_LAYERS + [OUTPUT_DIM]


def forward(params, x, act_fn):
    """params: list of (W, b) tuples, x: (..., N_FEATURES) -> (..., OUTPUT_DIM). Matches
    bde.models.models.Fnn.forward exactly (A8)."""
    for W, b in params[:-1]:
        x = act_fn(jnp.dot(x, W) + b)
    W, b = params[-1]
    return jnp.dot(x, W) + b


def decode_head(raw, ymu, ystd):
    """Matches src/bnn.py's _summarize_correlated_raw decode exactly: mu/scale destandardized to
    raw theta units, corr left dimensionless (A9 for MIN_SIGMA)."""
    mu = raw[..., 0:2] * ystd + ymu
    scale = (jax.nn.softplus(raw[..., 2:4]) + MIN_SIGMA) * ystd
    corr = jnp.tanh(raw[..., 4])
    return mu, scale, corr


def init_phi(sizes, seed):
    """phi = (loc, raw_scale), each a list of (W,b) tuples -- same pytree structure `_init_mlp`
    would produce, so ravel_pytree's flattening order matches what bde builds internally (A8).
    loc starts at 0 (not bde's own random N(0,1) init -- this is a PRIOR mean, not a training
    starting point, so 0 is a neutral initial guess, not something derived from the paper).

    raw_scale is initialized PER LAYER, fan-in-scaled (std = 1/sqrt(fan_in)) -- 2026-08-28 fix,
    replacing a flat std=1.0 for every layer regardless of fan-in (matching bde's own current
    fixed-scale default, which was the ORIGINAL rationale for that choice). That flat init put
    weight variance far too large for this architecture's largest layer (fan_in=100, the first
    layer) -- confirmed via a real run (job 57682669) that diverged to NaN, with the KL surrogate
    already ~2.5e20 at step 0, BEFORE any gradient update, pointing at initialization rather than
    training dynamics as the proximate cause: an unscaled first layer produces wildly unstable raw
    network outputs, and since the scale head goes through softplus, a sufficiently negative raw
    output drives the predicted std toward 0, which sends bivariate_normal_logpdf (and the KL
    surrogate loss built on it) toward +-inf. Fan-in-scaled ("LeCun normal"-style) init is the
    standard fix for this class of problem; not itself a paper-specified value -- the paper's own
    weight-prior optimization hyperparameters (Appendix C) were not consulted, see the A1-A9
    assumption block above."""
    del seed  # shapes only depend on `sizes` -- no randomness needed at init
    loc, raw_scale = [], []
    for m, n in zip(sizes[:-1], sizes[1:]):
        loc.append((jnp.zeros((m, n)), jnp.zeros((n,))))
        init_scale = 1.0 / jnp.sqrt(m)                          # fan-in-scaled std
        init_raw_scale = float(jnp.log(jnp.expm1(init_scale)))  # softplus^{-1}(init_scale)
        raw_scale.append((jnp.full((m, n), init_raw_scale), jnp.full((n,), init_raw_scale)))
    return loc, raw_scale


def sample_weights(loc, raw_scale, key):
    """Reparameterized draw w = loc + softplus(raw_scale) * eps (A7)."""
    flat_loc, unravel = ravel_pytree(loc)
    flat_raw_scale, _ = ravel_pytree(raw_scale)
    eps = jax.random.normal(key, flat_loc.shape)
    flat_w = flat_loc + jax.nn.softplus(flat_raw_scale) * eps
    return unravel(flat_w)


def target_log_prior(theta):
    """Exact, NORMALIZED log density of the true one-sided-truncated-normal generative prior over
    (sigma, rho) -- matches data.py's simulate_dataset truncation exactly (lower=0, no upper
    bound). NOTE: this deliberately does NOT reuse data._logprior, which is an unnormalized,
    untruncated stand-in adequate for that module's own grid-posterior use (truncation is handled
    there by the grid's own positive-only bounds) but not accurate enough for a real KL-divergence
    target here."""
    sigma, rho = theta[..., 0], theta[..., 1]

    def trunc_log_normal(x, mu, sd):
        a = (0.0 - mu) / sd
        z = (x - mu) / sd
        log_norm_const = (
            jnp.log(sd) + 0.5 * jnp.log(2 * jnp.pi)
            + jnp.log1p(-jax.scipy.stats.norm.cdf(a))
        )
        return -0.5 * z ** 2 - log_norm_const

    return trunc_log_normal(sigma, data.SIGMA_MU, data.SIGMA_STD) + trunc_log_normal(
        rho, data.RHO_MU, data.RHO_STD
    )


def build_measurement_set(key, m_size):
    """A2: draws M from the true joint simulator, not the paper's stated independent-marginal
    construction."""
    keys = jax.random.split(key, m_size)
    X, theta = jax.vmap(data.simulate_dataset)(keys)
    return np.asarray(X), np.asarray(theta)


def quantile_lengthscale(Z, q=0.1):
    """A4: 0.1-quantile of pairwise squared distances, per the paper's Appendix A as read."""
    n = Z.shape[0]
    d2 = jnp.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
    iu = jnp.triu_indices(n, k=1)
    return jnp.sqrt(jnp.quantile(d2[iu], q) + 1e-12)


def gp_kernel(theta, X, l_theta, l_x, amplitude):
    """K((theta_i,x_i),(theta_j,x_j)) = amplitude * sqrt(RBF_theta) * sqrt(RBF_x), per the paper's
    Appendix A as read (A4/A5). `theta` is RAW (unstandardized) theta; `X` is standardized."""
    dtheta2 = jnp.sum((theta[:, None, :] - theta[None, :, :]) ** 2, axis=-1)
    dx2 = jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    rbf_theta = jnp.exp(-dtheta2 / (2 * l_theta ** 2))
    rbf_x = jnp.exp(-dx2 / (2 * l_x ** 2))
    return amplitude * jnp.sqrt(rbf_theta) * jnp.sqrt(rbf_x)


def kl_loss(loc, raw_scale, theta_M, X_std_M, mu_gp, K_gp, key, n_weight_samples, n_query,
            act_fn, ymu, ystd):
    """Monte Carlo surrogate for KL[p_BNN(f|phi) || p_GP(f|mu_gp,K_gp)]
    = -H[p_BNN(f)] - E_pBNN[log p_GP(f)], per the paper's Eq 10 as read. Only the GRADIENT of
    the returned scalar wrt (loc, raw_scale) is meaningful (via jax.grad on this function) -- the
    value itself is not a usable estimate of the KL divergence, since the entropy term is only
    valid as a surrogate for its gradient (see the note below)."""
    wkey, qkey = jax.random.split(key)
    wkeys = jax.random.split(wkey, n_weight_samples)

    idx = jax.random.choice(qkey, theta_M.shape[0], (n_query,), replace=False)
    theta_q, X_q = theta_M[idx], X_std_M[idx]

    def one_sample(k):
        w = sample_weights(loc, raw_scale, k)
        raw = forward(w, X_q, act_fn)  # (n_query, 5)
        mu, scale, corr = decode_head(raw, ymu, ystd)
        return metrics.bivariate_normal_logpdf(theta_q, mu, scale, corr)  # (n_query,)

    f_samples = jax.vmap(one_sample)(wkeys)  # (n_weight_samples, n_query)

    # Cross-entropy term: E_pBNN[log p_GP(f)], f's target distribution sliced to this step's batch.
    K_batch = K_gp[jnp.ix_(idx, idx)] + 1e-5 * jnp.eye(n_query)
    mu_batch = mu_gp[idx]
    L = jnp.linalg.cholesky(K_batch)

    def log_p_gp(f):
        d = f - mu_batch
        sol = jax.scipy.linalg.cho_solve((L, True), d)
        return -0.5 * jnp.dot(d, sol) - jnp.log(jnp.diag(L)).sum() - 0.5 * n_query * jnp.log(2 * jnp.pi)

    cross_entropy = jnp.mean(jax.vmap(log_p_gp)(f_samples))

    # Entropy term via SSGE's reparameterized-gradient trick: for f = g(phi, eps), one can show
    # d/dphi H[p_phi(f)] = -E_eps[ (grad_f log p_phi(f)) . dg/dphi ] (the score's OWN dependence on
    # phi vanishes in expectation by a standard score-function zero-expectation identity -- this is
    # the paper's Eq 11 as read, restated). So a surrogate loss whose GRADIENT matches
    # d/dphi H[p_BNN(f)] is L_H(phi) = -E_eps[ stop_grad(score(f)) . f ]; only ITS gradient is
    # meaningful, not its value, hence `stop_gradient` on the score and no interpretation of the
    # scalar `h_surrogate` itself as an entropy estimate.
    scores = jax.lax.stop_gradient(ssge.estimate_score(f_samples))  # (n_weight_samples, n_query)
    h_surrogate = -jnp.mean(jnp.sum(scores * f_samples, axis=-1))

    return -h_surrogate - cross_entropy  # surrogate whose gradient matches d(KL)/d(phi)


def optimize_phi(seed=0, m_size=2000, n_steps=2000, n_weight_samples=256, n_query=4,
                  lr=1e-2, gp_amplitude=None, gp_quantile=0.1, log_every=100):
    sizes = build_sizes()
    act_fn = ACT_FNS[ACTIVATION]
    key = jax.random.PRNGKey(seed)
    mkey, okey = jax.random.split(key)

    X_M, theta_M_np = build_measurement_set(mkey, m_size)  # A2
    theta_M = jnp.asarray(theta_M_np)

    # Standardization (A3): theta via fixed analytic population moments (exact); X via this
    # measurement set's own empirical mean/std (a stand-in for a real training subsample's).
    ymu = jnp.array([data.SIGMA_MU, data.RHO_MU])
    ystd = jnp.array([data.SIGMA_STD, data.RHO_STD])
    xmu, xstd = X_M.mean(0), X_M.std(0) + 1e-8
    X_std_M = jnp.asarray((X_M - xmu) / xstd)

    # Target GP (A4 lengthscales, A5 amplitude)
    l_theta = quantile_lengthscale(theta_M, gp_quantile)
    l_x = quantile_lengthscale(X_std_M, gp_quantile)
    mu_gp = jax.vmap(target_log_prior)(theta_M)  # target mean = true prior log-density, raw theta
    if gp_amplitude is None:
        gp_amplitude = float(jnp.var(mu_gp))  # A5
    K_gp = gp_kernel(theta_M, X_std_M, l_theta, l_x, gp_amplitude)

    loc, raw_scale = init_phi(sizes, seed)
    flat0, unravel_phi = ravel_pytree((loc, raw_scale))
    # 2026-08-28: global-norm gradient clipping added as a second safeguard alongside the init_phi
    # fix above -- the same diverged run (job 57682669) still showed wildly varying (not just
    # large-at-init) loss magnitudes across steps 100-1900 before finally NaN-ing at the last step,
    # suggesting ongoing instability during training, not purely a bad starting point. 1.0 is a
    # standard default clip norm, not tuned against this problem specifically (A7 already covers
    # the optimizer choice itself, adam -- this is an addition on top, not a replacement).
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))  # A7
    opt_state = optimizer.init(flat0)

    def loss_fn(flat_params, k):
        loc_, raw_scale_ = unravel_phi(flat_params)
        return kl_loss(loc_, raw_scale_, theta_M, X_std_M, mu_gp, K_gp, k,
                        n_weight_samples, n_query, act_fn, ymu, ystd)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    # 2026-09-04, real bug found and fixed: this loop previously returned the FINAL step's params
    # unconditionally, with no best-checkpoint tracking at all -- checked directly against a real
    # run's own saved history (results/bnn_prior_phi.npz from job 57904957) and found the loss is
    # NOT monotonically decreasing after an initial sharp drop (steps 0-200: 14.29 -> -0.03), then
    # bounces chaotically for the remaining ~1800 steps (only 45% of logged steps show a decrease
    # from the previous one) and NEVER returns to that early minimum -- the saved final-step loss
    # (2.88) was ~90x worse than the best value the run ever passed through. Standard fix (same
    # pattern src/bnn.py's own EarlyStoppingCallback already uses for BNN pretraining elsewhere in
    # this project): track the single lowest-loss step seen and save THOSE params, not whatever the
    # last step happened to be. NOTE: each step's loss is itself a noisy one-batch MC estimate
    # (n_weight_samples draws, a fresh random n_query subset per step) -- "best single step" is a
    # real improvement over "final step" (which this data shows can be arbitrarily bad) but is not
    # a smoothed/robust criterion; a future revision could track a running average instead if this
    # rerun's own history still looks noisy at its selected best point.
    flat_params = flat0
    best_loss, best_flat_params, best_step = float("inf"), flat0, 0
    keys = jax.random.split(okey, n_steps)
    history = []
    for step in range(n_steps):
        loss, grads = grad_fn(flat_params, keys[step])
        updates, opt_state = optimizer.update(grads, opt_state)
        flat_params = optax.apply_updates(flat_params, updates)
        loss_f = float(loss)
        if np.isfinite(loss_f) and loss_f < best_loss:
            best_loss, best_flat_params, best_step = loss_f, flat_params, step
        if step % log_every == 0 or step == n_steps - 1:
            history.append((step, loss_f))
            print(f"step {step:5d}  KL surrogate={loss_f: .4f}  (best so far: "
                  f"{best_loss:.4f} @ step {best_step})")

    final_loss = history[-1][1] if history else None
    print(f"\nBest checkpoint: step {best_step}, loss={best_loss:.4f}  "
          f"(final step {n_steps-1} loss={final_loss:.4f}) -- saving BEST, not final.")

    loc, raw_scale = unravel_phi(best_flat_params)
    flat_loc, _ = ravel_pytree(loc)
    flat_scale, _ = ravel_pytree(jax.tree_util.tree_map(jax.nn.softplus, raw_scale))
    return (np.asarray(flat_loc), np.asarray(flat_scale), sizes, history,
            best_loss, best_step, final_loss)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--m-size", type=int, default=2000)
    p.add_argument("--n-steps", type=int, default=2000)
    p.add_argument("--n-weight-samples", type=int, default=256)  # A10
    p.add_argument("--n-query", type=int, default=4)             # A10
    p.add_argument("--lr", type=float, default=1e-2)
    args = p.parse_args()

    loc, scale, sizes, history, best_loss, best_step, final_loss = optimize_phi(
        seed=args.seed, m_size=args.m_size, n_steps=args.n_steps,
        n_weight_samples=args.n_weight_samples, n_query=args.n_query, lr=args.lr,
    )

    out_dir = RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bnn_prior_phi.npz"
    np.savez_compressed(
        out_path, loc=loc, scale=scale, sizes=np.array(sizes), activation=ACTIVATION,
        seed=args.seed, m_size=args.m_size, n_steps=args.n_steps,
        n_weight_samples=args.n_weight_samples, n_query=args.n_query, lr=args.lr,
        history=np.array(history),
        # 2026-09-04: this file now holds the BEST-observed checkpoint, not the final step -- see
        # optimize_phi's docstring/comment above for the real bug this fixes (final-step loss was
        # ~90x worse than the best checkpoint on the previous run). best_loss/best_step/final_loss
        # kept for provenance so a later reader can see the gap directly without re-deriving it
        # from `history`.
        best_loss=best_loss, best_step=best_step, final_loss=final_loss,
    )
    print(f"\nSaved optimized prior to: {out_path}")
    print(f"loc: shape={loc.shape} mean={loc.mean():.4f} std={loc.std():.4f}")
    print(f"scale: shape={scale.shape} mean={scale.mean():.4f} min={scale.min():.4f} max={scale.max():.4f}")

    experiment_log.log_run(
        script="legacy/optimize_prior.py",
        config=vars(args),
        results={
            "best_kl_surrogate": best_loss, "best_step": best_step, "final_kl_surrogate": final_loss,
            "loc_mean": float(loc.mean()), "loc_std": float(loc.std()),
            "scale_mean": float(scale.mean()), "scale_min": float(scale.min()),
            "scale_max": float(scale.max()),
        },
        output_path=str(out_path),
        notes="Prior-optimization run per Delaunoy et al. (2024, arXiv:2408.15136) -- "
              "see this script's own module docstring for assumptions "
              "flagged (A1-A9) before trusting these numbers. 2026-09-04: now saves the "
              "BEST-observed checkpoint (by KL surrogate loss), not the final step -- the previous "
              "version had no best-checkpoint tracking at all and saved a final-step result that "
              "was, on a real run, ~90x worse than a checkpoint the same run passed through at "
              "step 200 (loss -0.03 vs. final 2.88). See best_step/best_kl_surrogate above.",
    )


if __name__ == "__main__":
    main()
