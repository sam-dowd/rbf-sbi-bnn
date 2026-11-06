"""Prior predictive of the BNN's network without training data: used to choose the data-free weight-prior scale
(scripts/tuning/optimize_prior_width.py) and for its simulation-free check (scripts/tuning/coherence_width_check.py).

Merged, with function bodies unchanged, from the research repository's optimize_prior.py (network constants,
forward pass, head decoding, prior initialisation, measurement set) and prior_coverage_check.py (standardisation,
predictive sampling, coverage curve).
"""
import jax
import jax.numpy as jnp
import numpy as np

from rbf_sbi import data, metrics

N_FEATURES = data.D * data.D

HIDDEN_LAYERS = [16, 16, 16]

OUTPUT_DIM = 5

ACTIVATION = "soft_sign"

MIN_SIGMA = 1e-6

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


def build_measurement_set(key, m_size):
    """A2: draws M from the true joint simulator, not the paper's stated independent-marginal
    construction."""
    keys = jax.random.split(key, m_size)
    X, theta = jax.vmap(data.simulate_dataset)(keys)
    return np.asarray(X), np.asarray(theta)


def reconstruct_standardization(seed, m_size):
    """B3: exactly reproduces optimize_phi()'s internal xmu/xstd by rebuilding the identical
    measurement set from the saved seed/m_size (same PRNGKey derivation as optimize_prior.py's
    optimize_phi(): key = PRNGKey(seed); mkey, okey = split(key))."""
    key = jax.random.PRNGKey(int(seed))
    mkey, _ = jax.random.split(key)
    X_M, _ = build_measurement_set(mkey, int(m_size))
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
        raw = forward(w, X_std, act_fn)  # (M, 5)
        return decode_head(raw, ymu, ystd)  # (mu, scale, corr)

    mu, scale, corr = jax.vmap(one_weight)(wkeys)  # (K, M, 2), (K, M, 2), (K, M)
    theta_samples = metrics.sample_posterior_predictive(dkey, mu, scale, corr)  # (K, M, 2)
    return np.asarray(theta_samples)


def coverage_curve(theta_samples, theta_true, tarp_lo, tarp_hi, tarp_seeds):
    f = metrics.compute_tarp_f_stats(theta_samples, theta_true, tarp_lo, tarp_hi, tarp_seeds)
    return metrics.tarp_curve_from_f([f])
