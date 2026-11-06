"""Evaluation utilities shared across all three models: predictive sampling, TARP/SBC calibration
statistics, and the train/val splitter used by both the BNN's pretraining stage and the NPE."""
import numpy as np
import jax
import jax.numpy as jnp


def split(X, y, seed):
    """80/20 train/val split."""
    idx = np.random.default_rng(seed).permutation(len(X))
    n = int(.8 * len(X))
    return X[idx[:n]], y[idx[:n]], X[idx[n:]], y[idx[n:]]


def sample_gaussian_predictive(key, mu, s, corr, n_draws):
    """n_draws samples per test point from a per-point bivariate Gaussian predictive."""
    M = mu.shape[0]
    z = jax.random.normal(key, (n_draws, M, 2))
    L21 = corr[None] * s[None, :, 1]
    L22 = s[None, :, 1] * jnp.sqrt(jnp.maximum(1.0 - corr[None]**2, 1e-8))
    x0 = mu[None, :, 0] + s[None, :, 0] * z[..., 0]
    x1 = mu[None, :, 1] + L21 * z[..., 0] + L22 * z[..., 1]
    return jnp.stack([x0, x1], axis=-1)


def sample_posterior_predictive(key, mu, s, corr):
    """Same as sample_gaussian_predictive but mu/s/corr already carry a leading ensemble/draw axis K
    (one bivariate Gaussian component per posterior draw, e.g. BNN chain samples)."""
    K, M, _ = mu.shape
    z = jax.random.normal(key, (K, M, 2))
    L21 = corr * s[..., 1]
    L22 = s[..., 1] * jnp.sqrt(jnp.maximum(1.0 - corr**2, 1e-8))
    x0 = mu[..., 0] + s[..., 0] * z[..., 0]
    x1 = mu[..., 1] + L21 * z[..., 0] + L22 * z[..., 1]
    return jnp.stack([x0, x1], axis=-1)


def compute_tarp_f_stats(samples, theta, lo, hi, seeds):
    """samples: (K, n_test, 2) posterior draws. theta: (n_test, 2) true values.
    Returns f: (len(seeds), n_test) — TARP rank statistic per test point per reference-draw seed."""
    s = (np.asarray(samples) - lo) / (hi - lo)
    th = (np.asarray(theta) - lo) / (hi - lo)
    K, M, _ = s.shape
    f_all = np.empty((len(seeds), M))

    for i, sd in enumerate(seeds):
        r = np.random.default_rng(sd)
        ref = r.uniform(size=(M, 2))
        d_s = np.linalg.norm(s - ref[None], axis=-1)
        d_th = np.linalg.norm(th - ref, axis=-1)
        f_all[i] = (d_s < d_th[None]).mean(axis=0)

    return f_all


def tarp_curve_from_f(f_list, n_alpha=50, n_boot=200, boot_seed=0):
    """f_list: list of (n_seeds, n_test) arrays — one per repeat, or just one repeat.
    Pools test points across the list (per seed), returns (alpha, ecp_avg, bootstrap_std)."""
    pooled = np.concatenate(f_list, axis=1)
    n_seeds = pooled.shape[0]
    alpha = np.linspace(0, 1, n_alpha + 1)

    ecp_per_seed = np.stack([
        (pooled[i][None, :] <= alpha[:, None]).mean(axis=1) for i in range(n_seeds)
    ])
    ecp_avg = ecp_per_seed.mean(axis=0)

    r = np.random.default_rng(boot_seed)
    f0 = pooled[0]
    M = f0.shape[0]
    boots = np.empty((n_boot, len(alpha)))
    for b in range(n_boot):
        fb = f0[r.integers(0, M, M)]
        boots[b] = (fb[None, :] <= alpha[:, None]).mean(axis=1)

    return alpha, ecp_avg, boots.std(axis=0)


def bivariate_normal_logpdf(x, mu, s, corr):
    """x, mu, s: (...,2) broadcastable. corr: (...,) broadcastable. Returns (...,) log-density."""
    one_minus_corr2 = jnp.clip(1.0 - corr**2, 1e-6, None)
    z0 = (x[..., 0] - mu[..., 0]) / s[..., 0]
    z1 = (x[..., 1] - mu[..., 1]) / s[..., 1]
    quad = (z0**2 + z1**2 - 2 * corr * z0 * z1) / one_minus_corr2
    log_det = jnp.log(s[..., 0]) + jnp.log(s[..., 1]) + 0.5 * jnp.log(one_minus_corr2)
    return -log_det - 0.5 * quad - jnp.log(2 * jnp.pi)


def sbc_marginal_rank(samples, theta_true):
    """samples: (L, n_test, 2) posterior draws. theta_true: (n_test, 2). Returns (n_test, 2) ranks in [0, L]."""
    return np.asarray((np.asarray(samples) < theta_true[None]).sum(axis=0))
