"""Dataset generation and exact-posterior calculation for the RBF-kernel covariance-recovery problem.

A "dataset" here is the empirical D x D covariance of N draws from a mean-zero Gaussian process with
an RBF kernel of scale sigma and lengthscale rho, plus eta^2 observation noise. The inference target
is theta = (sigma, rho); the feature fed to every model is the flattened empirical covariance.
"""
from pathlib import Path

from rbf_sbi.paths import DATA

import jax
import jax.numpy as jnp
import numpy as np

# ---- Fixed problem definition (not swept over) ----
D = 10                      # covariance dimension
N = 10                      # draws per simulated dataset
ETA = 0.05                  # observation noise std
T = jnp.linspace(0.0, 1.0, D)

SIGMA_MU, SIGMA_STD = 1.25, 0.3
RHO_MU, RHO_STD = 0.60, 0.2

SIGMA_GRID = jnp.linspace(0.05, 2.9, 240)
RHO_GRID = jnp.linspace(0.02, 1.6, 240)
TARP_LO = np.array([float(SIGMA_GRID[0]), float(RHO_GRID[0])])
TARP_HI = np.array([float(SIGMA_GRID[-1]), float(RHO_GRID[-1])])


def simulate_dataset(key):
    """Draw one (sigma, rho) pair and the empirical covariance of N samples from its RBF-kernel GP.
    Returns (feature, theta) — feature is the flattened D x D empirical covariance (length D*D)."""
    k1, k2, kx = jax.random.split(key, 3)

    sigma = SIGMA_MU + SIGMA_STD * jax.random.truncated_normal(
        k1, lower=(0.0 - SIGMA_MU) / SIGMA_STD, upper=jnp.inf
    )
    rho = RHO_MU + RHO_STD * jax.random.truncated_normal(
        k2, lower=(0.0 - RHO_MU) / RHO_STD, upper=jnp.inf
    )

    K = sigma**2 * jnp.exp(-(T[:, None] - T[None, :])**2 / (2 * rho**2))
    X = jax.random.multivariate_normal(kx, jnp.zeros(D), K + ETA**2 * jnp.eye(D), (N,))

    return (X.T @ X / N).reshape(-1), jnp.array([sigma, rho])


def generate_train_test(key, n_train, n_test):
    """vmaps simulate_dataset over n_train + n_test draws and splits the result.

    CAVEAT (confirmed empirically 2026-08-25): jax.random.split(key,
    n_train + n_test)'s output depends on the TOTAL count, so a different n_train produces a
    completely different X_test/theta_test even with the same key/n_test -- the test set is only
    identical across calls sharing the same (key, n_train, n_test) all three. This is exactly why
    the project now has ONE canonical dataset (build_and_save_dataset/load_dataset below) that
    every script/notebook reads, rather than each calling this function independently with
    whatever n_train it happens to need -- do not call this function directly outside of
    build_and_save_dataset; use load_dataset() + subsample_train() instead."""
    keys = jax.random.split(key, n_train + n_test)
    features, theta = jax.vmap(simulate_dataset)(keys)
    return features[:n_train], theta[:n_train], features[n_train:], theta[n_train:]


# ---- Canonical dataset: generated ONCE, saved to data/dataset.npz, loaded by every downstream
# script/notebook instead of each calling generate_train_test independently. This is what makes
# "same test set, and training data that's a nested subsample of one shared pool, for every
# model/size/script" an actual invariant rather than an assumption -- this replaces the 2026-08-25
# inconsistency (different scripts independently calling
# generate_train_test with different n_train were silently getting different X_test each, per the
# caveat above). Matches scripts/run_experiment.py's original data.generate_train_test(PRNGKey(0),
# 10_000, 1000) call and its subsample convention (np.random.default_rng(123 + repeat)) exactly --
# this factors those out to one place instead of leaving every other script to reimplement them.
DATA_DIR = DATA
DATASET_PATH = DATA_DIR / "dataset.npz"


def build_and_save_dataset(seed=0, n_train_pool=10_000, n_test=1000, path=DATASET_PATH):
    """Generates the canonical dataset and writes it to `path`. Re-run only if the seed/pool size
    is deliberately changed -- every other script/notebook should call load_dataset(), never this."""
    key = jax.random.PRNGKey(seed)
    X_train, theta_train, X_test, theta_test = generate_train_test(key, n_train_pool, n_test)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / (path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp_path, seed=seed, n_train_pool=n_train_pool, n_test=n_test,
        X_train=np.asarray(X_train), theta_train=np.asarray(theta_train),
        X_test=np.asarray(X_test), theta_test=np.asarray(theta_test),
    )
    tmp_path.rename(path)
    return X_train, theta_train, X_test, theta_test


def load_dataset(path=DATASET_PATH):
    """Returns (X_train_pool, theta_train_pool, X_test, theta_test) from the canonical dataset
    file. X_train_pool/theta_train_pool hold n_train_pool (10,000) points -- pass them through
    subsample_train() to get whatever training-set size a given run actually needs. X_test/
    theta_test (1,000 points) are used AS-IS, identically, by every script/notebook/size -- never
    resliced or regenerated."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python scripts/build_dataset.py` once to generate it."
        )
    with np.load(path) as f:
        return f["X_train"], f["theta_train"], f["X_test"], f["theta_test"]


def subsample_train(X_train, theta_train, n, repeat=0, base_seed=123):
    """Nested-subsample convention shared by every caller that needs a specific training-set size
    out of the canonical pool: np.random.default_rng(base_seed + repeat).permutation(...)[:n] --
    e.g. the n=50 subset is always a PREFIX of the n=100 subset for the same repeat, and (with the
    same repeat) the same n always picks the same rows across every script that calls this. Matches
    scripts/run_experiment.py's original inline subsampling exactly (base_seed=123, one repeat per
    sweep repeat); other scripts that only ever need one draw (validation scripts, hyperparameter
    search) use the default repeat=0, which reproduces run_experiment.py's own repeat=0 subsample
    for that size -- i.e. a validation run at a given n_train uses the EXACT SAME rows the real
    sweep's first repeat at that size would use, not an independently-drawn set."""
    idx = np.random.default_rng(base_seed + repeat).permutation(len(X_train))[:n]
    return X_train[idx], theta_train[idx]


def _loglik(sigma, rho, S):
    K = sigma**2 * jnp.exp(-(T[:, None] - T[None, :])**2 / (2 * rho**2))
    cov = K + ETA**2 * jnp.eye(D)
    chol = jnp.linalg.cholesky(cov)
    return -0.5 * N * (
        2 * jnp.log(jnp.diag(chol)).sum()
        + jnp.trace(jax.scipy.linalg.cho_solve((chol, True), S))
    )


def _logprior(sigma, rho):
    return (
        -0.5 * ((sigma - SIGMA_MU) / SIGMA_STD) ** 2
        -0.5 * ((rho - RHO_MU) / RHO_STD) ** 2
    )


def _logpost_one_dataset(S):
    return jax.vmap(
        lambda sigma: jax.vmap(
            lambda rho: _loglik(sigma, rho, S) + _logprior(sigma, rho)
        )(RHO_GRID)
    )(SIGMA_GRID)


def compute_reference_posterior(X_test, batch_size=25):
    """Exact grid posterior for every test dataset, via Gaussian-process log-likelihood + prior.
    Returns (post, p_sigma, p_rho) — post has shape (n_test, len(SIGMA_GRID), len(RHO_GRID))."""
    S_test = X_test.reshape(-1, D, D)

    log_posts = []
    for i in range(0, len(S_test), batch_size):
        batch = S_test[i:i + batch_size]
        log_posts.append(jax.vmap(_logpost_one_dataset)(batch))
    log_post = jnp.concatenate(log_posts, axis=0)

    log_post -= jnp.max(log_post, axis=(1, 2), keepdims=True)
    post = jnp.exp(log_post)
    post /= jnp.sum(post, axis=(1, 2), keepdims=True)

    p_sigma = jnp.sum(post, axis=2)
    p_rho = jnp.sum(post, axis=1)

    # Sanity check: mass sitting on the grid boundary should be ~0
    edge_mass_sigma = float(jnp.maximum(p_sigma[:, 0].max(), p_sigma[:, -1].max()))
    edge_mass_rho = float(jnp.maximum(p_rho[:, 0].max(), p_rho[:, -1].max()))
    print(f"Max boundary marginal mass — sigma: {edge_mass_sigma:.2e}, rho: {edge_mass_rho:.2e}")

    return post, p_sigma, p_rho


def make_reference_theta_samples(post, n_draws, seed=123):
    """Draw n_draws posterior samples per test point from the grid posterior (with jitter within each cell)."""
    rng = np.random.default_rng(seed)
    sigma_np, rho_np, post_np = np.asarray(SIGMA_GRID), np.asarray(RHO_GRID), np.asarray(post)
    n_test = post_np.shape[0]
    dsig = sigma_np[1] - sigma_np[0]
    drho = rho_np[1] - rho_np[0]

    ref_samples = np.empty((n_draws, n_test, 2))
    for i in range(n_test):
        p = post_np[i].ravel().copy()
        p /= p.sum()
        idx = rng.choice(len(p), size=n_draws, p=p)
        si, ri = np.unravel_index(idx, post_np[i].shape)
        ref_samples[:, i, 0] = sigma_np[si] + (rng.random(n_draws) - 0.5) * dsig
        ref_samples[:, i, 1] = rho_np[ri] + (rng.random(n_draws) - 0.5) * drho
    return ref_samples


def compute_reference_log_prob_true(post, theta_test):
    """Bilinear-interpolated log-density of the exact grid posterior at each test point's true theta."""
    sigma_np, rho_np, post_np = np.asarray(SIGMA_GRID), np.asarray(RHO_GRID), np.asarray(post)
    dsig = sigma_np[1] - sigma_np[0]
    drho = rho_np[1] - rho_np[0]
    density = post_np / (dsig * drho)  # convert grid probability mass -> proper density

    theta_true = np.asarray(theta_test)
    n_test = post_np.shape[0]
    log_prob_true = np.empty(n_test)

    for i in range(n_test):
        s_true, r_true = theta_true[i]
        si = int(np.clip(np.searchsorted(sigma_np, s_true) - 1, 0, len(sigma_np) - 2))
        ri = int(np.clip(np.searchsorted(rho_np, r_true) - 1, 0, len(rho_np) - 2))
        s0, s1 = sigma_np[si], sigma_np[si + 1]
        r0, r1 = rho_np[ri], rho_np[ri + 1]
        ws = (s_true - s0) / (s1 - s0)
        wr = (r_true - r0) / (r1 - r0)

        d00, d10 = density[i, si, ri], density[i, si + 1, ri]
        d01, d11 = density[i, si, ri + 1], density[i, si + 1, ri + 1]
        d_interp = d00 * (1 - ws) * (1 - wr) + d10 * ws * (1 - wr) + d01 * (1 - ws) * wr + d11 * ws * wr
        log_prob_true[i] = np.log(np.maximum(d_interp, 1e-300))

    return log_prob_true
