"""Cache for the exact reference posterior (data.compute_reference_posterior) over the canonical
dataset's test set AND, as of 2026-08-25, its training-set subsamples too -- a real grid
computation (240x240 per point), expensive enough that recomputing it inside a notebook every
time is the difference between "plotting a completed run instantly" and "needs a Slurm/GPU job".

--- Why the cache key isn't just (n_test, seed) for the test set ---

data.generate_train_test(key, n_train, n_test) calls jax.random.split(key, n_train + n_test) and
takes X_test as the TAIL of that split. Because the split's output depends on the total count, a
different n_train produces a completely different X_test/theta_test even with the same seed/n_test
-- confirmed empirically 2026-08-25. This is exactly why the project
now has ONE canonical dataset (data.build_and_save_dataset/load_dataset) instead of every script
calling generate_train_test itself -- see that module's docstring. get_test_reference_posterior's
key still includes n_train (alongside n_test/seed) as a defensive match against the canonical
dataset ever changing, verified against the actual X_test/theta_test passed in on every call
(raises loudly on a mismatch rather than silently serving a stale/wrong posterior -- this caught a
real stale-cache bug during the canonical-dataset migration).

--- Training-set caching ---

X_train is data.subsample_train(X_train_pool, theta_train_pool, n, repeat, base_seed) -- what
determines it is (n, repeat, base_seed), not n_test at all, so get_train_reference_posterior keys
on those instead. Both gaussian_analysis.ipynb's and correlated_bde_perchain_analysis.ipynb's
training-set fit-check cells were independently recomputing this every run; now cached like the
test set.

Cache files live under data/reference_posterior/ (project root's data/ folder -- unrelated to
src/data.py despite the similar name; this is a cached-artifact directory, not a Python package).
Deliberately small relative to results/raw/*.npz -- kept in git/$HOME, not $PSCRATCH.
"""
from pathlib import Path

import numpy as np

from rbf_sbi import data
from rbf_sbi.paths import DATA

CACHE_DIR = DATA / "reference_posterior"


def _cache_path(tag):
    return CACHE_DIR / f"{tag}.npz"


def _get_cached_or_compute(tag, X, theta, force_recompute):
    """Shared load/verify/compute/save logic for both the test-set and training-set caches below.
    Raises ValueError on a cache hit whose saved X/theta don't match what was passed in -- see
    module docstring for why this must fail loudly rather than silently serve stale data."""
    path = _cache_path(tag)

    if path.exists() and not force_recompute:
        with np.load(path) as f:
            if not (np.allclose(f["X"], np.asarray(X)) and np.allclose(f["theta"], np.asarray(theta))):
                raise ValueError(
                    f"Cached reference posterior at {path} does not match the given X/theta -- "
                    "the cache key may not uniquely determine this data the way this module "
                    "assumes (e.g. a stale cache from before a dataset-generation change). Not "
                    "using the cache; investigate before retrying, or delete the stale file, or "
                    "pass force_recompute=True once the cause is understood."
                )
            return f["post"], f["p_sigma"], f["p_rho"]

    post, p_sigma, p_rho = data.compute_reference_posterior(X)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / (path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp_path, X=np.asarray(X), theta=np.asarray(theta),
        post=np.asarray(post), p_sigma=np.asarray(p_sigma), p_rho=np.asarray(p_rho),
    )
    tmp_path.rename(path)
    return post, p_sigma, p_rho


def get_test_reference_posterior(X_test, theta_test, n_train, n_test, seed, force_recompute=False):
    """Returns (post, p_sigma, p_rho) for the canonical test set. Cached at
    data/reference_posterior/test_n{n_train}_ntest{n_test}_seed{seed}.npz."""
    tag = f"test_n{n_train}_ntest{n_test}_seed{seed}"
    return _get_cached_or_compute(tag, X_test, theta_test, force_recompute)


def get_train_reference_posterior(X_train, theta_train, n_train, repeat=0, base_seed=123, force_recompute=False):
    """Returns (post, p_sigma, p_rho) for a data.subsample_train(..., n_train, repeat, base_seed)
    training subsample. Cached at
    data/reference_posterior/train_n{n_train}_repeat{repeat}_seed{base_seed}.npz."""
    tag = f"train_n{n_train}_repeat{repeat}_seed{base_seed}"
    return _get_cached_or_compute(tag, X_train, theta_train, force_recompute)
