"""Build and cache the exact reference posterior of the 1,000 test datasets (240 x 240 grid per dataset, about
110 MB) in data/reference_posterior/test_n100_ntest1000_seed0.npz, the file every analysis script reads. Run once,
after scripts/data/build_dataset.py. JAX; a GPU makes it fast.

Usage: python scripts/data/build_reference.py
"""
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # this repository's src/
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401
from rbf_sbi import data, refpost_cache  # noqa: E402

if __name__ == "__main__":
    X_train, theta_train, X_test, theta_test = data.load_dataset()
    refpost_cache.get_test_reference_posterior(X_test, theta_test, n_train=100, n_test=len(X_test), seed=0)
    print(f"cached {refpost_cache.CACHE_DIR}")
