"""One-time (or rerun-only-if-deliberately-changing-the-seed/pool-size) generator for the
project's single canonical dataset -- see src/data.py's build_and_save_dataset/load_dataset/
subsample_train for why this exists and what depends on it.

Saves data/dataset.npz: X_train/theta_train (n_train_pool=10,000 points) and X_test/theta_test
(n_test=1,000 points), seed=0 -- the exact same data.generate_train_test(PRNGKey(0), 10_000, 1000)
call scripts/fit/run_experiment.py already made at its own top, now factored into one file every
script/notebook reads instead of each calling generate_train_test independently. That
independent-calling pattern is what silently produced a DIFFERENT X_test per script whenever the
n_train passed to generate_train_test differed between callers (found
2026-08-25) -- this file is the fix: one canonical test set, and training subsamples that
are nested prefixes of one shared pool (data.subsample_train), for every model/size/script.

Cheap: pure simulation (RBF-kernel draws + a 10x10 Cholesky per dataset, vmapped over 11,000
datasets) -- no model fitting, no exact-posterior grid computation. Runs in seconds; safe to run
directly on a login node, CPU-only (JAX_PLATFORMS=cpu) is plenty and avoids touching a
shared login-node GPU for no reason.

Usage: JAX_PLATFORMS=cpu python scripts/data/build_dataset.py
"""
import sys
from pathlib import Path

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # this repository's src/
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401

from rbf_sbi import data


def main():
    X_train, theta_train, X_test, theta_test = data.build_and_save_dataset()
    print(f"Saved {data.DATASET_PATH}")
    print(f"  X_train {X_train.shape}  theta_train {theta_train.shape}")
    print(f"  X_test  {X_test.shape}  theta_test  {theta_test.shape}")


if __name__ == "__main__":
    main()
