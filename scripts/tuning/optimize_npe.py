"""GP-based Bayesian-optimization search for NPE's (lr, weight_decay), run separately per
training-set size. Replaces scripts/hp_search.py's fixed 3x3 grid for NPE.

Uses skopt's ask/tell Optimizer loop under a wall-clock budget per size, not a fixed trial
count -- the search naturally adapts to how expensive a training run is at each size, and stays
within a predictable total budget regardless of how many trials that turns out to be. Reference:
Snoek, Larochelle & Adams (2012), "Practical Bayesian Optimization of Machine Learning
Algorithms," NeurIPS -- GP surrogate + expected improvement.

Resumable: writes results/npe_params.npz after every size, and skips any size already present in
that file on restart.
"""
import sys
import time
from pathlib import Path

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # this repository's src/
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401

import numpy as np
import torch
from skopt import Optimizer
from skopt.space import Real

from rbf_sbi import data
from rbf_sbi import metrics
from rbf_sbi import npe as npe_module

SIZES = [10000, 5000, 1000, 500, 100, 50]
PER_SIZE_TIME_BUDGET_SEC = 200  # ~20 min total across 6 sizes

# Same training budget as the production run (scripts/fit/run_experiment.py) -- early stopping
# keeps any one trial short; the wall-clock budget above bounds how many trials fit, not how
# long a single trial is allowed to run.
STEPS, EVAL_EVERY, PATIENCE = 20000, 100, 20

SEARCH_SPACE = [
    Real(1e-5, 1e-1, prior="log-uniform", name="lr"),
    Real(1e-7, 1e-2, prior="log-uniform", name="wd"),
]

RESULTS_DIR = RESULTS
OUT_PATH = RESULTS_DIR / "npe_params.npz"


def train_and_validate(Xt, yt, Xv, yv, lr, wd, seed):
    """Lightweight NPE fit for search purposes: trains + early-stops, returns best val loss only
    (no TARP/sampling evaluation, unlike npe.fit -- that's for the final production run)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    def T(z):
        return torch.as_tensor(np.asarray(z).reshape(len(z), -1).copy(),
                                dtype=torch.float32, device=npe_module.DEVICE)

    Xt_t, Xv_t = T(Xt), T(Xv)
    yt_t = torch.as_tensor(np.asarray(yt, dtype=np.float32).copy(), device=npe_module.DEVICE)
    yv_t = torch.as_tensor(np.asarray(yv, dtype=np.float32).copy(), device=npe_module.DEVICE)

    net = npe_module.create_npe_flow(
        theta_dim=2, input_context_dim=Xt_t.shape[1],
        encoded_context_dim=8, hidden_features=16, num_layers=2, num_blocks=1,
    ).to(npe_module.DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    def loss(x, y):
        return -net["flow"].log_prob(y, context=net["encoder"](x)).mean()

    best_val, bad = np.inf, 0
    for step in range(STEPS):
        net.train()
        opt.zero_grad(set_to_none=True)
        L = loss(Xt_t, yt_t)
        if not torch.isfinite(L):
            return float("inf")
        L.backward()
        opt.step()

        if step == 0 or (step + 1) % EVAL_EVERY == 0:
            net.eval()
            with torch.no_grad():
                v = loss(Xv_t, yv_t).item()
            if v < best_val - 1e-4:
                best_val, bad = v, 0
            else:
                bad += 1
            if bad >= PATIENCE:
                break

    return best_val


def search_one_size(X_train, theta_train, n, seed=123):
    Xb, yb = data.subsample_train(X_train, theta_train, n, repeat=0, base_seed=seed)
    Xt, yt, Xv, yv = metrics.split(np.asarray(Xb), np.asarray(yb), seed)

    opt = Optimizer(SEARCH_SPACE, base_estimator="GP", acq_func="EI", random_state=seed)

    t0 = time.time()
    n_trials = 0
    while n_trials < 1 or (time.time() - t0) < PER_SIZE_TIME_BUDGET_SEC:
        lr, wd = opt.ask()
        val = train_and_validate(Xt, yt, Xv, yv, lr, wd, seed=seed + n_trials)
        opt.tell([lr, wd], val)
        n_trials += 1

    best_idx = int(np.argmin(opt.yi))
    best_lr, best_wd = opt.Xi[best_idx]
    best_val = opt.yi[best_idx]
    elapsed = time.time() - t0
    print(f"n_train={n:>6} | {n_trials} trials in {elapsed:.0f}s | "
          f"best lr={best_lr:.2e} wd={best_wd:.2e} val={best_val:.4f}", flush=True)
    return best_lr, best_wd, best_val, n_trials, elapsed


def load_existing():
    if not OUT_PATH.exists():
        return {}
    with np.load(OUT_PATH) as f:
        return {
            int(n): (float(f["lr"][i]), float(f["wd"][i]), float(f["val_loss"][i]),
                      int(f["n_trials"][i]), float(f["elapsed_sec"][i]))
            for i, n in enumerate(f["sizes"])
        }


def save_atomic(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sizes = sorted(results, reverse=True)
    tmp_path = OUT_PATH.parent / (OUT_PATH.stem + ".tmp.npz")
    np.savez_compressed(
        tmp_path,
        sizes=np.array(sizes),
        lr=np.array([results[n][0] for n in sizes]),
        wd=np.array([results[n][1] for n in sizes]),
        val_loss=np.array([results[n][2] for n in sizes]),
        n_trials=np.array([results[n][3] for n in sizes]),
        elapsed_sec=np.array([results[n][4] for n in sizes]),
    )
    tmp_path.rename(OUT_PATH)


def main():
    results = load_existing()
    if results:
        print(f"Resuming -- already have results for sizes: {sorted(results, reverse=True)}")

    X_train, theta_train, _, _ = data.load_dataset()

    for n in SIZES:
        if n in results:
            print(f"Skipping n_train={n} (already in {OUT_PATH.name})")
            continue
        results[n] = search_one_size(X_train, theta_train, n)
        save_atomic(results)

    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
