"""Pretraining-only (lr, weight_decay) grid search for the BNN's network (2026-11-02).

Question: the BNN's (lr, wd) (results/bde_params*_n{n}.npz, from legacy/optimize_bde.py) were chosen by the
held-out NLL of the FULL MILE fit (pretraining + 50,000 warmup + 10,000 MCLMC steps), not by how good the
pretrained network is on its own. At n=500 that search picked lr=3.6e-5, under which every member hits the
2,000-step pretraining cap (pretrain_stop_epoch = -1). This script asks what (lr, wd) the SAME network would get
if it were tuned as a standalone predictor -- "the BNN without sampling" / a Gaussian predictor with the BNN's
architecture -- and how far that is from the values the BNN used.

What is fitted: ONE network per fit, via bnn.fit_pretrained_ensemble with n_members=1 -- the exact code path of the
BNN's pretraining (CorrelatedBdeRegressor with pretrain_only=True: standardisation, 100->16->16->16->5 soft-sign,
bivariate Gaussian NLL, full-batch AdamW, <= 2,000 steps, eval every 100, patience 5, min_delta 1e-4, 500-step
floor, bde's own 80/20 early-stopping split). No warmup, no MCLMC, no ensemble. bde seeds member i with seed + i
regardless of n_members, so this network is identical to member 0 of a multi-member fit with the same seed.

Objective (same construction as optimize_bde.py, so the two are comparable): mean over N_REPEATS=3 training draws
(repeats 0-2, optimize_bde.build_repeat_splits -- imported, so the splits are bit-identical, including its n=10,000
fallback to an inner 80/20 split of the subsample) of -mean log q(theta | x) on 1,000 held-out pool rows. The
three draws (different training data and seed) are what averages out fit-to-fit noise in the score.

Grid (not Bayesian optimisation, so the whole landscape is visible): lr = 10^[-5 .. -1] in quarter decades (17),
wd = 10^[-7 .. 0] in half decades (15). lr bounds and the wd lower bound are optimize_bde.py's SEARCH_SPACE
(ASSUMPTION: those bounds were never justified; see paper/paper.tex App. C); the wd upper bound is raised
from 1e-2 to 1 because optax AdamW decays decoupled, by lr * wd * w per step, so wd <= 1e-2 is nearly inert except
at the largest lr (smoke test: lr 1e-4, wd 1e-6 and 1e-3 gave the same NLL to 4 decimals). Plus one extra point per
size: the (lr, wd) the BNN actually uses (run_experiment.py's hyperparameter composite: bde_params_new_n{n}.npz if
present, else bde_params_n{n}.npz).

Seeds: COMMON random numbers -- draw k uses seed 123 + k at every grid point, so differences between grid points
are due to (lr, wd) alone (optimize_bde.py instead varied the seed per trial). No redo on failure (pretraining is
deterministic given the seed; a reseed would break the common-random-numbers design): a non-finite network is
recorded as NaN for that draw and the grid point's mean is NaN.

History: first written with a 12-network (then 3-network) mixture as a secondary score; dropped 2026-11-02 before
any job started -- the question is the standalone network's hyperparameters, and the 3 draws already average noise.

Usage:
  python scripts/tuning/optimize_pretrain_grid.py [sizes ...]           # default: all six sizes
Env:
  PRETRAIN_GRID_OUT_DIR  output folder (default results/hp_search/pretrain_grid) -- set to a scratch path for
                         smoke tests; real results are never overwritten (a size whose file exists is skipped).
  PRETRAIN_GRID_SMOKE=1  tiny grid (2 lr x 2 wd), 1 draw, no experiment-log entry -- timing / plumbing check only.
Output: {out_dir}/pretrain_grid_n{n}.npz (one per size, written atomically); one experiment_log entry per size.
"""
import os
import sys
import time
from pathlib import Path

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # this repository's src/
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401

import numpy as np

from rbf_sbi import data
from rbf_sbi import bnn
from rbf_sbi import experiment_log
from rbf_sbi.evaluation import build_repeat_splits, N_REPEATS as BDE_N_REPEATS, VAL_SIZE

SIZES = [50, 100, 500, 1000, 5000, 10000]
SMOKE = os.environ.get("PRETRAIN_GRID_SMOKE", "") == "1"
OUT_DIR = Path(os.environ.get("PRETRAIN_GRID_OUT_DIR", RESULTS / "hp_search" / "pretrain_grid"))

# Fixed at scripts/fit/run_experiment.py's production values (checked 2026-11-02, run_experiment.py:398-404).
BNN_HIDDEN_LAYERS = [16, 16, 16]
BNN_EPOCHS, BNN_PATIENCE, BNN_VALIDATION_SPLIT = 2000, 5, 0.2
BNN_PRETRAIN_EVAL_EVERY, BNN_PRETRAIN_MIN_STEPS = 100, 500
BNN_ACTIVATION = "soft_sign"
N_MEMBERS = 1           # one network per fit -- no ensemble
SEED_BASE = 123
TARP_N_DRAWS, TARP_SEEDS = 50, [0]      # TARP is not the objective; kept tiny

if SMOKE:
    LR_GRID = np.array([1e-4, 1e-2])
    WD_GRID = np.array([1e-6, 1e-3])
    N_DRAWS = 1
else:
    LR_GRID = 10.0 ** np.linspace(-5, -1, 17)
    WD_GRID = 10.0 ** np.linspace(-7, 0, 15)   # extends optimize_bde.py's upper bound 1e-2 -> 1 (see docstring)
    N_DRAWS = BDE_N_REPEATS


def used_hp(n):
    """The (lr, wd, file) the final sweep's BNNs use at size n (run_experiment.py prefers the _new_ file)."""
    for name in (f"bde_params_new_n{n}.npz", f"bde_params_n{n}.npz"):
        p = RESULTS / name
        if p.exists():
            with np.load(p) as d:
                return float(d["lr"]), float(d["weight_decay"]), name
    raise FileNotFoundError(f"no bde_params file for n={n}")


def fit_point(splits, lr, wd):
    """One (lr, wd): N_DRAWS single-network pretraining-only fits. Returns per-draw arrays."""
    out = {k: np.full(N_DRAWS, np.nan) for k in ("val_nll", "train_nll", "stop_epoch")}
    out["capped"] = np.zeros(N_DRAWS, bool)
    for k in range(N_DRAWS):
        Xt, yt, Xv, yv = splits[k]
        try:
            _, _, single_raw, _, diag = bnn.fit_pretrained_ensemble(
                Xt, yt, Xv, yv, SEED_BASE + k, N_MEMBERS, BNN_HIDDEN_LAYERS, BNN_EPOCHS, BNN_PATIENCE,
                BNN_VALIDATION_SPLIT, lr, wd, data.TARP_LO, data.TARP_HI, TARP_N_DRAWS, TARP_SEEDS,
                quiet=True, pretrain_eval_every=BNN_PRETRAIN_EVAL_EVERY, pretrain_min_steps=BNN_PRETRAIN_MIN_STEPS,
                activation=BNN_ACTIVATION)
        except RuntimeError as e:                       # the network came back non-finite
            print(f"      draw {k}: {e}", flush=True)
            continue
        se = int(np.asarray(diag["pretrain_stop_epoch"])[0])
        out["val_nll"][k] = -float(np.mean(single_raw["log_prob_true"]))
        out["train_nll"][k] = -float(np.mean(single_raw["train_log_prob_true"]))
        out["capped"][k] = se == -1
        out["stop_epoch"][k] = BNN_EPOCHS if se == -1 else se
    return out


def search_one_size(X_pool, theta_pool, n):
    splits = build_repeat_splits(X_pool, theta_pool, n)[:N_DRAWS]
    used_lr, used_wd, used_file = used_hp(n)
    points = [(lr, wd) for lr in LR_GRID for wd in WD_GRID] + [(used_lr, used_wd)]
    res = {}
    t0 = time.time()
    for i, (lr, wd) in enumerate(points):
        t = time.time()
        r = fit_point(splits, lr, wd)
        for key, v in r.items():
            res.setdefault(key, []).append(v)
        tag = " (BNN's used values)" if i == len(points) - 1 else ""
        print(f"  n={n} [{i + 1}/{len(points)}] lr={lr:.2e} wd={wd:.1e}{tag}: val NLL {np.mean(r['val_nll']):+.4f}"
              f" (draws {' '.join(f'{v:+.3f}' for v in r['val_nll'])}) capped {int(r['capped'].sum())}/{N_DRAWS}"
              f" ({time.time() - t:.1f}s)", flush=True)
    res = {k: np.array(v) for k, v in res.items()}            # (n_points, N_DRAWS)
    grid = res["val_nll"][:-1].mean(1).reshape(len(LR_GRID), len(WD_GRID))
    g = np.where(np.isfinite(grid), grid, np.inf)
    i, j = np.unravel_index(np.argmin(g), g.shape)
    return dict(
        res=res, grid=grid, best=(float(LR_GRID[i]), float(WD_GRID[j]), float(g[i, j])),
        used=(used_lr, used_wd, used_file), used_val_nll=float(res["val_nll"][-1].mean()),
        used_capped=int(res["capped"][-1].sum()), elapsed=time.time() - t0,
        point_lr=np.array([p[0] for p in points]), point_wd=np.array([p[1] for p in points]),
    )


def save(n, r):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"pretrain_grid_n{n}.npz"
    tmp = OUT_DIR / f"pretrain_grid_n{n}.tmp.npz"
    np.savez_compressed(
        tmp, n_train=n, lr_grid=LR_GRID, wd_grid=WD_GRID, n_draws=N_DRAWS, val_size=VAL_SIZE,
        seed_base=SEED_BASE, smoke=SMOKE,
        grid_val_nll=r["grid"],                                                 # (n_lr, n_wd), mean over draws
        point_lr=r["point_lr"], point_wd=r["point_wd"], **{f"point_{k}": v for k, v in r["res"].items()},
        best_lr=r["best"][0], best_wd=r["best"][1], best_val_nll=r["best"][2],
        used_lr=r["used"][0], used_wd=r["used"][1], used_hp_file=r["used"][2],
        used_val_nll=r["used_val_nll"], used_capped=r["used_capped"], elapsed_sec=r["elapsed"],
        hidden_layers=np.array(BNN_HIDDEN_LAYERS), epochs=BNN_EPOCHS, patience=BNN_PATIENCE,
        validation_split=BNN_VALIDATION_SPLIT, pretrain_eval_every=BNN_PRETRAIN_EVAL_EVERY,
        pretrain_min_steps=BNN_PRETRAIN_MIN_STEPS, activation=BNN_ACTIVATION,
    )
    tmp.rename(path)
    return path


def main():
    sizes = [int(s) for s in sys.argv[1:]] or SIZES
    X_pool, theta_pool, _, _ = data.load_dataset()
    print(f"Pretraining-only grid, one network per fit: {len(LR_GRID)} lr x {len(WD_GRID)} wd + BNN's used point, "
          f"{N_DRAWS} draws, out={OUT_DIR}, smoke={SMOKE}", flush=True)
    for n in sizes:
        if (OUT_DIR / f"pretrain_grid_n{n}.npz").exists():
            print(f"Skipping n={n} (exists)")
            continue
        r = search_one_size(X_pool, theta_pool, n)
        path = save(n, r)
        (blr, bwd, bnll), (ulr, uwd, uf) = r["best"], r["used"]
        print(f"n={n}: best lr={blr:.2e} wd={bwd:.1e} val NLL {bnll:+.4f} | BNN used lr={ulr:.2e} wd={uwd:.1e} ({uf}): "
              f"val NLL {r['used_val_nll']:+.4f}, capped {r['used_capped']}/{N_DRAWS} -> {path}", flush=True)
        if SMOKE:                                           # smoke runs never touch the real experiment ledger
            continue
        experiment_log.log_run(
            script="scripts/tuning/optimize_pretrain_grid.py",
            config=dict(n_train=n, n_lr=len(LR_GRID), n_wd=len(WD_GRID), lr_range=(LR_GRID[0], LR_GRID[-1]),
                        wd_range=(WD_GRID[0], WD_GRID[-1]), n_draws=N_DRAWS, n_members=N_MEMBERS, val_size=VAL_SIZE,
                        seed_base=SEED_BASE, hidden_layers=BNN_HIDDEN_LAYERS, epochs=BNN_EPOCHS,
                        patience=BNN_PATIENCE, pretrain_min_steps=BNN_PRETRAIN_MIN_STEPS),
            results=dict(best_lr=blr, best_wd=bwd, best_val_nll=bnll, used_lr=ulr, used_wd=uwd, used_hp_file=uf,
                         used_val_nll=r["used_val_nll"], used_capped=r["used_capped"], elapsed_sec=r["elapsed"]),
            job_id=os.environ.get("SLURM_JOB_ID"), output_path=str(path),
            notes="Pretraining-only (no MCLMC), single-network lr/wd grid for the BNN's network, scored as a standalone "
                  "predictor with optimize_bde.py's held-out objective; compares to the (lr, wd) the BNN used, which "
                  "were tuned on the full MILE fit.",
        )


if __name__ == "__main__":
    main()
