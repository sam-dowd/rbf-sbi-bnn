"""Coarse per-size temperature search for the plain N(0,1)-prior BNN ("bnn_temp"'s setup), added
2026-10-01. Replaces the single T=0.305 that scripts/optimize_temperature.py found at n=100 ONLY,
under the Delaunoy-optimized prior and an older tuned lr (0.09996), and that the sweep then applied
unchanged to every size under N(0,1) -- where it fails from n=500 up (test NLL +372..+618 at n=500,
R-hat max 179-257; 2026-09-17 bnn_temp run).

Search: T on the grid 0.2, 0.3, ..., 1.0, by bisection on the sign of the local slope (compare
f(mid) with f(mid+1), keep the half that goes downhill), so at most 7 of the 9 grid points are
fit, plus T=1.0 always (the untempered baseline). f(T) = mean validation NLL over N_REPEATS
training draws. The reported best T is the argmin over every grid point actually evaluated, not
just the bisection's end point.

Everything except T matches scripts/fit/run_experiment.py's "bnn_vanilla"/"bnn_temp" exactly: N(0,1)
prior, the sweep's own tuned (lr, weight_decay) from results/bde_params_n{n}.npz (or, when it
exists, the re-run search's results/bde_params_new_n{n}.npz -- output then goes to
temperature_search_n{n}_stdprior_hpnew_grid.npz; see HP_PATH below), same chain
budget (10,000 steps / 1,000 kept per chain, 50,000 warmup), same per-(size, repeat) seed
(100 * index of n in the sweep's size list + repeat), and the training rows of repeat k are
bit-identical to data.subsample_train(..., n, repeat=k). So pretraining for repeats 0-2 is the
same computation as the sweep's, and only the MCLMC-phase temperature differs.

Validation set: the VAL_SIZE pool rows right after the n training rows in the same permutation --
never the training rows, never X_test (same construction as scripts/optimize_temperature.py).

ASSUMPTIONS, flagged directly:
  (a) bisection assumes mean validation NLL is unimodal in T on this grid. It is a noisy average
      of 3 fits, so a wrong turn is possible; every evaluated point is saved so this can be
      checked afterwards, and the argmin is taken over all of them.
  (b) N_REPEATS=3 (repeats 0-2 of the sweep's 5), as in the earlier search -- a cost choice.
  (c) a fit that fails outright (no finite member within BNN_MAX_SEED_TRIES=3 attempts, the
      sweep's own setting) or returns a non-finite NLL scores +inf for that repeat, with no redo
      -- so one failed repeat rules that temperature out.
  (d) mixing does not enter the objective. R-hat/ESS/member-fail-rate are recorded per trial;
      a temperature can win on validation NLL with unmixed chains (T=0.305 at n=100 does).
  (e) selection is on validation NLL only -- not on second-moment error or calibration.

Usage: python scripts/tuning/optimize_temperature_grid.py <n_train> [--smoke] [--full]
  --smoke: tiny budgets, 2 grid points, writes to a scratch path, no experiment-log entry.
  --full (2026-10-01): evaluate EVERY grid point instead of bisecting, re-using any points an
    earlier run of this script already wrote for this size. Added after the n=50 bisection under
    the re-run hyperparameters turned at T=0.6 vs 0.7 on a 0.018 difference (inside the ~0.04
    trial-to-trial noise) and so never evaluated T <= 0.5 -- assumption (a) below failing in
    practice.
Output: results/temperature_search_n{n}_stdprior_grid.npz (rewritten after every trial, so a
timed-out job keeps what it finished; an existing complete file is not redone).
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

SWEEP_SIZES = [10000, 5000, 1000, 500, 100, 50]  # scripts/fit/run_experiment.py's ALL_SIZES (seed index)
SMOKE = "--smoke" in sys.argv
FULL = "--full" in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
if len(ARGS) != 1 or int(ARGS[0]) not in SWEEP_SIZES:
    raise SystemExit(f"Usage: python scripts/tuning/optimize_temperature_grid.py <n_train> [--smoke]\n"
                     f"  n_train must be one of {SWEEP_SIZES}")
N_TRAIN = int(ARGS[0])

T_GRID = [round(0.2 + 0.1 * i, 1) for i in range(9)]  # 0.2 ... 1.0
N_REPEATS = 3
SUBSAMPLE_BASE_SEED = 123  # data.subsample_train's default / the sweep's convention
VAL_SIZE = 1000

BNN_N_MEMBERS = 3
BNN_HIDDEN_LAYERS = [16, 16, 16]
BNN_EPOCHS, BNN_PATIENCE, BNN_VALIDATION_SPLIT = 2000, 5, 0.2
BNN_N_SAMPLES, BNN_WARMUP_STEPS, BNN_N_THINNING = 10000, 50000, 10  # 10,000 raw steps, 1,000 kept
BNN_DESIRED_ENERGY_VAR_START, BNN_DESIRED_ENERGY_VAR_END = 0.5, 0.1
BNN_MAX_SEED_TRIES, BNN_RHAT_THRESHOLD = 3, 1.01
BNN_MIN_GOOD_MEMBERS = BNN_N_MEMBERS
BNN_PRETRAIN_EVAL_EVERY, BNN_PRETRAIN_MIN_STEPS = 100, 500
TARP_N_DRAWS, TARP_SEEDS = 200, list(range(5))

# Hyperparameter version -- same rule as scripts/fit/run_experiment.py's _load_bnn_hp (see its module
# docstring's HYPERPARAMETER VERSIONS): prefer the re-run search's file when it exists, and tag
# this search's output so it is the one "bnn_temp_opt" reads when it runs under those values.
_NEW_HP_PATH = RESULTS / f"bde_params_new_n{N_TRAIN}.npz"
HP_PATH = _NEW_HP_PATH if _NEW_HP_PATH.exists() else RESULTS / f"bde_params_n{N_TRAIN}.npz"
HP_TAG = "_hpnew" if HP_PATH == _NEW_HP_PATH else ""
# 2026-11-02: PRETUNED_HP=1 -> (lr, wd) from the pretraining-only grid (scripts/tuning/optimize_pretrain_grid.py,
# results/hp_search/pretrain_grid/pretrain_grid_n{n}.npz best_lr / best_wd) and output tagged _pretuned -- the
# temperatures read by run_experiment.py's final_bnn_temp_pretuned. Search otherwise unchanged.
PRETUNED = os.environ.get("PRETUNED_HP", "") == "1"
if PRETUNED:
    HP_PATH = RESULTS / "hp_search" / "pretrain_grid" / f"pretrain_grid_n{N_TRAIN}.npz"
    HP_TAG = "_pretuned"
OUT_PATH = RESULTS / f"temperature_search_n{N_TRAIN}_stdprior{HP_TAG}_grid.npz"
if SMOKE:
    T_GRID = [0.5, 1.0]
    N_REPEATS = 1
    BNN_N_MEMBERS = BNN_MIN_GOOD_MEMBERS = 2
    BNN_EPOCHS, BNN_N_SAMPLES, BNN_WARMUP_STEPS = 300, 200, 400
    BNN_PRETRAIN_MIN_STEPS = 100
    OUT_PATH = SCRATCH / "sbi_smoke" / f"temperature_grid_smoke_n{N_TRAIN}.npz"


def load_lr_weight_decay():
    path = HP_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run legacy/optimize_bde.py {N_TRAIN} first.")
    d = np.load(path)
    if PRETUNED:
        if bool(d["smoke"]):
            raise RuntimeError(f"{path.name} is a smoke-test grid, not a result.")
        return float(d["best_lr"]), float(d["best_wd"])
    return float(d["lr"]), float(d["weight_decay"])


def build_repeat_splits(X_train, theta_train):
    """(Xt, yt, Xv, yv) per repeat. Xt/yt are data.subsample_train(..., N_TRAIN, repeat=k)'s rows;
    Xv/yv are the next VAL_SIZE rows of the same permutation."""
    splits = []
    for k in range(N_REPEATS):
        perm = np.random.default_rng(SUBSAMPLE_BASE_SEED + k).permutation(len(X_train))
        tr, va = perm[:N_TRAIN], perm[N_TRAIN:N_TRAIN + VAL_SIZE]
        splits.append((X_train[tr], theta_train[tr], X_train[va], theta_train[va]))
    return splits


def fit_one(Xt, yt, Xv, yv, temperature, seed, lr, weight_decay):
    """One bnn.fit_correlated() call -> (val NLL, rhat_max, ess_min, member fail rate); NLL is
    +inf if the fit failed outright or came back non-finite."""
    try:
        bnn_raw, _, diag = bnn.fit_correlated(
            Xt, yt, Xv, yv, seed,
            BNN_N_MEMBERS, BNN_HIDDEN_LAYERS, BNN_EPOCHS, BNN_PATIENCE, BNN_VALIDATION_SPLIT,
            BNN_N_SAMPLES, BNN_WARMUP_STEPS, lr, weight_decay, BNN_N_THINNING,
            BNN_DESIRED_ENERGY_VAR_START, BNN_DESIRED_ENERGY_VAR_END,
            BNN_MAX_SEED_TRIES, BNN_RHAT_THRESHOLD,
            data.TARP_LO, data.TARP_HI, TARP_N_DRAWS, TARP_SEEDS, quiet=True,
            min_good_members=BNN_MIN_GOOD_MEMBERS,
            pretrain_eval_every=BNN_PRETRAIN_EVAL_EVERY, pretrain_min_steps=BNN_PRETRAIN_MIN_STEPS,
            prior_family="standardnormal", prior_kwargs=None, temperature=temperature,
        )
    except RuntimeError:
        return float("inf"), float("nan"), float("nan"), 1.0
    nll = -float(np.mean(bnn_raw["log_prob_true"]))
    return (nll if np.isfinite(nll) else float("inf"), float(diag["rhat_max"]),
            float(diag["ess_min"]), float(diag["member_fail_rate_across_attempts"]))


def main():
    t0 = time.time()
    prior = np.load(OUT_PATH, allow_pickle=True) if OUT_PATH.exists() else None
    if prior is not None and bool(prior["complete"]) and not FULL:
        print(f"{OUT_PATH} already complete -- delete it for a fresh run (or pass --full to "
              f"evaluate the grid points it skipped).")
        return
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lr, weight_decay = load_lr_weight_decay()
    X_train, theta_train, _, _ = data.load_dataset()
    splits = build_repeat_splits(np.asarray(X_train), np.asarray(theta_train))
    seed_base = 100 * SWEEP_SIZES.index(N_TRAIN)
    print(f"hyperparameters from {HP_PATH.name} -> output {OUT_PATH.name}", flush=True)
    print(f"n_train={N_TRAIN} lr={lr:.3e} wd={weight_decay:.3e} grid={T_GRID} repeats={N_REPEATS} "
          f"seeds={[seed_base + k for k in range(N_REPEATS)]} smoke={SMOKE}", flush=True)

    trials = {}  # grid index -> dict
    if FULL and prior is not None:  # re-use what an earlier (bisection) run already evaluated
        for j, t in enumerate(prior["trial_temperatures"]):
            i = T_GRID.index(round(float(t), 1))
            trials[i] = dict(order=int(prior["trial_order"][j]), nll=list(prior["trial_per_repeat_nll"][j]),
                             rhat=list(prior["trial_per_repeat_rhat_max"][j]),
                             ess=list(prior["trial_per_repeat_ess_min"][j]),
                             fail=list(prior["trial_per_repeat_member_fail_rate"][j]),
                             mean_nll=float(prior["trial_mean_nll"][j]), sec=float(prior["trial_elapsed_sec"][j]))
        print(f"--full: re-using {len(trials)} already-evaluated grid points "
              f"{[T_GRID[i] for i in sorted(trials)]}", flush=True)

    def save(complete):
        idx = sorted(trials)
        best = min(idx, key=lambda i: trials[i]["mean_nll"])
        np.savez(
            OUT_PATH, n_train=N_TRAIN, n_repeats=N_REPEATS, lr=lr, weight_decay=weight_decay,
            prior_family="standardnormal", t_grid=np.asarray(T_GRID), hp_file=HP_PATH.name,
            trial_temperatures=np.asarray([T_GRID[i] for i in idx]),
            trial_order=np.asarray([trials[i]["order"] for i in idx]),
            trial_mean_nll=np.asarray([trials[i]["mean_nll"] for i in idx]),
            trial_per_repeat_nll=np.asarray([trials[i]["nll"] for i in idx]),
            trial_per_repeat_rhat_max=np.asarray([trials[i]["rhat"] for i in idx]),
            trial_per_repeat_ess_min=np.asarray([trials[i]["ess"] for i in idx]),
            trial_per_repeat_member_fail_rate=np.asarray([trials[i]["fail"] for i in idx]),
            trial_elapsed_sec=np.asarray([trials[i]["sec"] for i in idx]),
            best_temperature=T_GRID[best], best_val=trials[best]["mean_nll"],
            baseline_t1_val=trials[len(T_GRID) - 1]["mean_nll"] if len(T_GRID) - 1 in trials else np.nan,
            n_samples=BNN_N_SAMPLES, n_thinning=BNN_N_THINNING, warmup_steps=BNN_WARMUP_STEPS,
            n_members=BNN_N_MEMBERS, max_seed_tries=BNN_MAX_SEED_TRIES, val_size=VAL_SIZE,
            elapsed_sec=time.time() - t0, complete=complete, full_grid=FULL,
        )

    def f(i):
        if i not in trials:
            t1 = time.time()
            res = [fit_one(Xt, yt, Xv, yv, T_GRID[i], seed_base + k, lr, weight_decay)
                   for k, (Xt, yt, Xv, yv) in enumerate(splits)]
            nll, rhat, ess, fail = (list(x) for x in zip(*res))
            trials[i] = dict(order=len(trials), nll=nll, rhat=rhat, ess=ess, fail=fail,
                             mean_nll=float(np.mean(nll)), sec=time.time() - t1)
            print(f"trial={len(trials)} T={T_GRID[i]:.1f} -> mean_val_nll={trials[i]['mean_nll']:.4f} "
                  f"per-repeat={[round(x, 4) for x in nll]} rhat_max={[round(x, 2) for x in rhat]} "
                  f"ess_min={[round(x, 1) for x in ess]} fail_rate={[round(x, 2) for x in fail]} "
                  f"({trials[i]['sec']:.0f}s, total {time.time() - t0:.0f}s)", flush=True)
            save(complete=False)
        return trials[i]["mean_nll"]

    f(len(T_GRID) - 1)  # T=1.0 baseline, always
    lo, hi = 0, len(T_GRID) - 1
    if FULL:
        for i in reversed(range(len(T_GRID))):
            f(i)
        lo = hi = min(trials, key=lambda i: trials[i]["mean_nll"])
    while lo < hi:
        mid = (lo + hi) // 2
        if f(mid) <= f(mid + 1):
            hi = mid
        else:
            lo = mid + 1
    save(complete=True)

    best = min(trials, key=lambda i: trials[i]["mean_nll"])
    elapsed = time.time() - t0
    print(f"DONE n_train={N_TRAIN}: best T={T_GRID[best]:.1f} (mean val NLL {trials[best]['mean_nll']:.4f}; "
          f"T=1.0 baseline {trials[len(T_GRID) - 1]['mean_nll']:.4f}); bisection ended at T={T_GRID[lo]:.1f}; "
          f"{len(trials)} of {len(T_GRID)} grid points evaluated; {elapsed:.0f}s -> {OUT_PATH}", flush=True)

    if SMOKE:
        return
    experiment_log.log_run(
        script="scripts/tuning/optimize_temperature_grid.py",
        config=dict(n_train=N_TRAIN, hp_file=HP_PATH.name, t_grid=T_GRID, n_repeats=N_REPEATS, prior_family="standardnormal",
                    lr=lr, weight_decay=weight_decay, n_chain_steps=BNN_N_SAMPLES,
                    n_kept_samples=-(-BNN_N_SAMPLES // BNN_N_THINNING), warmup_steps=BNN_WARMUP_STEPS,
                    max_seed_tries=BNN_MAX_SEED_TRIES, val_size=VAL_SIZE,
                    search="full grid" if FULL else "slope bisection"),
        results=dict(best_temperature=T_GRID[best], best_val=trials[best]["mean_nll"],
                     baseline_t1_val=trials[len(T_GRID) - 1]["mean_nll"],
                     bisection_end_temperature=T_GRID[lo],
                     evaluated={f"{T_GRID[i]:.1f}": trials[i]["mean_nll"] for i in sorted(trials)},
                     n_trials=len(trials), elapsed_sec=elapsed),
        job_id=os.environ.get("SLURM_JOB_ID"),
        output_path=str(OUT_PATH),
        notes="Coarse per-size temperature search under the plain N(0,1) prior (likelihood-only "
              "tempering), replacing the n=100/optimized-prior/old-lr T=0.305 that bnn_temp applied "
              "to every size. Validation NLL only; mixing recorded, not gated. See the script "
              "docstring for the flagged assumptions (unimodality, 3 repeats, failed fit = +inf).",
    )


if __name__ == "__main__":
    main()
