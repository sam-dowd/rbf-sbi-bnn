"""GP-based Bayesian-optimization search over the correlated BDE's (lr, weight_decay), run
separately per training-set size. Mirrors scripts/tuning/optimize_npe.py / scripts/optimize_gaussian.py's
skopt ask/tell design and search space exactly -- same two hyperparameters, same convention.

--- Rewritten 2026-08-28: narrowed from a 7-dimensional architecture+pretraining search to just
(lr, weight_decay) ---

The original version of this script also searched hidden_layers, n_members, epochs, patience, and
validation_split. Narrowed for consistency with optimize_npe.py/optimize_gaussian.py, which only
ever searched (lr, weight_decay) against a fixed architecture (width=16) -- letting BDE alone tune
its architecture/ensemble size per training-set size while the two baselines stay architecturally
fixed would confound "is MILE better" with "did MILE get a bigger/better-tuned net," undermining
the fairness of the three-way comparison this is feeding into. Everything else (hidden_layers,
n_members, epochs, patience, validation_split, and the MCLMC-side knobs below) is now fixed at
scripts/fit/run_experiment.py's / scripts/bde_correlated_perchain_test.py's current production values.

--- Rewritten again 2026-09-04: base the search on a validation set and check that pretraining does
not diverge immediately, to prevent overtraining. Three changes ---

1. **Robust validation split, mirroring optimize_temperature.py's already-fixed design.** The
   PREVIOUS version of this script had the exact same bug optimize_temperature.py was caught and
   fixed for on 2026-08-29: it scored candidates on an INNER 80/20 split of the (already
   n_train-subsampled) training pool -- i.e. it trained (and validated) on a SHRUNK set (e.g. 80/20
   points at n_train=100), not the full n_train the real pipeline actually trains on, and with only
   one single fixed split reused across every BO trial. That bug was found and fixed in
   optimize_temperature.py the same week but never propagated back here. Fixed the same way:
   `build_repeat_splits` below trains on the FULL n_train subsample (bit-identical to
   data.subsample_train(..., repeat=k, base_seed=123)) and validates on a disjoint VAL_SIZE-point
   slice of the training pool's LEFTOVER rows (never touching the n_train training rows, never
   touching the canonical X_test), averaged over N_REPEATS=3 independent (train, val) draws per
   trial so the search isn't chasing noise in one small, fixed validation sample -- the same
   "shaky data" concern raised for optimize_temperature.py applies here too, arguably more so since
   an 8-15-trial GP-BO search hunting for the single best-scoring point is exactly the kind of
   process that can overfit a small fixed validation set even while never touching the training set
   directly.
   **Edge case, not present in optimize_temperature.py (which only ever ran at n_train=100):** at
   n_train=10000 (this script's LARGEST size), the training pool (data.load_dataset()'s 10,000-row
   X_train_pool) is entirely consumed by training -- there are zero leftover rows for a validation
   slice. `build_repeat_splits` falls back to the OLD inner-80/20-split behavor ONLY at that one
   size (flagged via a printed warning), since there is no leftover-pool alternative available.
2. **Train-vs-validation NLL gap tracked and saved per trial**, not just the raw validation score.
   Free to compute -- bnn.fit_correlated() already returns train_log_prob_true (the training-set
   analogue of log_prob_true, added 2026-08-25 for an unrelated diagnostic) alongside
   log_prob_true, so train NLL costs nothing extra to read out per repeat. Saved per trial as
   trial_mean_train_nll / trial_mean_val_nll / trial_mean_gap (train_nll - val_nll; more negative =
   training set fit much better than validation = larger generalization gap) so a later read can
   directly see whether the winning candidate is the best on validation NLL AND has a small gap, or
   the best on validation NLL only because it happens to fit hard and get lucky on one validation
   draw -- exactly the "prevent overtraining" question raised.
3. **Pretraining loss curve captured for the winning candidate.** bde's own BdeBuilder already
   computes a full per-member, per-epoch train/val loss history internally
   (bde/bde_builder.py:BdeBuilder._training_loop's self.history) but nothing in this project
   previously read it back out -- bnn.py's CorrelatedBdeRegressor.fit()/_fit_correlated()/
   fit_correlated() were extended (2026-09-04) to surface it as
   bnn_raw["pretrain_loss_history"], a list of per-kept-member {epoch, trainloss, valloss,
   stop_epoch} arrays. After the search picks a size's best (lr, weight_decay), this script does
   ONE additional refit (repeat=0's split, not counted against the search budget) purely to save
   that curve into the output .npz under per-member keys (winning_member{i}_epoch/trainloss/
   valloss/stop_epoch) -- lets "did it diverge in the first few epochs, or run away late" be
   answered by actually plotting the curve, not just inferred from the final NLL number.

--- Cost estimate (measured, not guessed) ---

`sacct` on real completed jobs at the CURRENT production budget (n_samples=10000,
warmup_steps=50000, n_thinning=10 -- raised 2026-08-27 toward the MILE paper's own defaults, see
scripts/bde_correlated_perchain_test.py) shows a single bnn.fit_correlated() call taking 81-153s
wall-clock at n_train=100 (jobs 57660570, 57685190, 57685891). The MCLMC warmup/sampling budget is
fixed regardless of n_train, so cost is not expected to shrink much at smaller sizes the way
NPE/Gaussian's early-stopped training does -- but it's also not expected to grow much at larger
n_train either, since the fixed warmup/sampling cost dominates over the (cheap, size-dependent)
pretraining forward/backward passes. **With N_REPEATS=3 (2026-09-04), one BO trial now costs
roughly 3x a single fit (~4-8 min instead of ~1.5-2.5 min)** -- a direct, acknowledged consequence
of item 1 above: the search gets proportionally FEWER trials within the same
PER_SIZE_TIME_BUDGET_SEC than the original single-split version did (roughly 3-5 trials instead of
8-15 at the original 1200s budget). PER_SIZE_TIME_BUDGET_SEC is NOT increased to compensate in this
version -- left as an explicit follow-up once a first real run's own logged elapsed_sec/n_trials
shows whether that few trials is enough for the 2D (lr, weight_decay) space.

--- Objective (score used by opt.tell -- unchanged in kind, more robust in construction) ---

Mean of -mean(log_prob_true) (average NLL) across N_REPEATS held-out validation splits. Lower is
better. A trial where every repeat fails (RuntimeError or non-finite NLL, one redo attempted per
repeat first) scores +inf, same non-finite-loss convention the rest of this project's optimize_*.py
scripts use.

--- Output ---

One file per training-set size (results/bde_params_n{n}.npz), same as before -- so running
different sizes as parallel Slurm array tasks (see slurm/optimize_bde.sbatch) can't race on a
shared results file. Resumable: a size already present on disk is skipped -- delete the file first
to force a fresh search at that size.

Usage: python legacy/optimize_bde.py [n_train ...]   (default: all of SIZES)
"""
import os
import sys
import time
from pathlib import Path

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # this repository's src/
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401

import numpy as np
from skopt import Optimizer
from skopt.space import Real

from rbf_sbi import data
from rbf_sbi import bnn
from rbf_sbi import experiment_log

SIZES = [10000, 5000, 1000, 500, 100, 50]
PER_SIZE_TIME_BUDGET_SEC = 1200  # ~20 min/size -- see cost estimate above (NOT increased for the
# 2026-09-04 multi-repeat rewrite; expect ~3-5 trials/size instead of the original 8-15, see docstring.

N_REPEATS = 3  # 2026-09-04: mirrors optimize_temperature.py's fix for the same shaky-single-split concern.
SUBSAMPLE_BASE_SEED = 123  # matches data.subsample_train's own default / run_experiment.py's convention
VAL_SIZE = 1000  # size of the held-out validation slice drawn from the training pool's leftover
                  # rows (never the n_train rows) -- matches the canonical X_test size so NLL
                  # magnitudes here are roughly comparable to test_nll elsewhere. Not achievable at
                  # n_train=10000 (whole pool consumed) -- see build_repeat_splits's fallback.
REDO_SEED_OFFSET = 5000

# ---- Fixed at scripts/fit/run_experiment.py's / scripts/bde_correlated_perchain_test.py's current
# production values -- NOT searched. Architecture/ensemble-size/pretraining-schedule fields were
# searched by this script's original version; narrowed away per the module docstring above.
# MCLMC-side fields were never searched (MILE's own ablations found these insensitive). ----
BNN_N_MEMBERS = 3
BNN_HIDDEN_LAYERS = [16, 16, 16]
BNN_EPOCHS, BNN_PATIENCE, BNN_VALIDATION_SPLIT = 2000, 5, 0.2
BNN_N_SAMPLES, BNN_WARMUP_STEPS, BNN_N_THINNING = 10000, 50000, 10
BNN_DESIRED_ENERGY_VAR_START, BNN_DESIRED_ENERGY_VAR_END = 0.5, 0.1
BNN_MAX_SEED_TRIES, BNN_RHAT_THRESHOLD = 15, 1.01
BNN_MIN_GOOD_MEMBERS = BNN_N_MEMBERS  # 3 -- matches production; search-time cost is already cheap
# enough (2D search, ~100-150s/fit) that there's no need to economize via a lower value here.
BNN_PRETRAIN_EVAL_EVERY, BNN_PRETRAIN_MIN_STEPS = 100, 500

# Prior-agnostic: prior_family/prior_kwargs only affect the MCLMC log-posterior's prior term, not
# the pretraining stage lr/weight_decay control -- so a single search (under the standard prior)
# covers both the standard-prior and optimized-prior BNN variants scripts/fit/run_experiment.py can run.
PRIOR_FAMILY, PRIOR_KWARGS = "standardnormal", None

# Small -- TARP/SBC aren't the search objective (only log_prob_true is), just cheap to leave
# enabled for visibility into what a candidate's calibration looks like from the search log.
TARP_N_DRAWS = 200
TARP_SEEDS = list(range(5))

SEARCH_SPACE = [
    Real(1e-5, 1e-1, prior="log-uniform", name="lr"),
    Real(1e-7, 1e-2, prior="log-uniform", name="weight_decay"),
]

RESULTS_DIR = RESULTS


def out_path(n):
    """results/bde_params_n{n}.npz, the file scripts/fit/run_experiment.py reads. The environment
    variable BDE_PARAMS_TAG (2026-10-01), when set, writes results/bde_params_{tag}_n{n}.npz
    instead -- for re-running a size's search without replacing the file that in-flight sweep
    jobs are still reading their (lr, weight_decay) from."""
    tag = os.environ.get("BDE_PARAMS_TAG", "")
    return RESULTS_DIR / (f"bde_params_{tag}_n{n}.npz" if tag else f"bde_params_n{n}.npz")


def build_repeat_splits(X_train, theta_train, n):
    """N_REPEATS independent (train, val) splits, precomputed once and reused unchanged across
    every BO trial for this size -- mirrors optimize_temperature.py's build_repeat_splits exactly.
    Xt/yt is the FULL n-sized subsample (bit-identical to data.subsample_train(..., repeat=k,
    base_seed=SUBSAMPLE_BASE_SEED)), Xv/yv is a disjoint VAL_SIZE-point slice of the training
    pool's leftover rows.

    Edge case (2026-09-04): at n == len(X_train) (n_train=10000, the whole pool), there are no
    leftover rows at all -- falls back to an inner 80/20 split of the n-sized subsample itself
    (the OLD, pre-fix behavior) for that size only, printing a warning so this is visible in the
    job log rather than silently happening."""
    pool_size = len(X_train)
    splits = []
    for k in range(N_REPEATS):
        perm = np.random.default_rng(SUBSAMPLE_BASE_SEED + k).permutation(pool_size)
        train_idx = perm[:n]
        leftover = pool_size - n
        if leftover <= 0:
            print(f"  [warn] n_train={n} consumes the entire {pool_size}-row pool -- no leftover "
                  f"rows for a real held-out validation slice. Falling back to an inner 80/20 "
                  f"split of the n_train subsample itself for repeat={k} (the pre-2026-09-04 "
                  f"behavior) -- this repeat's validation set is NOT independent of training data "
                  f"the way the other sizes' repeats are.", flush=True)
            n_val = max(1, int(0.2 * n))
            val_idx = train_idx[-n_val:]
            train_idx = train_idx[:-n_val]
        else:
            val_idx = perm[n:n + min(VAL_SIZE, leftover)]
        Xt, yt = X_train[train_idx], theta_train[train_idx]
        Xv, yv = X_train[val_idx], theta_train[val_idx]
        splits.append((Xt, yt, Xv, yv))
    return splits


def fit_one(Xt, yt, Xv, yv, lr, weight_decay, seed):
    """One bnn.fit_correlated() call -> (val_nll, train_nll), or (None, None) if it failed
    outright (RuntimeError from fit_correlated's own internal seed-retry loop) or came back
    non-finite. train_nll comes from the already-computed train_log_prob_true field -- free."""
    tarp_lo, tarp_hi = data.TARP_LO, data.TARP_HI
    try:
        bnn_raw, _, _ = bnn.fit_correlated(
            Xt, yt, Xv, yv, seed,
            BNN_N_MEMBERS, BNN_HIDDEN_LAYERS, BNN_EPOCHS, BNN_PATIENCE, BNN_VALIDATION_SPLIT,
            BNN_N_SAMPLES, BNN_WARMUP_STEPS, lr, weight_decay, BNN_N_THINNING,
            BNN_DESIRED_ENERGY_VAR_START, BNN_DESIRED_ENERGY_VAR_END,
            BNN_MAX_SEED_TRIES, BNN_RHAT_THRESHOLD,
            tarp_lo, tarp_hi, TARP_N_DRAWS, TARP_SEEDS, quiet=True,
            min_good_members=BNN_MIN_GOOD_MEMBERS,
            pretrain_eval_every=BNN_PRETRAIN_EVAL_EVERY, pretrain_min_steps=BNN_PRETRAIN_MIN_STEPS,
            prior_family=PRIOR_FAMILY, prior_kwargs=PRIOR_KWARGS,
        )
    except RuntimeError:
        return None, None

    val_nll = -float(np.mean(bnn_raw["log_prob_true"]))
    train_nll = -float(np.mean(bnn_raw["train_log_prob_true"]))
    if not (np.isfinite(val_nll) and np.isfinite(train_nll)):
        return None, None
    return val_nll, train_nll


def evaluate_candidate(lr, weight_decay, splits, seed_base, log_prefix=""):
    """Averages val/train NLL over all N_REPEATS splits, one redo per repeat on failure -- mirrors
    optimize_temperature.py's evaluate_temperature. Returns
    (mean_val_nll, mean_train_nll, per_repeat_val, per_repeat_train, n_redos)."""
    per_repeat_val, per_repeat_train = [], []
    n_redos = 0
    for k, (Xt, yt, Xv, yv) in enumerate(splits):
        seed = seed_base + k
        val_nll, train_nll = fit_one(Xt, yt, Xv, yv, lr, weight_decay, seed)
        if val_nll is None:
            n_redos += 1
            redo_seed = seed_base + k + REDO_SEED_OFFSET
            print(f"{log_prefix}repeat={k} failed (non-finite or RuntimeError) at seed={seed} "
                  f"-- redoing once with seed={redo_seed}", flush=True)
            val_nll, train_nll = fit_one(Xt, yt, Xv, yv, lr, weight_decay, redo_seed)
            if val_nll is None:
                print(f"{log_prefix}repeat={k} failed again on redo -- treating as +inf", flush=True)
                val_nll, train_nll = float("inf"), float("inf")
        per_repeat_val.append(val_nll)
        per_repeat_train.append(train_nll)
    return (float(np.mean(per_repeat_val)), float(np.mean(per_repeat_train)),
            per_repeat_val, per_repeat_train, n_redos)


def search_one_size(X_train, theta_train, n, seed=123):
    splits = build_repeat_splits(X_train, theta_train, n)
    print(f"  n_train={n:>6}: built {N_REPEATS} (train, val) splits, reused across every trial.", flush=True)

    opt = Optimizer(SEARCH_SPACE, base_estimator="GP", acq_func="EI", random_state=seed)

    trials = []
    t0 = time.time()
    n_trials = 0
    while n_trials < 1 or (time.time() - t0) < PER_SIZE_TIME_BUDGET_SEC:
        lr, weight_decay = opt.ask()
        trial_t0 = time.time()
        mean_val, mean_train, per_val, per_train, n_redos = evaluate_candidate(
            lr, weight_decay, splits, seed_base=seed + 1000 * n_trials,
            log_prefix=f"  [n={n} trial {n_trials + 1}] lr={lr:.2e} wd={weight_decay:.2e} ",
        )
        opt.tell([lr, weight_decay], mean_val)
        n_trials += 1
        gap = mean_train - mean_val
        trial_elapsed = time.time() - trial_t0
        trials.append((lr, weight_decay, mean_val, mean_train, gap, per_val, per_train, n_redos, trial_elapsed))
        print(f"  n_train={n:>6} trial={n_trials:>3} lr={lr:.2e} wd={weight_decay:.2e} -> "
              f"val_nll={mean_val:.4f} train_nll={mean_train:.4f} gap={gap:+.4f} "
              f"(redos={n_redos}, {trial_elapsed:.0f}s, elapsed {time.time() - t0:.0f}s)", flush=True)

    best_idx = int(np.argmin(opt.yi))
    best_lr, best_wd = opt.Xi[best_idx]
    best_val = float(opt.yi[best_idx])
    best_train = trials[best_idx][3]
    best_gap = trials[best_idx][4]
    elapsed = time.time() - t0
    print(f"n_train={n:>6} | {n_trials} trials in {elapsed:.0f}s | "
          f"best lr={best_lr:.2e} wd={best_wd:.2e} val_nll={best_val:.4f} "
          f"train_nll={best_train:.4f} gap={best_gap:+.4f}", flush=True)
    return dict(
        best_lr=best_lr, best_wd=best_wd, best_val=best_val, best_train=best_train, best_gap=best_gap,
        n_trials=n_trials, elapsed=elapsed, trials=trials, splits=splits,
    )


def capture_winning_loss_curve(splits, lr, weight_decay, seed):
    """One extra bnn.fit_correlated() call at the winning (lr, weight_decay), using repeat=0's
    split, purely to surface bnn_raw['pretrain_loss_history'] (see module docstring item 3) -- not
    counted against the search's own trial budget/score. Returns a dict of flat, npz-savable arrays
    (winning_member{i}_epoch/trainloss/valloss/stop_epoch), or {} if this refit itself fails
    (should be rare -- the winning candidate already succeeded during the search -- but not
    treated as fatal to the rest of this script's output if it does)."""
    Xt, yt, Xv, yv = splits[0]
    tarp_lo, tarp_hi = data.TARP_LO, data.TARP_HI
    try:
        bnn_raw, _, _ = bnn.fit_correlated(
            Xt, yt, Xv, yv, seed,
            BNN_N_MEMBERS, BNN_HIDDEN_LAYERS, BNN_EPOCHS, BNN_PATIENCE, BNN_VALIDATION_SPLIT,
            BNN_N_SAMPLES, BNN_WARMUP_STEPS, lr, weight_decay, BNN_N_THINNING,
            BNN_DESIRED_ENERGY_VAR_START, BNN_DESIRED_ENERGY_VAR_END,
            BNN_MAX_SEED_TRIES, BNN_RHAT_THRESHOLD,
            tarp_lo, tarp_hi, TARP_N_DRAWS, TARP_SEEDS, quiet=True,
            min_good_members=BNN_MIN_GOOD_MEMBERS,
            pretrain_eval_every=BNN_PRETRAIN_EVAL_EVERY, pretrain_min_steps=BNN_PRETRAIN_MIN_STEPS,
            prior_family=PRIOR_FAMILY, prior_kwargs=PRIOR_KWARGS,
        )
    except RuntimeError as e:
        print(f"  [warn] loss-curve capture refit failed ({e}) -- omitting loss-curve arrays "
              f"from this size's output.", flush=True)
        return {}

    out = {}
    for i, h in enumerate(bnn_raw["pretrain_loss_history"]):
        out[f"winning_member{i}_epoch"] = h["epoch"]
        out[f"winning_member{i}_trainloss"] = h["trainloss"]
        out[f"winning_member{i}_valloss"] = h["valloss"]
        out[f"winning_member{i}_stop_epoch"] = h["stop_epoch"]
    return out


def save_one_size(n, result, loss_curve_arrays):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path(n).parent / (out_path(n).stem + ".tmp.npz")
    trials = result["trials"]
    np.savez_compressed(
        tmp_path,
        n_train=n, val_loss=result["best_val"], n_trials=result["n_trials"], elapsed_sec=result["elapsed"],
        lr=result["best_lr"], weight_decay=result["best_wd"],
        # 2026-09-04 additions -- see module docstring items 1-2.
        n_repeats=N_REPEATS, val_size=VAL_SIZE,
        best_train_nll=result["best_train"], best_gap=result["best_gap"],
        trial_lr=np.array([t[0] for t in trials]), trial_weight_decay=np.array([t[1] for t in trials]),
        trial_mean_val_nll=np.array([t[2] for t in trials]), trial_mean_train_nll=np.array([t[3] for t in trials]),
        trial_gap=np.array([t[4] for t in trials]), trial_n_redos=np.array([t[7] for t in trials]),
        trial_elapsed_sec=np.array([t[8] for t in trials]),
        # Fixed (not searched) settings this result assumes, for provenance -- a later reader
        # shouldn't have to cross-reference this script to know what was held fixed.
        hidden_layers=np.array(BNN_HIDDEN_LAYERS), n_members=BNN_N_MEMBERS,
        epochs=BNN_EPOCHS, patience=BNN_PATIENCE, validation_split=BNN_VALIDATION_SPLIT,
        warmup_steps=BNN_WARMUP_STEPS, n_samples=BNN_N_SAMPLES, n_thinning=BNN_N_THINNING,
        desired_energy_var_start=BNN_DESIRED_ENERGY_VAR_START,
        desired_energy_var_end=BNN_DESIRED_ENERGY_VAR_END, min_good_members=BNN_MIN_GOOD_MEMBERS,
        pretrain_eval_every=BNN_PRETRAIN_EVAL_EVERY, pretrain_min_steps=BNN_PRETRAIN_MIN_STEPS,
        **loss_curve_arrays,
    )
    tmp_path.rename(out_path(n))


def main():
    sizes_to_run = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else SIZES

    print(f"Searching (lr, weight_decay) only, N_REPEATS={N_REPEATS} val splits/trial "
          f"(2026-09-04 rewrite). Fixed this run: hidden_layers={BNN_HIDDEN_LAYERS} "
          f"n_members={BNN_N_MEMBERS} epochs={BNN_EPOCHS} patience={BNN_PATIENCE} "
          f"validation_split={BNN_VALIDATION_SPLIT} warmup_steps={BNN_WARMUP_STEPS} "
          f"n_samples={BNN_N_SAMPLES} n_thinning={BNN_N_THINNING} "
          f"desired_energy_var=({BNN_DESIRED_ENERGY_VAR_START}, {BNN_DESIRED_ENERGY_VAR_END}) "
          f"min_good_members={BNN_MIN_GOOD_MEMBERS}", flush=True)

    X_train, theta_train, _, _ = data.load_dataset()

    for n in sizes_to_run:
        if out_path(n).exists():
            print(f"Skipping n_train={n} (already have {out_path(n).name})")
            continue
        result = search_one_size(X_train, theta_train, n)
        print(f"  n_train={n:>6}: capturing pretraining loss curve for the winning candidate...", flush=True)
        loss_curve_arrays = capture_winning_loss_curve(
            result["splits"], result["best_lr"], result["best_wd"], seed=123,
        )
        save_one_size(n, result, loss_curve_arrays)
        print(f"Saved: {out_path(n)}")

        # 2026-09-04: wired in on this rewrite -- this script previously had no
        # log_run() call at all. One entry per size (not one for the whole multi-size invocation),
        # since each size is its own independent search with its own result.
        experiment_log.log_run(
            script="legacy/optimize_bde.py",
            config=dict(
                n_train=n, n_repeats=N_REPEATS, val_size=VAL_SIZE,
                hidden_layers=BNN_HIDDEN_LAYERS, n_members=BNN_N_MEMBERS,
                epochs=BNN_EPOCHS, patience=BNN_PATIENCE, validation_split=BNN_VALIDATION_SPLIT,
                warmup_steps=BNN_WARMUP_STEPS, n_samples=BNN_N_SAMPLES, n_thinning=BNN_N_THINNING,
                desired_energy_var_start=BNN_DESIRED_ENERGY_VAR_START,
                desired_energy_var_end=BNN_DESIRED_ENERGY_VAR_END,
                min_good_members=BNN_MIN_GOOD_MEMBERS,
                pretrain_eval_every=BNN_PRETRAIN_EVAL_EVERY, pretrain_min_steps=BNN_PRETRAIN_MIN_STEPS,
            ),
            results=dict(
                n_trials=result["n_trials"], elapsed_sec=result["elapsed"],
                best_lr=result["best_lr"], best_weight_decay=result["best_wd"],
                best_val_nll=result["best_val"], best_train_nll=result["best_train"],
                best_gap=result["best_gap"],
            ),
            job_id=os.environ.get("SLURM_JOB_ID"),
            output_path=str(out_path(n)),
            notes="2026-09-04 rewrite: validation split now trains on the FULL n_train subsample "
                  "and validates on a disjoint leftover-pool slice (previously an inner 80/20 "
                  "split of a shrunk subsample -- the same bug already found/fixed in "
                  "optimize_temperature.py), averaged over n_repeats splits per trial; "
                  "train-vs-validation NLL gap now tracked per trial; pretraining loss curve "
                  "captured for the winning candidate (validation-set-based search, check for early "
                  "divergence/overtraining).",
        )


if __name__ == "__main__":
    main()
