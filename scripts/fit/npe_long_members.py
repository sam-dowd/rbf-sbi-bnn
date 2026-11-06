"""Long-training NPE ensemble members (2026-11-04): does NPE's large-n plateau (test NLL -2.067 / -2.105 at n=5,000 /
10,000 vs the sampled BNN's -2.226 / -2.284) come from its training budget? Fits individual members of the 12-flow
NPE ensemble with a larger step cap and patience, everything else EXACTLY as `final_npe_ens12` in
scripts/fit/run_experiment.py: same training subsample (data.subsample_train), split (metrics.split with the run seed),
seed / lr / wd (read from results/raw/final/final_npe_ens12_n{n}_r{r}.npz and asserted), member seeds
(seed + 10,000 k), architecture, optimiser, eval interval, TARP / posterior-sample settings.

One member per invocation item; each is written to its own file immediately, and an existing member file is skipped,
so a packed debug job that hits its time limit loses only the members in flight. scripts/fit/npe_long_merge.py combines
the 12 members of a repeat into final_npe_ens12_long / final_npe_long (and writes the experiment-log entry -- the
member files are intermediate, so this script does not log).

ASSUMPTION (flagged): the "long" budget -- steps 200,000 (10x NPE_STEPS) and patience 100 evaluations (10,000 steps
without improvement; 5x NPE_PATIENCE = 2,000 steps) -- is a choice, not from a source. Raising the cap alone would
change nothing for flows that already stop early, hence the patience increase.

--compare: also compare the member's test log_prob_true with the saved member_log_prob_true[k] of final_npe_ens12
(only meaningful with the default budget --steps 20000 --patience 20: a reproduction check of this script's setup
and of the npe.fit info_out change on GPU).

Usage: python scripts/fit/npe_long_members.py N R K [K ...] [--steps 200000] [--patience 100] [--out-dir DIR] [--compare]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # this repository's src/
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401
from rbf_sbi import npe       # noqa: E402  torch FIRST, before anything that pulls in jax (as run_experiment.py; see
#                                first seen 2026-09-15: CUBLAS_STATUS_NOT_INITIALIZED otherwise -- seen again
#                                2026-11-04, jobs 60800363/64, with data/metrics imported first)
from rbf_sbi import data      # noqa: E402
from rbf_sbi import metrics   # noqa: E402

FINAL = RESULTS / "raw" / "final"
# Copied from scripts/fit/run_experiment.py (2026-11-04): NPE_EVAL_EVERY, TARP_N_DRAWS, NPE_N_POSTERIOR_SAMPLES, TARP_SEEDS
EVAL_EVERY, TARP_N_DRAWS, N_POSTERIOR, TARP_SEEDS = 100, 1000, 5000, list(range(20))
DEFAULT_OUT = SCRATCH / "sbi_npe_long" / "members"


def member_path(out_dir, n, r, k, steps, patience):
    return Path(out_dir) / f"npe_long_s{steps}_p{patience}_n{n}_r{r}_k{k}.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("r", type=int)
    ap.add_argument("ks", type=int, nargs="+")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    ref = np.load(FINAL / f"final_npe_ens12_n{a.n}_r{a.r}.npz", allow_pickle=True)
    seed, lr, wd = int(ref["seed"]), float(ref["lr"]), float(ref["wd"])
    X_train, theta_train, X_test, theta_test = data.load_dataset()
    Xb, yb = data.subsample_train(X_train, theta_train, a.n, a.r)
    assert np.array_equal(np.asarray(Xb).reshape(len(Xb), -1), ref["X_train"]), "subsample differs from final_npe_ens12"
    Xt, yt, Xv, yv = metrics.split(np.asarray(Xb), np.asarray(yb), seed)

    for k in a.ks:
        out = member_path(a.out_dir, a.n, a.r, k, a.steps, a.patience)
        if out.exists():
            print(f"skip n={a.n} r={a.r} k={k}: {out.name} exists")
            continue
        mseed = seed if k == 0 else seed + 10_000 * k
        assert mseed == int(ref["member_seeds"][k]), f"member seed {mseed} != saved {int(ref['member_seeds'][k])}"
        info, t0 = {}, time.time()
        raw, f = npe.fit(Xt, yt, Xv, yv, X_test, theta_test, mseed, lr, wd, a.steps, EVAL_EVERY, a.patience,
                         TARP_N_DRAWS, N_POSTERIOR, TARP_SEEDS, data.TARP_LO, data.TARP_HI, info_out=info)
        sec = time.time() - t0
        try:
            import torch
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        except Exception:
            gpu = "unknown"
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, n_train=a.n, repeat=a.r, seed=seed, member=k, member_seed=mseed, lr=lr, wd=wd,
                            hp_file=str(ref["hp_file"]), steps=a.steps, patience=a.patience, eval_every=EVAL_EVERY,
                            fit_sec=sec, gpu_name=gpu, f=f, steps_run=info["steps_run"], best_step=info["best_step"],
                            stopped_early=info["stopped_early"], val_history=info["val_history"],
                            **{key: np.asarray(v) for key, v in raw.items()})
        os.replace(tmp, out)
        nll = -float(np.mean(raw["log_prob_true"]))
        ref_nll = -float(np.mean(ref["member_log_prob_true"][k]))
        print(f"done n={a.n} r={a.r} k={k}: {sec:.0f} s, steps_run {info['steps_run']}, best_step {info['best_step']}, "
              f"stopped_early {info['stopped_early']}, member NLL {nll:+.4f} (default-budget member {ref_nll:+.4f}) "
              f"[{gpu}]", flush=True)
        if a.compare:
            same = np.array_equal(raw["log_prob_true"], ref["member_log_prob_true"][k])
            d = float(np.max(np.abs(raw["log_prob_true"] - ref["member_log_prob_true"][k])))
            print(f"COMPARE n={a.n} r={a.r} k={k} vs saved member: {'IDENTICAL' if same else f'DIFFERENT (max|d| {d:.3g})'}",
                  flush=True)


if __name__ == "__main__":
    main()
