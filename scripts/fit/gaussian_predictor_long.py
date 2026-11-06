"""Gaussian predictor (= the BNN without sampling) with pretraining run to convergence (2026-11-04).

The paper-ready Gaussian predictor `final_bnn_pretrained_ens12_pretuned` is the 12 pretrained networks of the
pretraining-tuned N(0,1) BNN run. Its pretraining is capped at BNN_EPOCHS = 2,000 full-batch AdamW steps with
patience 5 evaluations (x 100 steps), and most members hit the cap (pretrain_stop_epoch = -1: 12/12 at n=50 and
1,000, 11/12 at 500, 8-9/12 at 5,000 / 10,000, r0). This script refits it with EVERYTHING identical -- same
bnn.fit_pretrained_ensemble (the function verified bit-identical to fit_correlated's pretraining), same training
subsample, seed, lr / wd (read from and asserted against the existing final_bnn_pretrained_ens12_pretuned file),
12 members, hidden [16,16,16], soft-sign, validation split 0.2, eval every 100 steps, 500-step floor, min_delta 1e-4,
TARP settings -- except the step cap and the patience.

ASSUMPTION (flagged): the long budget -- cap 200,000 steps (100x), patience 100 evaluations = 10,000 steps without a
validation improvement (20x) -- is a choice (it mirrors the long NPE budget), not from a source. "Converged" here
means: every member stopped by patience (pretrain_stop_epoch != -1); members still at the cap are reported, and such a
fit should be rerun with a larger cap before it is trusted.

--default-budget reproduces the existing file's budget (2,000 / 5) into --out-dir and compares with it (a check of this
script's setup; at n=50 these networks were bit-identical across GPU runs before, 2026-11-02).

Writes results/raw/final/final_bnn_pretrained_ens12_pretuned_long_n{n}_r{r}.npz and
final_bnn_pretrained_pretuned_long_n{n}_r{r}.npz (the pretuned files' schema + pretrain_epochs_cap,
pretrain_patience, pretrain_best_val, pretrain_last_epoch); skips existing outputs; one experiment-log entry per call.
Usage: python scripts/fit/gaussian_predictor_long.py N R [R ...] [--epochs 200000] [--patience 100] [--out-dir DIR]
       [--default-budget]      (R may be a range "0-4")
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
from rbf_sbi import data            # noqa: E402
from rbf_sbi import bnn             # noqa: E402
from rbf_sbi import experiment_log  # noqa: E402

FINAL = RESULTS / "raw" / "final"
SRC = "final_bnn_pretrained_ens12_pretuned"
OUT_ENS, OUT_ONE = "final_bnn_pretrained_ens12_pretuned_long", "final_bnn_pretrained_pretuned_long"
# Copied from scripts/fit/run_experiment.py (2026-11-04): FINAL_N_MEMBERS, BNN_HIDDEN_LAYERS, BNN_VALIDATION_SPLIT,
# BNN_PRETRAIN_EVAL_EVERY / MIN_STEPS, BNN_ACTIVATION, BNN_WARMUP_STEPS / N_SAMPLES / N_THINNING, energy targets
# (the last five are constructor-only, unused by pretraining), TARP_N_DRAWS, TARP_SEEDS.
N_MEMBERS, HIDDEN, VAL_SPLIT, EVAL_EVERY, MIN_STEPS, ACT = 12, [16, 16, 16], 0.2, 100, 500, "soft_sign"
WARMUP, N_SAMPLES, THIN, EV0, EV1 = 50000, 10000, 10, 0.5, 0.1
TARP_N_DRAWS, TARP_SEEDS = 1000, list(range(20))
DEFAULT_BUDGET = (2000, 5)   # BNN_EPOCHS, BNN_PATIENCE


def parse(tokens):
    out = []
    for t in tokens:
        if "-" in t:
            a, b = t.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(t))
    return out


def gpu_name():
    try:
        import jax
        d = jax.devices()[0]
        return getattr(d, "device_kind", str(d))
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("repeats", nargs="+")
    ap.add_argument("--epochs", type=int, default=200_000)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--out-dir", type=Path, default=FINAL)
    ap.add_argument("--default-budget", action="store_true")
    a = ap.parse_args()
    epochs, patience = DEFAULT_BUDGET if a.default_budget else (a.epochs, a.patience)
    if a.default_budget:
        assert a.out_dir.resolve() != FINAL.resolve(), "--default-budget is a check: give a scratch --out-dir"
    a.out_dir.mkdir(parents=True, exist_ok=True)
    X_train, theta_train, X_test, theta_test = data.load_dataset()
    rows = []
    for r in parse(a.repeats):
        ens_out, one_out = a.out_dir / f"{OUT_ENS}_n{a.n}_r{r}.npz", a.out_dir / f"{OUT_ONE}_n{a.n}_r{r}.npz"
        if ens_out.exists() and one_out.exists():
            print(f"skip n={a.n} r={r}: outputs exist", flush=True)
            continue
        ref = np.load(FINAL / f"{SRC}_n{a.n}_r{r}.npz", allow_pickle=True)
        seed, lr, wd = int(ref["seed"]), float(ref["lr"]), float(ref["wd"])
        Xb, yb = data.subsample_train(X_train, theta_train, a.n, r)
        assert np.array_equal(np.asarray(Xb).reshape(len(Xb), -1), ref["X_train"]), "subsample differs from source"
        t0 = time.time()
        ens_raw, ens_f, single_raw, single_f, diag = bnn.fit_pretrained_ensemble(
            Xb, yb, X_test, theta_test, seed, N_MEMBERS, HIDDEN, epochs, patience, VAL_SPLIT, lr, wd,
            data.TARP_LO, data.TARP_HI, TARP_N_DRAWS, TARP_SEEDS, quiet=True, pretrain_eval_every=EVAL_EVERY,
            pretrain_min_steps=MIN_STEPS, activation=ACT, warmup_steps=WARMUP, n_samples=N_SAMPLES, n_thinning=THIN,
            desired_energy_var_start=EV0, desired_energy_var_end=EV1)
        sec = time.time() - t0
        stop = np.asarray(diag["pretrain_stop_epoch"])
        hist = diag["pretrain_loss_history"]
        best_val = np.array([float(np.nanmin(np.asarray(h["valloss"], dtype=float))) for h in hist])
        last_ep = np.array([int(np.asarray(h["epoch"])[-1]) for h in hist])
        common = dict(n_train=a.n, repeat=r, seed=seed, lr=lr, wd=wd, X_test=ref["X_test"], theta_test=ref["theta_test"],
                      X_train=ref["X_train"], theta_train=ref["theta_train"], hp_file=str(ref["hp_file"]),
                      gpu_name=gpu_name(), source_model=f"{SRC} (refit, long pretraining)", fit_sec=sec,
                      n_members_total=N_MEMBERS, member_finite=diag["member_finite"], pretrain_stop_epoch=stop,
                      seeds_used=np.full(N_MEMBERS, seed),   # same convention as the source file (bde derives members)
                      pretrain_epochs_cap=epochs, pretrain_patience=patience, pretrain_best_val=best_val,
                      pretrain_last_epoch=last_ep)
        tmp = one_out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **common, **single_raw, f=single_f, n_ensemble=1); os.replace(tmp, one_out)
        tmp = ens_out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **common, **ens_raw, f=ens_f, n_ensemble=diag["n_members_used"],
                            pretrain_weights=diag["pretrain_weights"]); os.replace(tmp, ens_out)
        nll, nll0 = -float(np.mean(ens_raw["log_prob_true"])), -float(np.mean(ref["log_prob_true"]))
        capped = int((stop == -1).sum())
        rows.append(dict(r=r, nll=nll, nll_cap2000=nll0, capped=capped, last_epoch_max=int(last_ep.max()), sec=sec))
        print(f"done n={a.n} r={r}: {sec:.0f} s, ens12 NLL {nll:+.4f} (2,000-step version {nll0:+.4f}); members at cap "
              f"{capped}/{N_MEMBERS}; stop epochs {stop.tolist()}; members finite {int(diag['n_members_used'])}/{N_MEMBERS}",
              flush=True)
        if a.default_budget:
            for key in ("log_prob_true", "mean", "aleatoric", "epistemic"):
                same = np.array_equal(ens_raw[key], ref[key])
                print(f"COMPARE n={a.n} r={r} {key} vs saved {SRC}: {'IDENTICAL' if same else 'DIFFERENT (max|d| %.3g)' % float(np.max(np.abs(ens_raw[key] - ref[key])))}"
                      f"; stop epochs saved {np.asarray(ref['pretrain_stop_epoch']).tolist()}", flush=True)
    if rows and not a.default_budget:
        experiment_log.log_run(
            "scripts/fit/gaussian_predictor_long.py", job_id=os.environ.get("SLURM_JOB_ID"),
            config=dict(model=OUT_ENS, single_model=OUT_ONE, n_train=a.n, repeats=[x["r"] for x in rows],
                        n_members=N_MEMBERS, pretrain_epochs_cap=epochs, pretrain_patience_evals=patience,
                        eval_every=EVAL_EVERY, lr=float(ref["lr"]), wd=float(ref["wd"]), source=SRC),
            results=dict(test_nll_ens12=[round(x["nll"], 4) for x in rows],
                         test_nll_ens12_cap2000=[round(x["nll_cap2000"], 4) for x in rows],
                         members_at_cap=[x["capped"] for x in rows], max_last_epoch=[x["last_epoch_max"] for x in rows],
                         fit_sec=[round(x["sec"], 1) for x in rows]),
            output_path=f"results/raw/final/{OUT_ENS}_n{a.n}_r*.npz",
            notes="Gaussian predictor refit with pretraining run to convergence (only cap and patience changed vs the "
                  "paper-ready final_bnn_pretrained_ens12_pretuned; ASSUMPTION: cap 200,000 / patience 100 evals chosen). "
                  "Converged = no member at the cap.")


if __name__ == "__main__":
    main()
