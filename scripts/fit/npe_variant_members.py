"""NPE architecture variants for the head x body comparison (2026-11-04): members of 12-network ensembles that differ
from `final_npe_ens12` in ONE architectural choice each, everything else identical (training subsample, split, seed,
member seeds, lr / wd -- read from and asserted against results/raw/final/final_npe_ens12_n{n}_r{r}.npz --, ReLU
encoder, optimiser, eval interval, TARP / posterior-sample settings).

Variants (VARIANTS below):
  flow16   encoder output (context) width 8 -> 16, i.e. encoder 100->16->16->16 = the BNN's width; MAF flow head
  gauss16  same encoder; bivariate Gaussian head (npe.ConditionalBivariateGaussian, 16 hidden) instead of the flow
  base8    = final_npe_ens12 exactly (context 8, flow) -- only for the --compare reproduction check
Budget (--budget auto): the budget of each size's NPE comparator -- default (steps 20,000, patience 20 evals) at
n <= 1,000 (vs final_npe_ens12; flows stop early there), long (200,000, patience 100) at n >= 5,000 (vs
final_npe_ens12_long; the default budget under-trains there, found 2026-11-04).

ASSUMPTIONS (flagged): NPE's lr / wd are reused, not re-tuned for the new width / head; the encoder keeps ReLU and
NPE's input handling (the BNN uses soft-sign and standardised inputs) so that each variant changes one thing vs NPE;
the encoder's last layer is linear (ContextEncoder), so 'BNN-matched' means matched widths, not identical layers.

One process runs the listed members of one (variant, n, repeat) in turn; each member file is written as soon as it is
done and existing ones are skipped. Merge with scripts/fit/npe_variant_merge.py (which writes the log entries).
Usage: python scripts/fit/npe_variant_members.py VARIANT N R K [K ...] [--budget auto|default|long] [--out-dir] [--compare]
  K may be a range "0-11".
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
from rbf_sbi import npe       # noqa: E402  torch FIRST, before anything that pulls in jax (CUBLAS_STATUS_NOT_INITIALIZED otherwise;
#                                first seen 2026-09-15, seen again 2026-11-04)
from rbf_sbi import data      # noqa: E402
from rbf_sbi import metrics   # noqa: E402

FINAL = RESULTS / "raw" / "final"
EVAL_EVERY, TARP_N_DRAWS, N_POSTERIOR, TARP_SEEDS = 100, 1000, 5000, list(range(20))   # as run_experiment.py
VARIANTS = {"flow16": dict(head="flow", context_dim=16), "gauss16": dict(head="gaussian", context_dim=16),
            "base8": dict(head="flow", context_dim=8)}
BUDGETS = {"default": (20_000, 20), "long": (200_000, 100)}
DEFAULT_OUT = SCRATCH / "sbi_npe_variants" / "members"


def budget_for(n, budget):
    return BUDGETS[("long" if n >= 5000 else "default") if budget == "auto" else budget]


def member_path(out_dir, variant, n, r, k, steps, patience):
    return Path(out_dir) / f"npe_{variant}_s{steps}_p{patience}_n{n}_r{r}_k{k}.npz"


def parse_ks(tokens):
    ks = []
    for t in tokens:
        if "-" in t:
            a, b = t.split("-"); ks += list(range(int(a), int(b) + 1))
        else:
            ks.append(int(t))
    return ks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=sorted(VARIANTS))
    ap.add_argument("n", type=int)
    ap.add_argument("r", type=int)
    ap.add_argument("ks", nargs="+")
    ap.add_argument("--budget", choices=["auto", "default", "long"], default="auto")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    steps, patience = budget_for(a.n, a.budget)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    ref = np.load(FINAL / f"final_npe_ens12_n{a.n}_r{a.r}.npz", allow_pickle=True)
    seed, lr, wd = int(ref["seed"]), float(ref["lr"]), float(ref["wd"])
    X_train, theta_train, X_test, theta_test = data.load_dataset()
    Xb, yb = data.subsample_train(X_train, theta_train, a.n, a.r)
    assert np.array_equal(np.asarray(Xb).reshape(len(Xb), -1), ref["X_train"]), "subsample differs from final_npe_ens12"
    Xt, yt, Xv, yv = metrics.split(np.asarray(Xb), np.asarray(yb), seed)

    for k in parse_ks(a.ks):
        out = member_path(a.out_dir, a.variant, a.n, a.r, k, steps, patience)
        if out.exists():
            print(f"skip {out.name} (exists)", flush=True)
            continue
        mseed = seed if k == 0 else seed + 10_000 * k
        assert mseed == int(ref["member_seeds"][k]), f"member seed {mseed} != saved {int(ref['member_seeds'][k])}"
        info, t0 = {}, time.time()
        raw, f = npe.fit(Xt, yt, Xv, yv, X_test, theta_test, mseed, lr, wd, steps, EVAL_EVERY, patience,
                         TARP_N_DRAWS, N_POSTERIOR, TARP_SEEDS, data.TARP_LO, data.TARP_HI, info_out=info,
                         **VARIANTS[a.variant])
        sec = time.time() - t0
        try:
            import torch
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        except Exception:
            gpu = "unknown"
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, variant=a.variant, n_train=a.n, repeat=a.r, seed=seed, member=k, member_seed=mseed,
                            lr=lr, wd=wd, hp_file=str(ref["hp_file"]), steps=steps, patience=patience,
                            eval_every=EVAL_EVERY, head=VARIANTS[a.variant]["head"],
                            context_dim=VARIANTS[a.variant]["context_dim"], fit_sec=sec, gpu_name=gpu, f=f,
                            steps_run=info["steps_run"], best_step=info["best_step"],
                            stopped_early=info["stopped_early"], val_history=info["val_history"],
                            **{key: np.asarray(v) for key, v in raw.items()})
        os.replace(tmp, out)
        print(f"done {a.variant} n={a.n} r={a.r} k={k}: {sec:.0f} s, steps_run {info['steps_run']}, best_step "
              f"{info['best_step']}, NLL {-float(np.mean(raw['log_prob_true'])):+.4f} "
              f"(NPE member {-float(np.mean(ref['member_log_prob_true'][k])):+.4f}) [{gpu}]", flush=True)
        if a.compare:
            same = np.array_equal(raw["log_prob_true"], ref["member_log_prob_true"][k])
            d = float(np.max(np.abs(raw["log_prob_true"] - ref["member_log_prob_true"][k])))
            print(f"COMPARE {a.variant} n={a.n} r={a.r} k={k} vs saved final_npe_ens12 member: "
                  f"{'IDENTICAL' if same else f'DIFFERENT (max|d| {d:.3g})'}", flush=True)


if __name__ == "__main__":
    main()
