"""Where the sampled BNN's test NLL comes from: single draw -> one chain -> 12 chains, against its own starting
networks (2026-11-05). Analysis only: reads the final sweep's per-draw files and saved results; fits nothing; no
experiment-log entry.

Per (model, n, repeat), from $PSCRATCH/sbi_final_draws/{model}_n{n}_r{r}_draws.npz (mu, s, corr: 12 members x 1,000
kept draws x 1,000 test datasets (x 2), original units), with lp = log N2(theta* | mu, s, corr) in float64
(chainwise_diagnostics_rbf.logpdf2, the same density as the saved files):
  nll_draw     median over all 12 x 1,000 draws of the draw's own test NLL (-mean over test datasets of lp): a
               typical single posterior draw used alone. The MEAN over draws is not usable: at n=50 a few draws
               predict near-zero widths and dominate it (smoke test n=50 r0: mean 3e12, median finite), so it is
               kept only as nll_draw_mean, to document that tail.
  nll_draw_m0  the same median for member 0's 1,000 draws only            (pairs with the start network, member 0)
  nll_chain    -mean over members of the test NLL of that member's 1,000-draw mixture   (one chain)
  nll_bnn      test NLL of the 12 x 1,000-draw mixture (the BNN as reported; asserted to reproduce the saved file)
and, from results/raw/final, the BNN's own starting networks (the capped Gaussian predictor, MILE's pretraining):
  nll_start1   member 0 alone (final_bnn_pretrained_pretuned)        nll_start12  the 12-network mixture
              (final_bnn_pretrained_ens12_pretuned)
These starting networks are the N(0,1) run's own pretrained networks; the data-free and tempered runs pretrain
identically (same seed, data, lr/wd; neither the prior scale nor T enters pretraining), up to GPU floating-point
differences (ASSUMPTION: not re-verified per file here).
Differences reported downstream: nll_draw_m0 - nll_start1 (a typical draw vs its start), nll_chain - nll_draw
(averaging within a chain: sampling), nll_bnn - nll_chain (mixing 12 chains: ensembling), nll_start12 - nll_start1 (ensembling
the unsampled starting networks).

Usage: python scripts/analysis/lppd_decomposition.py MODEL N [N ...]      (one process per model/size group; repeats 0-19 at
       n <= 100, 0-4 above; missing draw files are reported and skipped)
       env LPPD_OUT_DIR (default $PSCRATCH/lppd_decomposition) -> one {MODEL}_n{N}.npz per size; merge with --merge.
       python scripts/analysis/lppd_decomposition.py --merge   -> results/lppd_decomposition.npz
       python scripts/analysis/lppd_decomposition.py --smoke OUT.npz   (n=50 r0 of the N(0,1) model only, scratch output)
"""
import os
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # this repository's src/
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401
from rbf_sbi.diagnostics import logpdf2
from rbf_sbi.evaluation import REF   # noqa: E402

DRAWS = SCRATCH / "sbi_final_draws"
FINAL = RESULTS / "raw" / "final"
OUT_DIR = Path(os.environ.get("LPPD_OUT_DIR", str(SCRATCH / "lppd_decomposition")))
MERGED = RESULTS / "lppd_decomposition.npz"
KEYS = ["nll_draw", "nll_draw_mean", "nll_draw_m0", "nll_chain", "nll_bnn", "nll_saved", "nll_start1", "nll_start12"]


def saved_nll(stem, n, r):
    with np.load(FINAL / f"{stem}_n{n}_r{r}.npz", allow_pickle=True) as d:
        return -float(np.mean(d["log_prob_true"]))


def one(model, n, r, th):
    f = DRAWS / f"{model}_n{n}_r{r}_draws.npz"
    with np.load(f) as d:
        mu, s, c = (d[k].astype(np.float64) for k in ("mu", "s", "corr"))
    lp = logpdf2(th, mu, s, c)                     # (members, draws, test)
    M, S, _ = lp.shape
    per_draw = -lp.mean(axis=2)                    # (members, draws): each draw's own test NLL
    row = dict(nll_draw=float(np.median(per_draw)), nll_draw_mean=float(per_draw.mean()),
               nll_draw_m0=float(np.median(per_draw[0])),
               nll_chain=-float(np.mean(logsumexp(lp, axis=1) - np.log(S))),
               nll_bnn=-float(np.mean(logsumexp(lp.reshape(M * S, -1), axis=0) - np.log(M * S))),
               nll_saved=saved_nll(model, n, r),
               nll_start1=saved_nll("final_bnn_pretrained_pretuned", n, r),
               nll_start12=saved_nll("final_bnn_pretrained_ens12_pretuned", n, r))
    assert abs(row["nll_bnn"] - row["nll_saved"]) < 1e-4, f"{f.name}: recomputed {row['nll_bnn']} vs saved {row['nll_saved']}"
    return row


def run(model, sizes, out_dir):
    th = np.load(REF)["theta"].astype(np.float64)
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in sizes:
        rows, reps = [], []
        for r in (range(20) if n <= 100 else range(5)):
            if not (DRAWS / f"{model}_n{n}_r{r}_draws.npz").exists():
                print(f"missing {model} n={n} r={r}")
                continue
            row = one(model, n, r, th)
            rows.append(row); reps.append(r)
            print(f"{model} n={n} r={r}: " + " ".join(f"{k} {row[k]:+.4f}" for k in KEYS), flush=True)
        np.savez(out_dir / f"{model}_n{n}.npz", model=model, n=n, repeat=np.array(reps),
                 **{k: np.array([x[k] for x in rows]) for k in KEYS})


def merge():
    parts = sorted(OUT_DIR.glob("*_n*.npz"))
    cols = {k: [] for k in ["model", "n", "repeat"] + KEYS}
    for p in parts:
        with np.load(p) as d:
            R = len(d["repeat"])
            cols["model"] += [str(d["model"])] * R; cols["n"] += [int(d["n"])] * R; cols["repeat"] += list(d["repeat"])
            for k in KEYS:
                cols[k] += list(d[k])
    np.savez(MERGED, **{k: np.array(v) for k, v in cols.items()})
    print(f"merged {len(parts)} parts, {len(cols['n'])} rows -> {MERGED}")


if __name__ == "__main__":
    if sys.argv[1] == "--merge":
        merge()
    elif sys.argv[1] == "--smoke":
        th = np.load(REF)["theta"].astype(np.float64)
        row = one("final_bnn_stdnormal_pretuned", 50, 0, th)
        np.savez(sys.argv[2], **row)
        print(" ".join(f"{k} {v:+.4f}" for k, v in row.items()))
    else:
        run(sys.argv[1], [int(x) for x in sys.argv[2:]], OUT_DIR)
