"""Merge chain-wise diagnostic shards into one file (2026-11-05). Analysis only, no log entry.

Concatenates results/rbf_chainwise_diagnostics.npz (repeats 0-4, 280 rows, left untouched) with every shard in the
given directory (written by scripts/analysis/chainwise_diagnostics.py with CHAINWISE_OUT, e.g. by
slurm/section6_analysis.sbatch for the pretuned repeats 5-19 at n=50/100). A (model, n, repeat) present in both is an
error. Output: results/rbf_chainwise_diagnostics_r0_19.npz.
Usage: python scripts/analysis/merge_chainwise_shards.py SHARD_DIR
"""
import sys
from pathlib import Path

import numpy as np

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # this repository's src/
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401
BASE = RESULTS / "rbf_chainwise_diagnostics.npz"
OUT = RESULTS / "rbf_chainwise_diagnostics_r0_19.npz"


def main():
    parts = [BASE] + sorted(Path(sys.argv[1]).glob("*.npz"))
    data = [dict(np.load(p, allow_pickle=True)) for p in parts]
    keys = sorted(data[0])
    for p, d in zip(parts, data):
        assert sorted(d) == keys, f"{p.name}: keys differ"
    merged = {k: np.concatenate([np.atleast_1d(d[k]) for d in data]) for k in keys}
    ids = list(zip(merged["model"], merged["n"].tolist(), merged["repeat"].tolist()))
    assert len(ids) == len(set(ids)), "duplicate (model, n, repeat) rows"
    np.savez_compressed(OUT, **merged)
    print(f"{len(parts)} files, {len(ids)} rows -> {OUT}")


if __name__ == "__main__":
    main()
