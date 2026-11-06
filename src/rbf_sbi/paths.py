"""Repository paths. Data, results and scratch can be redirected with environment variables:
RBF_SBI_DATA, RBF_SBI_RESULTS, RBF_SBI_SCRATCH (default for scratch: $PSCRATCH on NERSC, else ./scratch)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("RBF_SBI_DATA", ROOT / "data"))
RESULTS = Path(os.environ.get("RBF_SBI_RESULTS", ROOT / "results"))
SCRATCH = Path(os.environ.get("RBF_SBI_SCRATCH") or os.environ.get("PSCRATCH") or ROOT / "scratch")
