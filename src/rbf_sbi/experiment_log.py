"""Append-only, structured experiment ledger. Off unless RBF_SBI_EXPERIMENT_LOG names a Markdown file. Every script that produces
a real result (not a toy/smoke test) should call log_run() once at the end of main(), recording its
own config and key result metrics with a timestamp.

Complements, but does not replace:
- Slurm .out/.err files under logs/ -- raw stdout/stderr for one specific job.
- a separate narrative record, where a run's *interpretation* (what it means for open questions)
  belongs. This file is just the raw record: what ran, with what
  config, and what numbers came out.
"""
import datetime
import os
from pathlib import Path

LOG_PATH = Path(os.environ["RBF_SBI_EXPERIMENT_LOG"]) if os.environ.get("RBF_SBI_EXPERIMENT_LOG") else None


def log_run(script, config, results, job_id=None, output_path=None, notes=None, dataset="rbf_kernel"):
    """Append one structured entry to the experiment log (RBF_SBI_EXPERIMENT_LOG).

    Parameters
    ----------
    script : str -- e.g. "scripts/bde_correlated_perchain_test.py"
    config : dict -- the run's hyperparameters (n_train, n_members, warmup_steps, ... whatever the
        calling script considers its own config)
    results : dict -- key result metrics (ess_mean, rhat_max, mae_sigma, ... whatever's relevant)
    job_id : str or None -- Slurm job ID, if run via sbatch (e.g. os.environ.get("SLURM_JOB_ID"))
    output_path : str or None -- path to the .npz this run produced
    notes : str or None -- free-text context (e.g. "rerun after pretraining-schedule fix")
    dataset : str -- which dataset the run used: "rbf_kernel" (the default, and every entry written
        before 2026-10-04, which carry no Dataset line) or "gp_surface".
    """
    if LOG_PATH is None:   # logging disabled
        return
    ts = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    header = f"## {ts} — {script}"
    if job_id:
        header += f" (job {job_id})"
    lines = [header, f"\n**Dataset:** {dataset}"]
    if notes:
        lines.append(f"\n{notes}")
    lines.append("\n**Config:**")
    for k, v in config.items():
        lines.append(f"- {k} = {v}")
    lines.append("\n**Results:**")
    for k, v in results.items():
        lines.append(f"- {k} = {v}")
    if output_path:
        lines.append(f"\nOutput: `{output_path}`")
    lines.append("\n---\n")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write("\n".join(lines) + "\n")
