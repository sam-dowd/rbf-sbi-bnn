# Bayesian neural networks versus neural posterior estimation with little data

Companion code for the report *Bayesian neural networks versus neural posterior estimation for simulation-based
inference with little data* (Sam Dowd, Seljak Group, UC Berkeley), in
[`paper/paper.pdf`](paper/paper.pdf).

The task is a toy version of simulation-based inference on cosmological fields: infer the amplitude σ and length
scale ρ of an RBF kernel, which set a Gaussian field's power spectrum, from the empirical covariance of ten noisy
draws, a task whose exact posterior is computed on a grid. Bayesian neural networks sampled with Microcanonical
Langevin Ensembles (MILE, via [`sklearn-contrib-bde`](https://github.com/scikit-learn-contrib/bde)) are compared with
neural posterior estimation (a masked autoregressive flow) and with the BNN's own networks trained without sampling,
on 50 to 10,000 simulations.

## Layout

```
src/rbf_sbi/        the library
  data.py             simulator, canonical dataset, exact grid posterior
  refpost_cache.py    cache of the exact test-set posteriors
  metrics.py          test NLL, TARP, shared evaluation
  bnn.py              MILE BNN with a correlated bivariate Gaussian head; re-tuning and empirical-Bayes samplers;
                      the unsampled networks ("Gaussian predictor")
  npe.py              conditional MAF neural posterior estimator
  evaluation.py       exact-reference moments, per-fit metrics, paired-test bolding, repeat splits
  diagnostics.py      chain-wise convergence diagnostics
  prior_predictive.py prior predictive of the untrained network (data-free prior scale)
  experiment_log.py   optional run ledger (set RBF_SBI_EXPERIMENT_LOG)
  paths.py            repository paths (overridable, see below)
scripts/data/       dataset, reference posterior, grid check
scripts/tuning/     hyperparameter searches: pretraining grid (BNN), NPE, temperatures, data-free prior scale
scripts/fit/        the fits: run_experiment.py (BNNs, capped Gaussian predictor, NPE ensemble), converged Gaussian
                    predictor, long-trained NPE, NPE variants
scripts/analysis/   post-processing of the per-draw outputs: NLL decomposition, chain-wise diagnostics
paper/              report source, tables, figures; paper/scripts/ regenerate them; make_paper.sh runs everything
legacy/             Appendix G (earlier 3-member runs) and superseded searches; see legacy/README.md
slurm/              batch templates for NERSC Perlmutter
results/            small result files shipped with the code (selected hyperparameters, NLL decomposition,
                    convergence diagnostics); the full result files go in results/raw/final/
data/dataset.npz    the canonical dataset (10,000 training pairs, 1,000 test pairs; seed 0)
```

## Installation

```
conda create -n rbf-sbi python=3.11 pip
conda activate rbf-sbi
pip install -r requirements.txt
pip install -e .          # optional: scripts add src/ to the path themselves
```

The versions are those used for the paper. `src/rbf_sbi/bnn.py` subclasses undocumented internals of
`sklearn-contrib-bde` 1.0.2 (Appendix D of the paper), so keep that version.

## Reproducing the paper

**Tables and figures from the released results.** Download the result files (Zenodo DOI to be added) into
`results/raw/final/`, build the reference posterior once, then regenerate everything:

```
python scripts/data/build_reference.py      # data/reference_posterior/, about 110 MB; JAX, a GPU makes it fast
bash paper/make_paper.sh
```

**The fits.** The selected hyperparameters are in `results/`, so the searches are optional
(`slurm/pretrain_grid.sbatch`, `slurm/optimize_npe.sbatch`, `slurm/optimize_temperature_grid.sbatch`,
`slurm/optimize_prior_width.sbatch`). On Perlmutter:

```
export SBATCH_ACCOUNT=<project>_g
bash slurm/submit_final_sweep.sh list        # then: submit
# converged Gaussian predictor, long-trained NPE, NPE variants: items are n:repeat (or n:repeat:member, v:n:repeat:members)
sbatch slurm/gp_long_packed.sbatch 50:0 50:1 ...
sbatch slurm/npe_long_packed.sbatch 10000:0:0 ...     then  sbatch slurm/merge_cpu.sbatch scripts/fit/npe_long_merge.py 10000 0 1 2 3 4
sbatch slurm/npe_variant_packed.sbatch ctx16:50:0:0-11 ...   then  sbatch slurm/merge_cpu.sbatch scripts/fit/npe_variant_merge.py ...
# post-processing of the per-draw outputs (written to $SCRATCH/sbi_final_draws/)
sbatch slurm/lppd_decomposition.sbatch
sbatch slurm/chainwise_shards.sbatch final_bnn_stdnormal_pretuned final_bnn_gaussprior_pretuned ...
python scripts/analysis/merge_chainwise_shards.py
```

Each fitting script's docstring gives its exact arguments. Wall time per fit on one A100: 36-72 s for a 12-chain
BNN under stock MILE, 410-530 s with the re-tuning sampler, 160-240 s (n <= 1,000) and 660-910 s (n >= 5,000) for
the 12-flow NPE ensemble (Appendix C). GPU runs are not bit-reproducible; `slurm/verify_reproduction.sh` checks a
re-fit against the released result.

## Environment variables

| variable | default | purpose |
|---|---|---|
| `RBF_SBI_DATA` | `data/` | dataset and reference posterior |
| `RBF_SBI_RESULTS` | `results/` | result files |
| `RBF_SBI_SCRATCH` | `$PSCRATCH`, else `scratch/` | per-draw outputs and intermediate member files |
| `RBF_SBI_EXPERIMENT_LOG` | unset (off) | Markdown file to which fitting scripts append a run record |
| `SBATCH_ACCOUNT` | -- | Slurm account for the templates in `slurm/` |
| `RBF_SBI_CONDA_ENV` | `rbf-sbi` | conda environment activated by the templates |

## Notes

Code, analysis scripts and text were developed with the assistance of Claude models (Anthropic), most recently
Claude Opus 5.5, used through Claude Code; all results were checked by the author.

License: to be added. Two functions in `src/rbf_sbi/bnn.py` are derived from `sklearn-contrib-bde` and remain
under its BSD 3-Clause License; see [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
