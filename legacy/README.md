# legacy/

Code behind Appendix G of the paper (prior and sampler variants from the **earlier 3-member runs**: 5 repeats,
earlier BNN hyperparameters, NPE as a single flow) and the superseded searches it used:

- `optimize_bde.py` -- the earlier BNN learning-rate / weight-decay search on full MILE fits (superseded by
  `scripts/tuning/optimize_pretrain_grid.py`; Appendix C).
- `optimize_prior.py`, `prior_coverage_check.py`, `ssge.py` -- the optimized per-weight Gaussian prior
  (Appendix G.4, after Delaunoy et al. 2024).
- `results4_{scale,shape,side}.py`, `old_runs_common.py` -- Tables 18-20.

These need the earlier runs' result files (`results/raw/sweep_*.npz`), which are not released; the tables they
produced are committed in `paper/tables/results4_*.tex`. The sweep keys of the earlier runs are documented in
`scripts/fit/run_experiment.py`'s module docstring.
