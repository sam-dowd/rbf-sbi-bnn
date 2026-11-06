"""Companion-repository note: the standalone ParamNet Gaussian baseline (models "gaussian" and
"final_gaussian_ens12", src/gaussian_predictor.py) has been removed; it is not part of the paper. The paper's
Gaussian predictor is the BNN's own networks without sampling (final_bnn_pretrained*).

Entry point for the full model-comparison sweep: fits one posterior-inference method across
training-set sizes and repeats, saving each (size, repeat)'s raw per-test-point predictions/
diagnostics as a compressed .npz under results/raw/ -- enough to build TARP curves, SBC rank
histograms, and NLL/MAE summary tables later without rerunning anything.

--- Rewritten 2026-09-13: the model set and the sweep's own
loop structure both changed ---

**New model set** (replaces the previous vanilla/opt/opt_temp-BNN + NPE + Gaussian five-way
comparison -- the "opt"/"opt_temp" variants, built around the Gaussian-width/Delaunoy OPTIMIZED
priors, are retired from this sweep in favor of a newer, more promising finding):
  1. "gaussian" -- the standalone heteroskedastic Gaussian baseline (src/gaussian_predictor.py),
     unchanged.
  2. "npe" -- the conditional normalizing-flow NPE (src/npe.py), unchanged.
  3. "bnn_vanilla" -- bde's own default prior (prior_family="standardnormal", flat N(0,1) over
     every weight), untempered (temperature=1.0). REVISED 2026-09-13 (was: UNTUNED lr/weight_decay,
     the "no optimization at all" baseline) -- per direct request, ALL THREE BNN variants
     (bnn_vanilla/bnn_temp/gibbs) now share the SAME per-size TUNED (lr, weight_decay) (see
     _load_bnn_hp), so pretraining quality is equalized across them and the only thing that
     differs between variants is the prior/temperature choice itself, not a confound from
     under-tuned pretraining. "bnn_vanilla" is now "N(0,1) prior, tuned pretraining, T=1" -- the
     floor for how well a fixed, unoptimized PRIOR does, not a floor on the whole pipeline.
  4. "bnn_temp" -- SAME N(0,1) prior and TUNED (lr, weight_decay) as "bnn_vanilla", plus
     temperature=BNN_TEMP_TEMPERATURE. ASSUMPTION, flagged directly: T=0.305
     (BNN_TEMP_TEMPERATURE's exact value, results/temperature_search_n100_optprior.npz's own
     best_temperature) was found by scripts/optimize_temperature.py's BO search run under the
     Delaunoy-OPTIMIZED prior at n=100, tuned lr/wd -- NOT under the plain N(0,1) prior this
     variant actually uses. Reusing it here is an untested transfer assumption (same kind the
     retired "opt_temp" variant already made, transferring the SAME number to the Gaussian-width
     prior instead) -- re-validate with a real search over temperature specifically for the plain
     N(0,1) prior if this comparison motivates it.
  5. "gibbs" -- the block Gibbs sampler (bnn.fit_gibbs_hierarchical, see its own docstring and
     scripts/bde_gibbs_test.py's module docstring for the full algorithm/citation history):
     CENTERED per-weight N(0, tau_i^2) prior, tau_i^2 ~ InverseGamma(2,1) conjugate hyperprior,
     alternating ordinary MCLMC weight-sampling at fixed tau with an exact closed-form tau update.
     Headline result motivating this whole rewrite: at n=100 this was the best untempered
     run across an 18-way leaderboard (best test NLL, 3rd-best mixing); at n=10,000 it reproduces
     the project's persistent large-n mixing failure exactly like every other prior tried, ruling
     out the weight/tau MCMC coupling as the (sole) cause. GIBBS_N_BURNIN_SWEEPS/
     GIBBS_BURNIN_WARMUP_STEPS/GIBBS_BURNIN_N_SAMPLES below match scripts/bde_gibbs_test.py's own
     defaults exactly -- kept in sync manually (both read from the same underlying
     bnn.fit_gibbs_hierarchical, but the CLI wrapper and this sweep don't share a single constants
     module, a real but low-priority duplication).
  6. "rasmussen_gibbs_retune" -- added 2026-09-17. Rasmussen (1995)/Neal (1996)-style single
     continuously-evolving-chain Gibbs sampler (bnn.fit_rasmussen_gibbs_hierarchical, see its own
     docstring and scripts/bde_rasmussen_gibbs_retune_test.py's module docstring): SAME
     IG(a=2,b=1) hyperprior/tau_init as "gibbs" above (shared GIBBS_IG_A/IG_B/TAU_INIT, for a fair
     comparison -- the two variants differ only in HOW tau and the weights are sampled, not in the
     hyperprior itself), but tau is updated via a closed-form conjugate draw at EVERY raw MCLMC
     step within one evolving chain (not "gibbs"'s restart-per-burn-in-sweep approximation), PLUS
     a periodic warm-started re-tune of (L, step_size) every RASMUSSEN_RETUNE_EVERY_SAMPLES kept
     samples (added specifically because tau's continuous drift changes the target's curvature,
     while a one-time initial warmup's (L, step_size) would otherwise stay frozen for the whole
     run). Real-compute validated at n=100 (job 58468952: rhat_max 1.097->1.006, ess_min
     10.4->830.9 vs. the same run without retuning, at a small cost in test_nll) and n=10,000 (job
     58469674, rhat_max=2.992 -- competitive with the best n=10,000 result on record across every
     prior family tried, though still failing the 1.01 mixing gate like everything else at that
     size). Two real-compute
     failures were needed to get a stable warm-started retune (naive warm-starting only carries the
     OUTPUT (L, step_size), not the internal EMA/position-variance accumulators MCLMC's own
     adaptation needs to stay stable across retunes). ASSUMPTION, not yet tested: the retune
     cadence/budget (100 kept samples, tune1/2=200/50 steps) validated at n=100/10,000 is NOT
     re-validated at this sweep's other four sizes (50/500/1000/5000) before this sweep runs.
  7. "bnn_studentt" -- added 2026-10-01. IDENTICAL code path to "bnn_vanilla" (plain
     bnn.fit_correlated, same tuned (lr, weight_decay), same seeds/subsamples, T=1.0, no Gibbs
     machinery, no (L, step_size) retuning) except the prior: a FIXED Student-t(nu=2a,
     scale=sqrt(b/a)) on every weight, with (a, b) = (GIBBS_IG_A, GIBBS_IG_B) -- i.e. nu=4,
     scale~0.7071 for IG(2,1). This is the exact marginal of "rasmussen_gibbs_retune"'s own prior
     (weight_i ~ N(0, tau_i^2), tau_i^2 ~ InverseGamma(a, b), tau integrated out -- a standard
     scale-mixture identity, verified numerically against scipy to 8 decimal places
     at w in {0, 0.3, 1, 2.5, 6}), so the two models target the SAME posterior over the weights.
     Purpose: isolate whether "rasmussen_gibbs_retune"'s low-data advantage over "bnn_vanilla"
     comes from the prior's SHAPE (then this model should match it) or from the Gibbs sampler/
     retuning itself (then it won't). No earlier Student-t run used these parameters (all were
     nu=3, scale 1 or 0.221), so this is the first like-for-like test. ASSUMPTION, flagged
     directly: equal targets do NOT imply equal sampler behavior -- this model has no periodic
     retune, so a mixing difference vs. "rasmussen_gibbs_retune" is expected and is not by itself
     evidence the priors differ.
  8. "rasmussen_gibbs_noretune" -- added 2026-10-01. "rasmussen_gibbs_retune" with the periodic
     retune switched OFF (retune_every_samples=None), everything else identical (same hyperprior,
     seeds, subsamples, tuned lr/wd, BNN_N_THINNING). 100,000 chain steps, 10,000 kept. The
     equal-step baseline for model 9 below. CORRECTION 2026-10-01: an earlier version of this
     entry called this "the same count as bnn_vanilla" -- wrong. bnn_vanilla runs 10,000 chain
     steps (1,000 kept); see STEP-COUNT CORRECTION below.
  9. "rasmussen_gibbs_noretune_extrasteps" -- added 2026-10-01. Model 8 with n_thinning raised
     from BNN_N_THINNING (10) to RASMUSSEN_EXTRASTEPS_N_THINNING (13): 130,000 chain steps, still
     10,000 kept samples. Purpose: "rasmussen_gibbs_retune" runs 250 unsaved MCLMC steps per
     retune (tune1+tune2 = 200+50) every 100 kept samples -- 25,000 extra chain steps on top of
     100,000, i.e. 12.5 per kept sample -- so its gains over a no-retune run confound "(L,
     step_size) corrected" with "25% more chain steps". This model gives the extra steps WITHOUT
     any retuning. ASSUMPTIONS, flagged directly: (a) thinning must be an integer, so 13 (not
     12.5) -- a 4% overshoot vs. the retuned run, chosen deliberately on the high side so that
     falling short of the retuned run rules the extra steps out; (b) the retune's extra steps run
     at a FIXED tau snapshot while these redraw tau every step, so the match is in step count, not
     in kind; (c) a different thinning changes the per-step RNG key layout, so this is a different
     chain realization from model 8 on the same data, not the same chain subsampled differently.
  10. "bnn_vanilla_extrasteps" -- REDEFINED 2026-10-01 (the first definition, n_thinning=13,
     never produced a file: its only job, 59160980, was cancelled while pending). "bnn_vanilla"
     (N(0,1) prior, plain bnn.fit_correlated, same tuned lr/wd, seeds, subsamples, T=1) run for
     BNN_EXTRASTEPS_N_STEPS = 125,000 chain steps with exactly BNN_EXTRASTEPS_N_KEPT = 10,000
     kept samples -- the step and kept-sample budget "rasmussen_gibbs_retune" actually has
     (100,000 + 100 retunes x 250), vs. "bnn_vanilla"'s 10,000 steps / 1,000 kept. Uses
     bnn.fit_correlated's n_kept_samples (12.5 steps per kept sample is not an integer thinning:
     every 12th or 13th step is kept alternately). The first definition was not a step-count
     control at all -- under bde's convention n_thinning=13 is still 10,000 steps, ~770 kept.
  11. "rasmussen_gibbs_noretune_10k" -- added 2026-10-01. Model 8 at "bnn_vanilla"'s real step
     budget: n_samples=RASMUSSEN_10K_N_KEPT (1,000 kept) x BNN_N_THINNING (10) = 10,000 chain
     steps, no retune.
  12. "rasmussen_gibbs_retune_10k" -- added 2026-10-01. Model 6 at the same budget: 10,000 chain
     steps / 1,000 kept, retuned every RASMUSSEN_RETUNE_EVERY_SAMPLES (100) kept samples -> 10
     retunes x 250 unsaved steps = 12,500 steps total. ASSUMPTION, flagged directly: the retune
     cadence (per 100 kept samples, tune1/2=200/50) is unchanged from model 6, so the retune
     count scales down with chain length (10, not 100); at n=100 model 6's L correction (55 ->
     ~20) took the first 2-3 retunes, i.e. ~20-30% of this shorter chain runs before it lands.
  13. "bnn_wideprior" -- added 2026-10-01. "bnn_vanilla" (same code path, tuned lr/wd, seeds,
     subsamples, 10,000 steps / 1,000 kept, T=1) with the prior widened from N(0,1) to
     N(0, 1/T) for T = BNN_TEMP_TEMPERATURE, i.e. scale 1/sqrt(0.305) ~ 1.811. Tests whether
     "bnn_temp"'s gain is the prior being effectively weakened: likelihood^(1/T) * prior =
     (likelihood * prior^T)^(1/T), and N(0,1)^T is N(0, 1/T) up to a constant, so bnn_temp is a
     fully tempered version of exactly this model's posterior. See MODEL_PURPOSE.
  14. "bnn_temp_opt" -- added 2026-10-01. "bnn_temp" with the temperature taken PER SIZE from
     scripts/tuning/optimize_temperature_grid.py's search (results/temperature_search_n{n}_stdprior_grid
     .npz: plain N(0,1) prior, the sweep's own tuned lr/wd, grid 0.2..1.0 step 0.1) instead of
     the single T=0.305 that was fitted at n=100 under the optimized prior. Everything else is
     "bnn_vanilla"/"bnn_temp": same code path, seeds, subsamples, 10,000 steps / 1,000 kept.
     Refuses to run a size whose search file is missing or incomplete. NOTE: repeats 0-2 are the
     training draws the temperature was selected on (against separate validation rows, never
     the test set); repeats 3-4 were not used in the search.
  15. "bnn_temp_T<a>p<b>" (e.g. "bnn_temp_T0p15" for T=0.15) -- added 2026-10-01. "bnn_temp" at
     a temperature given directly in the model name, for one-off manual checks. Same code path,
     seeds, subsamples and chain budget as "bnn_vanilla"/"bnn_temp"; files are written as
     sweep_bnn_temp_T<a>p<b>_n{n}_r{repeat}.npz so different temperatures never collide.
  16. "bnn_studentt_extrasteps" -- added 2026-10-01. "bnn_studentt" (fixed Student-t(nu=4,
     scale~0.7071) prior, the Gibbs models' exact marginal) at "bnn_vanilla_extrasteps"'s budget:
     BNN_EXTRASTEPS_N_STEPS = 125,000 chain steps, exactly BNN_EXTRASTEPS_N_KEPT = 10,000 kept.
     Step- and kept-count-matched to "rasmussen_gibbs_retune", which targets the same weight
     posterior -- so a remaining gap between the two is the sampler (per-step tau redraw +
     retuning), not the prior and not chain length.
  17. "bnn_vanilla_retune_10k" -- added 2026-10-01. Plain N(0,1) with the periodic (L, step_size)
     retune and NO tau sampling: bnn.fit_rasmussen_gibbs_hierarchical with resample_tau=False
     (tau fixed at GIBBS_TAU_INIT = 1 for every weight), otherwise identical to
     "rasmussen_gibbs_retune_10k" -- 10,000 chain steps / 1,000 kept + 10 retunes x 250 unsaved
     steps, same seeds, subsamples and tuned lr/wd. See MODEL_PURPOSE.
  18. "bnn_gaussprior" -- added 2026-10-02. "bnn_vanilla" (same code path, tuned lr/wd, seeds,
     subsamples, 10,000 steps / 1,000 kept, T=1) with the prior narrowed from N(0,1) to
     N(0, GAUSSPRIOR_SCALE), one scale for every weight, read from
     results/gaussian_width_prior_phi.npz (0.36553; scripts/tuning/optimize_prior_width.py, 2026-09-03).
     That width was chosen WITHOUT any training data -- it minimizes |coverage AUC| of the
     untrained network's prior predictive -- so the same value is used at every n_train. Same
     mechanism as "bnn_wideprior" (prior_family="normal"), opposite direction. See MODEL_PURPOSE.
  19. "bnn_ebscale_10k" -- added 2026-10-02. EMPIRICAL-BAYES global prior scale: one scale tau
     shared by every weight, estimated from the training data by Monte Carlo EM during an unsaved
     adaptation phase (EB_ADAPT_SEGMENTS segments of 1,000 chain steps + a 250-step retune each,
     starting from tau=1), then FROZEN for production. Production is exactly
     "bnn_vanilla_retune_10k"'s chain (10,000 steps / 1,000 kept + 10 retunes x 250 unsaved steps,
     no tau sampling) at the estimated tau instead of tau=1 -- so that model, not "bnn_vanilla",
     is the like-for-like control (the retune alone is known to change fit at n=500/1,000).
     bnn.fit_rasmussen_gibbs_hierarchical(resample_tau=False, eb_global_scale=True); see its
     docstring for the algorithm and flagged assumptions (per-member tau, fixed adaptation budget
     with no convergence test). The estimated tau and its trajectory are saved per member as
     rasmussen_eb_tau_history. See MODEL_PURPOSE. NOT RUN: its jobs were cancelled before
     writing anything -- 150 x 1,250 = 187,500 unsaved steps per member was rejected as far too
     expensive. Superseded by models 20/21.
  20. "bnn_ebstep_10k" -- added 2026-10-02. PER-STEP empirical-Bayes global prior scale, the
     design adopted in place of model 19: one scale shared by every weight, (a) initialised
     per member to the RMS of that member's Adam-pretrained weights, which is the prior the
     warmup runs under, and (b) reset to the RMS of the CURRENT weights at every chain step.
     No adaptation phase, no retune: exactly 10,000 chain steps / 1,000 kept, "bnn_vanilla"'s
     budget. bnn.fit_rasmussen_gibbs_hierarchical(resample_tau=False, eb_per_step=True,
     tau_init_from_pretrained=True); see its docstring for the flagged assumptions (the scale is
     never frozen; stability of the weight norm is not guaranteed). See MODEL_PURPOSE.
  21. "bnn_ebstep_retune_10k" -- added 2026-10-02. Model 20 plus the periodic (L, step_size)
     retune (10 x 250 unsaved steps -> 12,500 steps / 1,000 kept, "rasmussen_gibbs_retune_10k"'s
     budget). Run alongside model 20 because the request did not say whether to keep the retune,
     and the retune alone is known to change fit at n=500/1,000: 20 vs. bnn_vanilla and 21 vs.
     bnn_vanilla_retune_10k are the two like-for-like pairs.
  22. "bnn_ebblock_iso_10k" -- added 2026-10-04. BLOCK empirical-Bayes prior scale, one scale
     for all weights. Initial scale = RMS of the member's Adam-pretrained weights; the one-time
     warmup runs under it. Then, during sampling only, after each 1,000-step block (the retune
     segment) the scale is updated from the block's per-weight mean m and variance v --
     gamma = d - sum(v)/tau^2, tau^2 = |m|^2 / gamma (MacKay's evidence update with sampled
     moments), damped (half-way in log space, at most x2 per block) -- and (L, step_size) are
     retuned. The prior is FIXED within a block, unlike models 20/21. Budget identical to
     bnn_vanilla_retune_10k (12,500 steps / 1,000 kept). bnn.fit_rasmussen_gibbs_hierarchical(
     resample_tau=False, eb_block="iso"); see its docstring for every flagged assumption.
  23. "bnn_ebblock_layer_10k" -- added 2026-10-04. Model 22 with one scale per layer (4 groups:
     1,616 / 272 / 272 / 85 weights; a layer's biases share its scale), each initialised to that
     layer's pretrained RMS and updated with its own gamma. eb_block="layer".
  24. "bnn_elu_h8" -- added 2026-11-02. bnn_vanilla on a small network: 100 -> 8 -> 8 -> 5 with ELU
     (925 weights; bnn_vanilla is 100 -> 16 -> 16 -> 16 -> 5 soft-sign, 2,245). Everything else as
     bnn_vanilla. Purpose: does the BNN's n=50 excess width (predictive about as wide as the theta
     prior) persist on a network small enough to sample easily? Run at n=1,000 (repeat 0: can it fit
     correctly at all?) and n=50 (repeats 0-2). ASSUMPTIONS: the per-size lr/wd tuned for 16-16-16
     soft-sign are reused, not re-tuned; the N(0,1) weight prior is kept, so the implied prior over
     outputs differs from bnn_vanilla's (width and activation change it) -- a result here mixes
     sampling difficulty with that prior change.
  25. "bnn_vanilla_kfold" -- added 2026-11-02. bnn_vanilla with per-member K-fold pretraining (K=5;
     bnn.fit_correlated(pretrain_kfold=5)): member m pretrains on the 4/5 of the training points outside fold m
     and early-stops on fold m (40 / 10 at n=50, the default 80/20 sizes), instead of all members sharing
     bde's single fixed split. Each member starts from the same initial weights as in bnn_vanilla; MCLMC uses
     all n points, as before. Purpose: does the shared 10-point validation set at n=50 hurt pretraining?
     Run at n=50, repeats 0-4. ASSUMPTION: the n=50 lr/wd, tuned under the default split, are reused.
  26/27. "npe_gausshead" / "npe_gausshead_wide" -- added 2026-11-02. "npe" with the conditional MAF flow replaced by a
     full bivariate Gaussian head on the same 8-dim encoder output (npe.fit(head="gaussian", gaussian_hidden=16 / 128);
     229 / 1,797 head parameters vs the flow's 1,896). Everything else is "npe" (split, seeds, lr/wd, optimiser, stopping,
     5,000 draws for mean/std). Purpose: does NPE's better sigma mean at n=50 come from the flow head? Run at n=50,
     repeats 0-4. ASSUMPTION: NPE's lr/wd (tuned for the flow) are reused for the Gaussian head.

HYPERPARAMETER VERSIONS (2026-10-01): results/bde_params_n{50,500,1000,5000}.npz date from
2026-08-28, i.e. from legacy/optimize_bde.py BEFORE its 2026-09-04 fix (candidates scored on an
inner 80/20 split of the training subsample -- 10 validation points at n=50 -- with one fixed
split reused for every trial). n=50's lr (0.0953) sits at the top of the search range and is
1,000x n=100's post-fix value (9.5e-5; n=100's own pre-fix value was 0.09996). Re-run searches are
written to results/bde_params_new_n{n}.npz (BDE_PARAMS_TAG=new) so the old files stay in place for
jobs already running. _load_bnn_hp PREFERS the "new" file for a size when it exists; results fit
with it are written as sweep_{model}_hpnew_n{n}_r{repeat}.npz (and read their searched temperature
from temperature_search_n{n}_stdprior_hpnew_grid.npz), so they never collide with, or get skipped
in favour of, results fit under the old hyperparameters. Setting the environment variable
BDE_HP_VERSION=old forces the original file (and untagged result names) regardless -- used to
finish runs that were started under the old values. Which file a size used is decided once,
when its (lr, weight_decay) are loaded, and saved in every result file as hp_file.

STEP-COUNT CORRECTION (2026-10-01): the two BNN code paths read (n_samples, n_thinning)
differently. bnn.fit_correlated (bde) runs n_samples RAW steps and keeps every n_thinning-th;
bnn.fit_rasmussen_gibbs_hierarchical keeps n_samples and runs n_samples * n_thinning steps. Both
are passed BNN_N_SAMPLES=10000, BNN_N_THINNING=10, so per chain, after the 50,000-step warmup:
  bnn_vanilla / bnn_temp / bnn_studentt       10,000 steps    1,000 kept
  gibbs (production fit, via fit_correlated)  10,000 steps    1,000 kept
  rasmussen_gibbs_noretune                   100,000 steps   10,000 kept
  rasmussen_gibbs_noretune_extrasteps        130,000 steps   10,000 kept
  rasmussen_gibbs_retune                     125,000 steps   10,000 kept
  bnn_vanilla_extrasteps                     125,000 steps   10,000 kept
  rasmussen_gibbs_noretune_10k                10,000 steps    1,000 kept
  rasmussen_gibbs_retune_10k                  12,500 steps    1,000 kept
Every Rasmussen-vs-fit_correlated comparison made before this note compared chains 10x apart in
length. Step-matched pairs: bnn_vanilla / bnn_studentt vs. the two _10k models; and
bnn_vanilla_extrasteps vs. rasmussen_gibbs_retune.

SEED CORRECTION (2026-10-01): seed was 100 * (position of n in SIZES) + repeat, and a
single-size invocation shrinks SIZES to one entry, so per-size jobs ran with seed = repeat while
full sweeps ran with e.g. 400 + repeat at n=100. Now indexed against ALL_SIZES, so both give the
full-sweep seed. Files written by per-size jobs BEFORE this fix (rasmussen_gibbs_noretune n=50
and n=100, rasmussen_gibbs_noretune_extrasteps n=50 -- jobs 59160976/59160978/59160977) carry
seed = repeat: same training subsample as every other model, but different pretraining/chain
seeds from the full-sweep models, so they are not seed-paired with them.

MODEL_PURPOSE (below) records, per model added 2026-10-01, what the run is FOR and what each
possible outcome would imply -- printed at job start and written into every experiment-log
entry, so the reasoning travels with the result.

**New loop structure**: MODEL is now REQUIRED as the first CLI argument (was: implicit, ran all
models together) -- "run one model at a time" rather than "one dataset size at
a time." Loops SIZES x N_REPEATS internally for whichever single model is given. Rationale:
different models have very different per-(size,repeat) cost profiles (Gaussian is
seconds, NPE is GPU-bound minutes, BNN/Gibbs are MCLMC-bound minutes-to-tens-of-minutes) --
sweeping one model across every size/repeat as its own job lets each be resubmitted/debugged
independently without the other four models' unrelated failures blocking it, and lets a bad-luck
failure in one model's fit not waste an otherwise-complete run of the other four at that
(size, repeat). Saves to model-specific files (see raw_path below) rather than one shared
per-(size,repeat) file -- avoids any read-modify-write/merge logic entirely, at the cost of needing
a join-by-(size,repeat) step for combined analysis (done in the analysis scripts).

Owns all hyperparameters -- the model files (src/gaussian_predictor.py, src/bnn.py, src/npe.py) take
everything as explicit arguments, nothing is read from module-level globals.

--- Repeats ---

N_REPEATS=5, unchanged from before this rewrite -- each repeat redraws an independent n_train
TRAINING subsample (data.subsample_train(..., repeat=k)), so N_REPEATS controls how many
independent "what if we'd trained on a different random n_train draw" samples go into a per-size
summary metric's mean/std -- distinct from TARP/SBC's own internal smoothing (test-set draws/
seeds/bootstrap, unaffected by N_REPEATS).

--- Resumability (unchanged) ---

Per-(model, n_train, repeat) file-level checkpointing (repeat_done()/raw_path()) means a killed or
timed-out run can simply be resubmitted -- already-written files are skipped, not redone.

Usage: python scripts/fit/run_experiment.py <model> [n_train] [dataset]
  model: one of "gaussian", "npe", "bnn_vanilla", "bnn_temp", "gibbs", "rasmussen_gibbs_retune",
  "bnn_studentt", "rasmussen_gibbs_noretune", "rasmussen_gibbs_noretune_extrasteps",
  "bnn_vanilla_extrasteps", "rasmussen_gibbs_noretune_10k", "rasmussen_gibbs_retune_10k",
  "bnn_wideprior", "bnn_temp_opt", "bnn_studentt_extrasteps", "bnn_vanilla_retune_10k",
  "bnn_gaussprior", "bnn_ebscale_10k", "bnn_ebstep_10k", "bnn_ebstep_retune_10k",
  "bnn_ebblock_iso_10k", "bnn_ebblock_layer_10k", "bnn_elu_h8", "bnn_vanilla_kfold", "npe_gausshead",
  "npe_gausshead_wide"
  n_train: optional, restricts the sweep to a single training size (one of SIZES below); omit for
  the full SIZES sweep in one job (pass "" to keep the full sweep when also giving dataset).
  dataset: optional, "rbf_kernel" (default -- everything this script has ever run) or "gp_surface"
  (2026-10-04; NOT wired up yet -- the models are still written against the two-target RBF task,
  so this exits with a message).
See slurm/run_experiment.sbatch for the batch-submission wrapper.
"""
import os
import sys
from pathlib import Path

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # this repository's src/
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401

import numpy as np
from tqdm.auto import tqdm

MODELS = ["npe", "bnn_vanilla", "bnn_temp", "gibbs", "rasmussen_gibbs_retune",
          "bnn_studentt", "rasmussen_gibbs_noretune", "rasmussen_gibbs_noretune_extrasteps",
          "bnn_vanilla_extrasteps", "rasmussen_gibbs_noretune_10k", "rasmussen_gibbs_retune_10k",
          "bnn_wideprior", "bnn_temp_opt", "bnn_studentt_extrasteps", "bnn_vanilla_retune_10k",
          "bnn_gaussprior", "bnn_ebscale_10k", "bnn_ebstep_10k", "bnn_ebstep_retune_10k",
          "bnn_ebblock_iso_10k", "bnn_ebblock_layer_10k", "bnn_elu_h8", "bnn_vanilla_kfold",
          "npe_gausshead", "npe_gausshead_wide"]
# FINAL PRODUCTION SWEEP (2026-10-29) -- see the "FINAL SWEEP" block below.
# 2026-11-02, PRETRAINING-TUNED RE-RUN: each final BNN model has a "_pretuned" twin, identical (seeds, subsamples,
# 12 members, budget, prior, retune loop) except (lr, wd) come from the PRETRAINING-ONLY grid
# (scripts/tuning/optimize_pretrain_grid.py -> results/hp_search/pretrain_grid/pretrain_grid_n{n}.npz, best_lr / best_wd)
# instead of bde_params*_n{n}.npz (tuned on the full MILE fit), and final_bnn_temp_pretuned reads its temperature
# from the search re-run under those values (temperature_search_n{n}_stdprior_pretuned_grid.npz). Pairs
# repeat-by-repeat with the original. final_bnn_stdnormal_pretuned ALSO writes the BNN without sampling from the same
# run (bnn.fit_correlated(score_pretrained=True)): final_bnn_pretrained_ens12_pretuned (12-network mixture) and
# final_bnn_pretrained_pretuned (member 0) -- replaces a separate final_bnn_pretrained_ens12-style job.
# 2026-11-03: final_bnn_temp_oldT_pretuned = final_bnn_temp_pretuned (grid lr/wd) but with the ORIGINAL final_bnn_temp
# temperatures (FINAL_TEMPERATURES, 0.2/0.3/0.6/0.8) instead of the re-tuned search -- separates the lr/wd change from
# the T change in the tempered comparison (at n=500 both T are 0.6, so it repeats final_bnn_temp_pretuned's config).
PRETUNED_BASE = {"final_bnn_stdnormal_pretuned": "final_bnn_stdnormal", "final_bnn_temp_pretuned": "final_bnn_temp",
                 "final_bnn_gaussprior_pretuned": "final_bnn_gaussprior",
                 "final_bnn_stdnormal_retune_pretuned": "final_bnn_stdnormal_retune",
                 "final_bnn_ebblock_pretuned": "final_bnn_ebblock",
                 "final_bnn_temp_oldT_pretuned": "final_bnn_temp"}
PRETUNED_PRETRAINED_FILES = ("final_bnn_pretrained_ens12_pretuned", "final_bnn_pretrained_pretuned")
FINAL_BNN_MODELS = ("final_bnn_stdnormal", "final_bnn_temp", "final_bnn_gaussprior",
                    "final_bnn_stdnormal_retune", "final_bnn_ebblock") + tuple(PRETUNED_BASE)
PRETUNED_GRID_DIR = RESULTS / "hp_search" / "pretrain_grid"
FINAL_ENSEMBLE_MODELS = ("final_npe_ens12",)   # each also writes its single-model file
# 2026-11-02: "the BNN without sampling" -- the 12 pretrained networks the stock-MILE final BNNs start MCLMC from,
# scored directly (bnn.fit_pretrained_ensemble); writes final_bnn_pretrained_ens12 (12-network mixture) and
# final_bnn_pretrained (member 0 alone). Replaces the Gaussian predictor as the no-sampling counterpart.
FINAL_PRETRAIN_MODELS = ("final_bnn_pretrained_ens12",)
FINAL_MODELS = FINAL_BNN_MODELS + FINAL_ENSEMBLE_MODELS + FINAL_PRETRAIN_MODELS
MODELS = MODELS + list(FINAL_MODELS)

import re
_manual_t = re.fullmatch(r"bnn_temp_T(\d+)p(\d+)", sys.argv[1]) if len(sys.argv) > 1 else None
MANUAL_TEMPERATURE = float(f"{_manual_t.group(1)}.{_manual_t.group(2)}") if _manual_t else None  # model 15

if len(sys.argv) < 2 or (sys.argv[1] not in MODELS and MANUAL_TEMPERATURE is None):
    raise SystemExit(f"Usage: python scripts/fit/run_experiment.py <model> [n_train]\n"
                      f"  model must be one of {MODELS}, got {sys.argv[1] if len(sys.argv) > 1 else None!r}")
MODEL = sys.argv[1]
BASE_MODEL = PRETUNED_BASE.get(MODEL, MODEL)   # a _pretuned model behaves exactly like its base apart from lr/wd/T

# Which dataset. Defaults to the RBF-kernel dataset, so every
# existing command is unchanged. GP surface results would go to results/gp_surface/raw/.
DATASETS = ("rbf_kernel", "gp_surface")
DATASET = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else "rbf_kernel"
if DATASET not in DATASETS:
    raise SystemExit(f"dataset must be one of {DATASETS}, got {DATASET!r}")
if DATASET != "rbf_kernel":
    raise SystemExit(
        "run_experiment.py: dataset 'gp_surface' is not wired up yet -- the shared models "
        "(bnn.py / npe.py) still assume the two-target RBF task.")

# DIAGNOSTIC, 2026-09-15: every real "npe" sweep attempt has hit CUBLAS_STATUS_NOT_INITIALIZED
# inside torch, on both shared and debug QOS (ruling out GPU contention as the cause).
# The one thing that reliably worked was scripts/npe_test.py, whose only
# difference from this script is that it never imports bnn (-> bde -> blackjax) or gaussian_predictor
# (-> flax). Importing torch (via npe.py) BEFORE any JAX-importing module is untested but cheap to
# try -- this conditional/reordered import block is a hypothesis-driven fix, not a confirmed root
# cause; revert to plain unconditional imports if it turns out not to matter.
from rbf_sbi import npe  # torch -- imported first, before anything that pulls in jax
from rbf_sbi import data
from rbf_sbi import metrics
from rbf_sbi import experiment_log
NPE_HEAD_MODELS = {"npe_gausshead": 16, "npe_gausshead_wide": 128}   # 2026-11-02: NPE with a Gaussian head (hidden width)
if MODEL not in ("npe", "final_npe_ens12") + tuple(NPE_HEAD_MODELS):
    import jax
    import jax.numpy as jnp
    import optax
    from rbf_sbi import bnn

# ---- Sweep hyperparameters ----
ALL_SIZES = [10000, 5000, 1000, 500, 100, 50]
SIZES = list(ALL_SIZES)
N_REPEATS = 5
if len(sys.argv) > 2 and sys.argv[2]:
    SIZES = [int(sys.argv[2])]
    if SIZES[0] not in ALL_SIZES:
        raise ValueError(f"n_train={SIZES[0]} is not one of this sweep's sizes: {ALL_SIZES}")

REPEATS = list(range(N_REPEATS))
if len(sys.argv) > 4 and sys.argv[4]:   # 2026-10-29: optional comma-separated repeat list, e.g. "0" or "0,1,2"
    REPEATS = [int(r) for r in sys.argv[4].split(",")]
    # 2026-11-04: repeats beyond N_REPEATS allowed (low-n extension, repeats 5-19); each repeat still draws its own
    # subsample (data.subsample_train, default_rng(123 + repeat)). Upper bound 100: seed = 100 * size_index + repeat
    # must not reach the next size's seeds.
    assert all(0 <= r < 100 for r in REPEATS), REPEATS

RESULTS_DIR = RESULTS
RAW_DIR = RESULTS_DIR / "raw"
# 2026-10-29: RUN_EXPERIMENT_OUT_DIR redirects every output file of any model to a scratch folder and
# suppresses the experiment-log entry -- for smoke tests and for the reproduction check
# after a shared-model change (re-run an existing configuration without touching its saved file).
OUT_OVERRIDE = os.environ.get("RUN_EXPERIMENT_OUT_DIR", "")
if OUT_OVERRIDE:
    RAW_DIR = Path(OUT_OVERRIDE)

# NPE (conditional MAF normalizing flow — see src/npe.py)
DEFAULT_NPE_LR, DEFAULT_NPE_WD = 1e-4, 0.0
NPE_STEPS, NPE_EVAL_EVERY, NPE_PATIENCE = 20000, 100, 20
NPE_N_POSTERIOR_SAMPLES = 5000

# ---- BNN (bde/MILE, correlated) — see src/bnn.py. Matches scripts/bde_correlated_perchain_test.py's
# current production settings. ----
BNN_N_MEMBERS = 3
BNN_HIDDEN_LAYERS = [16, 16, 16]
BNN_EPOCHS, BNN_PATIENCE, BNN_VALIDATION_SPLIT = 2000, 5, 0.2
BNN_N_SAMPLES, BNN_WARMUP_STEPS, BNN_N_THINNING = 10000, 50000, 10
DEFAULT_BNN_LR, DEFAULT_BNN_WD = 1e-3, 0.0
BNN_DESIRED_ENERGY_VAR_START, BNN_DESIRED_ENERGY_VAR_END = 0.5, 0.1
BNN_MAX_SEED_TRIES, BNN_RHAT_THRESHOLD = 3, 1.01
BNN_MIN_GOOD_MEMBERS = BNN_N_MEMBERS
BNN_PRETRAIN_EVAL_EVERY, BNN_PRETRAIN_MIN_STEPS = 100, 500
BNN_ACTIVATION = "soft_sign"   # bnn.fit_correlated's default; stated here so bnn_elu_h8's difference is explicit

# bnn_elu_h8 (2026-11-02) -- bnn_vanilla with a small network: 100 -> 8 -> 8 -> 5, ELU (925 weights,
# vs. 2,245 for 16-16-16 soft-sign). Everything else identical to bnn_vanilla (N(0,1), T=1, 3 members,
# 50,000 warmup / 10,000 steps / 1,000 kept, same seeds, subsamples and per-size tuned lr/wd).
SMALL_BNN_HIDDEN_LAYERS = [8, 8]
SMALL_BNN_ACTIVATION = "elu"

# bnn_vanilla_kfold (2026-11-02) -- bnn_vanilla with per-member K-fold pretraining (bnn.fit_correlated(pretrain_kfold=K)):
# member m pretrains on all points outside fold m and early-stops on fold m, instead of all members sharing bde's one
# 80/20 split. K = 5 keeps the per-member sizes at the default 80/20 (40 / 10 at n=50).
PRETRAIN_KFOLD_K = 5

# ---- FINAL SWEEP (2026-10-29) -- the production run behind the write-up. Everything not listed here is the existing setting of the matching
# model: final_bnn_stdnormal = bnn_vanilla, final_bnn_temp = bnn_temp_T<a>p<b> at the per-size T
# below, final_bnn_gaussprior = bnn_gaussprior, final_bnn_stdnormal_retune = bnn_vanilla_retune_10k,
# final_bnn_ebblock = bnn_ebblock_iso_10k -- same seeds, subsamples and hyperparameter composite --
# with 12 members instead of 3 (MILE's default configuration), every kept draw's test-set output saved
# to FINAL_DRAWS_DIR, and wall time + GPU type recorded. final_npe_ens12 / final_gaussian_ens12: 12
# networks on the SAME training/validation split with different initialisation seeds (member 0 uses
# the single model's seed, so member 0 IS the single model, written as final_npe / final_gaussian);
# the ensemble is the equal-weight mixture. ASSUMPTIONS: same split for every member (deep-ensemble
# convention: data fixed, initialisation varied); 12 ensemble members chosen to match the BNN.
FINAL_N_MEMBERS = 12
FINAL_TEMPERATURES = {50: 0.2, 100: 0.3, 500: 0.6, 1000: 0.8}   # the per-size T of bnn_temp_T<a>p<b>; T~1 from 5,000
FINAL_SMOKE = os.environ.get("FINAL_SMOKE", "") == "1"         # tiny budgets, output to FINAL_OUT_DIR, no log entry
if MODEL in FINAL_MODELS:
    if FINAL_SMOKE:
        if not OUT_OVERRIDE:
            raise SystemExit("FINAL_SMOKE=1 needs RUN_EXPERIMENT_OUT_DIR (a scratch folder, never results/)")
        FINAL_N_MEMBERS = 3
        BNN_EPOCHS, BNN_WARMUP_STEPS, BNN_N_SAMPLES = 200, 300, 1000
        NPE_STEPS = 300
    BNN_N_MEMBERS = FINAL_N_MEMBERS
    BNN_MIN_GOOD_MEMBERS = BNN_N_MEMBERS
FINAL_DIR = Path(OUT_OVERRIDE) if OUT_OVERRIDE else RESULTS_DIR / "raw" / "final"
FINAL_DRAWS_DIR = (Path(OUT_OVERRIDE) / "draws" if OUT_OVERRIDE
                   else SCRATCH / "sbi_final_draws")
if BASE_MODEL == "final_bnn_temp":
    if len(sys.argv) > 2 and sys.argv[2]:
        if SIZES[0] not in FINAL_TEMPERATURES:
            raise SystemExit(f"final_bnn_temp runs only at n in {sorted(FINAL_TEMPERATURES)} (T~1 above)")
    else:
        SIZES = [n for n in SIZES if n in FINAL_TEMPERATURES]

# See module docstring's "bnn_temp" section for the full transfer-assumption caveat.
BNN_TEMP_TEMPERATURE = 0.3050161024479261  # results/temperature_search_n100_optprior.npz's best_temperature

# bnn_wideprior -- see module docstring's model 13. N(0,1)^T is N(0, 1/T) up to a constant.
WIDEPRIOR_SCALE = float(1.0 / np.sqrt(BNN_TEMP_TEMPERATURE))  # ~1.811

# bnn_gaussprior -- see module docstring's model 18. Read from the file (not hardcoded) so the
# value is exactly the one scripts/tuning/optimize_prior_width.py found; only needed for that model.
GAUSSPRIOR_PATH = RESULTS / "gaussian_width_prior_phi.npz"
GAUSSPRIOR_SCALE = None
if MODEL == "bnn_gaussprior" or BASE_MODEL == "final_bnn_gaussprior":
    with np.load(GAUSSPRIOR_PATH) as _gp:
        # The width is only valid for the network it was calibrated on.
        assert list(_gp["sizes"]) == [100, 16, 16, 16, 5] and str(_gp["activation"]) == "soft_sign", \
            f"{GAUSSPRIOR_PATH.name} was calibrated for a different network: {_gp['sizes']}, {_gp['activation']}"
        GAUSSPRIOR_SCALE = float(_gp["width"])  # 0.36553
    print(f"{MODEL}: prior N(0, {GAUSSPRIOR_SCALE:.5f}) on every weight ({GAUSSPRIOR_PATH.name})")

# Gibbs (see src/bnn.fit_gibbs_hierarchical's own docstring) -- matches scripts/bde_gibbs_test.py's
# own defaults, kept in sync manually.
GIBBS_IG_A, GIBBS_IG_B, GIBBS_TAU_INIT = 2.0, 1.0, 1.0
GIBBS_N_BURNIN_SWEEPS = 15
GIBBS_BURNIN_WARMUP_STEPS, GIBBS_BURNIN_N_SAMPLES, GIBBS_BURNIN_N_THINNING = 2000, 500, 1

# Rasmussen-style single-chain Gibbs WITH periodic warm-started (L, step_size) retuning -- see
# module docstring's "rasmussen_gibbs_retune" section. Shares GIBBS_IG_A/IG_B/TAU_INIT above (same
# hyperprior as "gibbs", for a fair comparison). Cadence/budget matches
# scripts/bde_rasmussen_gibbs_retune_test.py's own validated (n=100, n=10,000) settings exactly --
# NOT re-validated at this sweep's other sizes before this sweep runs (see module docstring).
RASMUSSEN_RETUNE_EVERY_SAMPLES = 100
RASMUSSEN_RETUNE_TUNE1_STEPS, RASMUSSEN_RETUNE_TUNE2_STEPS, RASMUSSEN_RETUNE_TUNE3_STEPS = 200, 50, 0
RASMUSSEN_TAU_SAVE_STRIDE = 10

# Fixed Student-t prior matching the Gibbs models' marginal weight prior exactly -- see module
# docstring's "bnn_studentt" section. Derived from GIBBS_IG_A/IG_B, not set independently, so the
# two can't drift apart.
STUDENTT_NU, STUDENTT_SCALE = 2.0 * GIBBS_IG_A, float(np.sqrt(GIBBS_IG_B / GIBBS_IG_A))

# No-retune Rasmussen variants -- see module docstring's models 8/9. 13 = ceil(12.5), the retuned
# run's own chain steps per kept sample ((100*BNN_N_THINNING + 200 + 50) / 100).
RASMUSSEN_EXTRASTEPS_N_THINNING = 13
RASMUSSEN_NORETUNE_MODELS = ("rasmussen_gibbs_noretune", "rasmussen_gibbs_noretune_extrasteps",
                             "rasmussen_gibbs_noretune_10k", "bnn_ebstep_10k")
RASMUSSEN_RETUNE_MODELS = ("rasmussen_gibbs_retune", "rasmussen_gibbs_retune_10k", "bnn_vanilla_retune_10k",
                           "bnn_ebscale_10k", "bnn_ebstep_retune_10k",
                           "bnn_ebblock_iso_10k", "bnn_ebblock_layer_10k")
# Rasmussen sampler with NO per-weight tau draw: tau held at 1 (plain N(0,1)), or -- for
# EB_MODELS -- one global tau estimated by EM and then frozen.
FIXED_TAU_MODELS = ("bnn_vanilla_retune_10k", "bnn_ebscale_10k", "bnn_ebstep_10k", "bnn_ebstep_retune_10k",
                    "bnn_ebblock_iso_10k", "bnn_ebblock_layer_10k")
EB_MODELS = ("bnn_ebscale_10k",)                            # block Monte Carlo EM, then frozen (not run)
EB_STEP_MODELS = ("bnn_ebstep_10k", "bnn_ebstep_retune_10k")  # scale reset every chain step, init from Adam weights
# Block empirical Bayes: scale(s) updated between retune segments, fixed within one. model -> grouping.
EB_BLOCK_MODELS = {"bnn_ebblock_iso_10k": "iso", "bnn_ebblock_layer_10k": "layer"}
EB_ADAPT_SEGMENTS = 150  # unsaved EM iterations before production -- see module docstring's model 19
RASMUSSEN_MODELS = RASMUSSEN_RETUNE_MODELS + RASMUSSEN_NORETUNE_MODELS

# Step-matched controls -- see the module docstring's STEP-COUNT CORRECTION. The standard
# (fit_correlated) path counts n_samples as RAW steps, the Rasmussen path as KEPT samples.
# bnn_vanilla_extrasteps: rasmussen_gibbs_retune's real budget, on the standard path.
BNN_EXTRASTEPS_N_STEPS = (BNN_N_SAMPLES * BNN_N_THINNING
                          + (BNN_N_SAMPLES // RASMUSSEN_RETUNE_EVERY_SAMPLES)
                          * (RASMUSSEN_RETUNE_TUNE1_STEPS + RASMUSSEN_RETUNE_TUNE2_STEPS))  # 125,000
BNN_EXTRASTEPS_N_KEPT = BNN_N_SAMPLES                                                     # 10,000
# *_10k Rasmussen models: bnn_vanilla's real budget, on the Rasmussen path.
RASMUSSEN_10K_N_KEPT = BNN_N_SAMPLES // BNN_N_THINNING  # 1,000 kept x BNN_N_THINNING = 10,000 steps
RASMUSSEN_10K_MODELS = ("rasmussen_gibbs_noretune_10k", "rasmussen_gibbs_retune_10k", "bnn_vanilla_retune_10k",
                        "bnn_ebscale_10k", "bnn_ebstep_10k", "bnn_ebstep_retune_10k",
                        "bnn_ebblock_iso_10k", "bnn_ebblock_layer_10k")


def chain_budget(model):
    """(raw post-warmup chain steps incl. unsaved retune steps, kept samples) per chain."""
    model = PRETUNED_BASE.get(model, model)
    if model in ("final_bnn_stdnormal", "final_bnn_temp", "final_bnn_gaussprior"):
        return BNN_N_SAMPLES, -(-BNN_N_SAMPLES // BNN_N_THINNING)
    if model in ("final_bnn_stdnormal_retune", "final_bnn_ebblock"):
        kept = BNN_N_SAMPLES // BNN_N_THINNING
        return (kept * BNN_N_THINNING + (kept // RASMUSSEN_RETUNE_EVERY_SAMPLES)
                * (RASMUSSEN_RETUNE_TUNE1_STEPS + RASMUSSEN_RETUNE_TUNE2_STEPS)), kept
    if model in RASMUSSEN_MODELS:
        kept = RASMUSSEN_10K_N_KEPT if model in RASMUSSEN_10K_MODELS else BNN_N_SAMPLES
        thin = (RASMUSSEN_EXTRASTEPS_N_THINNING if model == "rasmussen_gibbs_noretune_extrasteps"
                else BNN_N_THINNING)
        retune_steps = ((kept // RASMUSSEN_RETUNE_EVERY_SAMPLES)
                        * (RASMUSSEN_RETUNE_TUNE1_STEPS + RASMUSSEN_RETUNE_TUNE2_STEPS)
                        if model in RASMUSSEN_RETUNE_MODELS else 0)
        return kept * thin + retune_steps, kept
    if model in ("bnn_vanilla_extrasteps", "bnn_studentt_extrasteps"):
        return BNN_EXTRASTEPS_N_STEPS, BNN_EXTRASTEPS_N_KEPT
    if (model in ("bnn_vanilla", "bnn_temp", "bnn_studentt", "gibbs", "bnn_wideprior", "bnn_temp_opt",
                  "bnn_gaussprior", "bnn_elu_h8", "bnn_vanilla_kfold")
            or MANUAL_TEMPERATURE is not None):
        return BNN_N_SAMPLES, -(-BNN_N_SAMPLES // BNN_N_THINNING)
    return None, None

# What each 2026-10-01 model is being run FOR, and what each outcome would imply. Compare
# per-repeat PAIRED differences (every repeat shares its training subsample across models), not
# each model's own across-repeat spread.
MODEL_PURPOSE = {
    **{m: ("FINAL SWEEP (2026-10-29): production run behind the write-up; 12 members/networks, every "
           "kept draw saved, wall time recorded.")
       for m in FINAL_MODELS},
    **{m: ("NPE head test (2026-11-02): NPE (same encoder, split, seeds, lr/wd, optimiser, stopping rule) with the "
           f"flow replaced by a conditional bivariate Gaussian head ({h} hidden units). Does NPE's better sigma mean at "
           "n=50 come from the flow head?") for m, h in NPE_HEAD_MODELS.items()},
    **{m: (f"PRETRAINING-TUNED RE-RUN (2026-11-02): {b} (same seeds, subsamples, 12 members, budget, prior) with (lr, wd) "
           "from the pretraining-only grid (results/hp_search/pretrain_grid) instead of the full-fit search"
           + ("; temperature from the search re-run under those values" if m == "final_bnn_temp_pretuned" else "")
           + ("; temperature = the ORIGINAL final_bnn_temp values (FINAL_TEMPERATURES), not re-tuned"
              if m == "final_bnn_temp_oldT_pretuned" else "")
           + ("; also writes the BNN without sampling (final_bnn_pretrained_ens12_pretuned / final_bnn_pretrained_pretuned)"
              " from the same run" if m == "final_bnn_stdnormal_pretuned" else "")
           + f". Pairs with {b} by repeat.") for m, b in PRETUNED_BASE.items()},
    "final_bnn_pretrained_ens12": (
        "BNN WITHOUT SAMPLING (2026-11-02): the 12 pretrained networks that final_bnn_stdnormal / _gaussprior / _temp "
        "start MCLMC from (same pretraining code, seed, data, lr/wd; pretraining does not use the weight prior), scored "
        "directly as a 12-network Gaussian mixture (final_bnn_pretrained_ens12) and as member 0 alone "
        "(final_bnn_pretrained). The no-sampling counterpart of the BNN; replaces the Gaussian predictor in that role."),
    "bnn_vanilla_kfold": (
        "Per-member k-fold pretraining (2026-11-02): bnn_vanilla, but member m pretrains on all points outside fold m "
        "(K=5) and early-stops on fold m, instead of every member sharing bde's one 80/20 split. Does pretraining on a "
        "single shared 10-point validation set explain part of the n=50 underperformance?"),
    "bnn_elu_h8": (
        "Small-network check (2026-11-02): bnn_vanilla with 100->8->8->5 ELU (925 weights). n=1,000: "
        "does it fit correctly (NLL/width near bnn_vanilla's)? n=50: does the predictive still sit at "
        "about the theta-prior width? Same N(0,1) weight prior, so the implied output prior differs "
        "from bnn_vanilla's."),
    "bnn_studentt": (
        "PURPOSE: test whether rasmussen_gibbs_retune's low-data advantage over bnn_vanilla comes "
        "from its prior's SHAPE or from the Gibbs sampler/retuning. This model is bnn_vanilla's "
        "exact code path with a fixed Student-t(nu=4, scale=0.7071) prior -- the exact marginal of "
        "the Gibbs prior (N(0,tau^2), tau^2~IG(2,1)), so both target the same weight posterior. "
        "IF it matches rasmussen_gibbs_retune on NLL/first+second moment/median|z|/TARP: the gain "
        "is the heavy-tailed prior; the Gibbs machinery is not needed for it (retuning may still "
        "matter for mixing). IF it is clearly worse than rasmussen_gibbs_retune: check R-hat/ESS "
        "FIRST -- this model has no retune, so it may simply be unmixed, which says nothing about "
        "the prior; only if it mixes and still loses does the sampler itself (tau sampling, "
        "retuning, extra steps) matter beyond the prior. IF it is no better than bnn_vanilla: the "
        "heavy tails are not what helps."
    ),
    "rasmussen_gibbs_noretune": (
        "PURPOSE: equal-step baseline (100,000 chain steps; NOT the same as bnn_vanilla, which "
        "runs 10,000 -- see the module docstring's STEP-COUNT CORRECTION) for "
        "rasmussen_gibbs_noretune_extrasteps -- rasmussen_gibbs_retune with retuning OFF, nothing "
        "else changed. By itself: IF it already beats bnn_vanilla (paired) on a metric, that gain "
        "needs neither retuning nor extra steps; whatever gap remains to rasmussen_gibbs_retune "
        "is the part the retune (L/step-size correction and/or its 25% extra steps) accounts for."
    ),
    "rasmussen_gibbs_noretune_extrasteps": (
        "PURPOSE: separate the two things rasmussen_gibbs_retune's retune adds -- a corrected (L, "
        "step_size) and ~25% more chain steps (250 unsaved steps per 100 kept samples). This model "
        "has the extra steps (n_thinning 13, 130,000 steps) but NO retuning. Read against "
        "rasmussen_gibbs_noretune (100,000 steps, no retune) and rasmussen_gibbs_retune (125,000 "
        "steps, retuned). IF it matches rasmussen_gibbs_retune: the retune's benefit (second "
        "moment, median|z|, TARP, mixing) is just more chain steps, and a longer/more-thinned run "
        "would do as well. IF it matches rasmussen_gibbs_noretune: the extra steps do nothing; "
        "the benefit is the (L, step_size) correction (L ~55 -> ~20 in the first few retunes at "
        "n=100), so a correct one-time warmup should suffice. IF it lands in between: both "
        "contribute."
    ),
    "bnn_vanilla_extrasteps": (
        "PURPOSE: give the N(0,1) BNN the chain budget rasmussen_gibbs_retune actually has -- "
        "125,000 chain steps, 10,000 kept samples -- instead of bnn_vanilla's 10,000 steps / 1,000 "
        "kept. Every earlier Gibbs-vs-N(0,1) comparison had the Gibbs chain 10-12.5x longer. Read "
        "against rasmussen_gibbs_retune (paired, per repeat) and against bnn_vanilla. IF it "
        "matches rasmussen_gibbs_retune on NLL/first+second moment/median|z|/TARP: the reported "
        "Gibbs advantage was chain length, not the prior or the sampler. IF rasmussen_gibbs_retune "
        "still wins at equal steps: the advantage is real (prior shape and/or retuning -- "
        "bnn_studentt and the _10k models separate those). IF it is no better than bnn_vanilla: "
        "N(0,1) is not step-limited at this budget."
    ),
    "bnn_wideprior": (
        "PURPOSE: test whether bnn_temp's improvement over bnn_vanilla (n=100: aleatoric width "
        "roughly halved, relative std error sigma 1.15 -> 0.43, rho 5.64 -> 3.08, 5/5 repeats) "
        "comes from the prior being effectively weakened. bnn_temp samples likelihood^(1/T) * "
        "N(0,1) with T=0.305, which equals (likelihood * N(0, 1/T))^(1/T): a fully tempered "
        "version of THIS model's posterior. This model is that inner posterior at T=1 -- "
        "bnn_vanilla with prior scale 1/sqrt(T) ~ 1.811, nothing else changed. The two share "
        "their posterior mode and differ only in concentration (bnn_temp's is sharper by 1/T). "
        "Read against bnn_vanilla and bnn_temp, paired per repeat. IF it matches bnn_temp on "
        "second moment / aleatoric width / rho first moment: the tempering gain is the weaker "
        "prior (where the posterior sits), not the extra concentration -- supports the "
        "prior-misspecification reading (Fortuin et al.), and N(0,1) is too STRONG here; a "
        "prior fix alone can replace tempering. IF it matches bnn_vanilla (or is worse/wider): "
        "weakening the prior does nothing by itself, so the gain comes from the 1/T sharpening "
        "of the whole posterior, i.e. up-weighting the data -- no choice of prior at T=1 "
        "reproduces that, and the prior-misspecification reading does not explain bnn_temp. IF "
        "it lands in between: both contribute. CHECK R-hat/ESS before reading either way -- a "
        "wider prior may mix differently, and bnn_temp itself is unmixed (ESS min ~3). "
        "ASSUMPTION: the identity is exact for the densities; it says nothing about what an "
        "unadjusted, finite-length MCLMC chain does with each."
    ),
    "bnn_gaussprior": (
        "PURPOSE: test whether narrowing the weight prior's SCALE (not its shape) fixes the "
        "low-data over-width that Student-t and Gibbs priors, both of scale ~1, did not. Prior is "
        "N(0, 0.36553) on every weight -- the width that makes the UNTRAINED network's prior "
        "predictive calibrated (scripts/tuning/optimize_prior_width.py; coverage AUC +0.0037 vs. N(0,1)'s "
        "+0.1999), chosen with no training data, so it is legitimate prior elicitation and the "
        "same at every n_train. Everything else is bnn_vanilla (tuned lr/wd, seeds, subsamples, "
        "10,000 steps / 1,000 kept, T=1). Read against bnn_vanilla and the per-size tempered runs, "
        "paired per repeat. Earlier evidence is single n=100 runs only (width error ~0.60 / 3.4 "
        "vs. N(0,1)'s 1.15 / 5.64), never replicated. IF widths/aleatoric improve at n<=1,000 by "
        "as much as tempering: prior scale is a data-free substitute for tempering. IF no change: "
        "the over-width is not a prior-scale effect. CHECK that n>=5,000 does not get worse (a "
        "narrower prior could over-regularize), and R-hat/ESS before reading either way."
    ),
    "bnn_ebscale_10k": (
        "PURPOSE: let the training data choose the weight prior's overall scale (empirical Bayes), "
        "instead of fixing it at 1 (bnn_vanilla), fixing it data-free at 0.366 (bnn_gaussprior), or "
        "redrawing per-weight scales from a fixed hyperprior (rasmussen_gibbs_*, which left the "
        "marginal prior unchanged and matched a fixed Student-t). One global tau is fit by Monte "
        "Carlo EM (tau^2 <- mean w^2 over all weights and the second half of each 1,000-step "
        "segment, then a warm (L, step_size) retune) for EB_ADAPT_SEGMENTS unsaved segments, then "
        "frozen; production is bnn_vanilla_retune_10k's chain at that tau. Read against "
        "bnn_vanilla_retune_10k (same sampler, tau=1 -- the clean control), bnn_gaussprior and "
        "bnn_vanilla, paired per repeat. FIRST check rasmussen_eb_tau_history: whether tau had "
        "stopped moving, and whether the 3 members agree. IF tau lands near 0.366 and results "
        "match bnn_gaussprior: the data-free calibration and the data agree on the scale. IF tau "
        "stays near 1 at small n: EM did not converge in the budget (prior-dominated posterior "
        "contracts slowly) and the run says nothing about the EB optimum. CAVEATS: data used "
        "twice, so check coverage for over-confidence; tau is per member, not shared."
    ),
    "bnn_ebstep_10k": (
        "PURPOSE: empirical-Bayes prior scale at bnn_vanilla's real budget (10,000 steps / 1,000 "
        "kept, nothing extra). One scale for all weights: starts at the RMS of the member's "
        "Adam-pretrained weights (the prior the warmup runs under), then is reset to the RMS of "
        "the current weights at every chain step -- the prior is recomputed from the weights, not "
        "redrawn from a fixed hyperprior as in rasmussen_gibbs_*. Read against bnn_vanilla "
        "(scale 1) and bnn_gaussprior (scale 0.366, data-free), paired per repeat. FIRST read "
        "rasmussen_eb_tau_history: where the scale starts (pretrained RMS was ~0.16-0.21 in a "
        "smoke test), whether it drifts, and whether the 3 members agree. IF it settles near "
        "0.366 and matches bnn_gaussprior: the data and the data-free calibration agree. IF it "
        "stays near the pretrained RMS: the result is mostly the initialisation. IF it collapses "
        "or grows without bound: the scale-free implied prior is unstable at that n. CAVEATS: the "
        "scale is never frozen, so the kept samples are not from a posterior under one fixed "
        "Gaussian prior; data used twice -- check coverage for over-confidence."
    ),
    "bnn_ebstep_retune_10k": (
        "PURPOSE: bnn_ebstep_10k plus the periodic (L, step_size) retune (12,500 steps / 1,000 "
        "kept), since the scale moves during sampling and (L, step_size) are otherwise frozen at "
        "their warmup values. Like-for-like control: bnn_vanilla_retune_10k (same sampler, scale "
        "fixed at 1). The difference between this and bnn_ebstep_10k is what the retune adds."
    ),
    "bnn_ebblock_iso_10k": (
        "PURPOSE: empirical-Bayes prior scale adapted DURING sampling without the per-step "
        "feedback that made bnn_ebstep_* unstable. One scale for all weights, started at the "
        "pretrained-weight RMS, held fixed within each 1,000-step block and updated between "
        "blocks from the block's per-weight mean and variance (gamma = d - sum(v)/tau^2, "
        "tau^2 = |m|^2/gamma, damped), followed by the (L, step_size) retune. Budget identical "
        "to bnn_vanilla_retune_10k (scale fixed at 1, same sampler) -- the like-for-like "
        "control; also read against bnn_gaussprior (0.366, data-free) and bnn_ebstep_retune_10k. "
        "FIRST read rasmussen_ebblock_tau_history: whether the scale converges within 10 blocks, "
        "whether the 3 members agree, and rasmussen_ebblock_n_gamma_clamped / _n_step_capped "
        "(nonzero = the update hit a safeguard). IF it settles near 0.366 and matches "
        "bnn_gaussprior: data and data-free calibration agree. IF it widens with n and beats "
        "bnn_gaussprior at n>=1,000: the data-chosen width removes the fixed width's "
        "over-regularisation. IF it collapses or runs to the cap at n=50: the evidence itself "
        "does not pin the scale there. CAVEATS: early blocks are sampled under a different prior "
        "than late ones and all 1,000 kept samples are retained; data used twice -- check "
        "coverage; the update rule and its damping are this project's own, untested."
    ),
    "bnn_ebblock_layer_10k": (
        "PURPOSE: bnn_ebblock_iso_10k with one scale per layer (4 groups: 1,616 / 272 / 272 / 85 "
        "weights), each started at that layer's pretrained RMS and updated with its own gamma. "
        "Read against bnn_ebblock_iso_10k, paired per repeat: the difference is what prior "
        "STRUCTURE across layers adds over a single data-chosen width (the 2026-10-02 'correlated "
        "tau across layers' question). FIRST read rasmussen_ebblock_tau_history per layer: "
        "whether the layers settle on different scales, and whether any layer collapses to the "
        "floor (the 85-weight output layer is the least well determined)."
    ),
    "bnn_vanilla_retune_10k": (
        "PURPOSE: test whether the periodic (L, step_size) retune helps a plain N(0,1) BNN. At "
        "10,000 steps, rasmussen_gibbs_retune_10k beat rasmussen_gibbs_noretune_10k on NLL, both "
        "first moments and both widths on 5/5 repeats at n=500 and n=1,000, while the no-retune "
        "Gibbs sampler was indistinguishable from bnn_vanilla -- so at those sizes the gain came "
        "from the retune, not the prior or the tau sampling. This model is that same sampler and "
        "retune schedule with tau held fixed at 1 (N(0,1) prior, no Gibbs step). Read against "
        "bnn_vanilla (paired, per repeat; same target, same steps, no retune) and "
        "rasmussen_gibbs_retune_10k. IF it matches rasmussen_gibbs_retune_10k: the gain is a "
        "sampler-tuning effect available to any prior -- nothing to do with Gibbs or heavy tails. "
        "IF it matches bnn_vanilla: the retune only helps when tau is being sampled (it corrects "
        "the tau=1 warmup's mismatch), so N(0,1)'s own warmup tuning is not the problem. IF in "
        "between: both. CAVEATS: (a) a retune also moves the chain 250 unsaved steps, not "
        "separated from the (L, step_size) update; (b) adaptation continues through the kept "
        "samples with no Metropolis correction, so a better fit is not by itself evidence of a "
        "more correct posterior; (c) this runs on the Rasmussen code path (one regressor per "
        "member, no retry on a failed member), not fit_correlated -- no fixed-tau, NO-retune run "
        "on this path exists to confirm the two paths agree when neither retunes."
    ),
    "rasmussen_gibbs_noretune_10k": (
        "PURPOSE: the Gibbs sampler at bnn_vanilla's real chain budget -- 10,000 chain steps, 1,000 "
        "kept, no retune. Read against bnn_vanilla and bnn_studentt (paired, per repeat; same "
        "steps, same kept count). IF it still beats bnn_vanilla: the Gibbs advantage does not "
        "need the longer chain. IF it matches bnn_vanilla: the earlier advantage was the 10x "
        "longer chain. vs. bnn_studentt (same target posterior, same steps): any gap is the "
        "sampler (per-step tau redraw, N(0,1)-tuned frozen L/step_size), not the prior."
    ),
    "rasmussen_gibbs_retune_10k": (
        "PURPOSE: rasmussen_gibbs_retune at bnn_vanilla's real chain budget -- 10,000 chain steps, "
        "1,000 kept, plus 10 retunes x 250 unsaved steps (12,500 total). Read against "
        "rasmussen_gibbs_noretune_10k (what the retune adds at this length), bnn_vanilla and "
        "bnn_studentt. CAVEAT: only 10 retunes; at n=100 the full-length run's L correction took "
        "the first 2-3, so a sizeable fraction of this chain runs before it lands."
    ),
}

TARP_N_DRAWS, TARP_N_BOOT = 1000, 200
TARP_SEEDS = list(range(20))


# ---- Per-size tuned-hyperparameter loading ----

def _load_sized_hp_npz(path, label):
    """Loads an optimize_npe.py/optimize_gaussian.py-style aggregated {sizes, lr, wd} .npz into a
    {n_train: (lr, wd)} dict. Returns {} (falls back to defaults everywhere) if the file doesn't
    exist yet."""
    if not path.exists():
        print(f"[warn] {path.name} not found -- falling back to default {label} lr/wd for every "
              f"size. Run scripts/optimize_{label.lower()}.py first for tuned values.")
        return {}
    with np.load(path) as f:
        return {int(n): (float(f["lr"][i]), float(f["wd"][i])) for i, n in enumerate(f["sizes"])}


HP_TAG_BY_SIZE = {}   # n -> "hpnew" or "" -- set by _load_bnn_hp, read by raw_path
HP_FILE_BY_SIZE = {}  # n -> file name the (lr, weight_decay) came from


def _load_bnn_hp(n):
    """BDE's tuned hyperparameters are one-file-per-size (results/bde_params_n{n}.npz), not a
    single aggregated file -- see legacy/optimize_bde.py. Prefers results/bde_params_new_n{n}.npz
    when it exists -- see the module docstring's HYPERPARAMETER VERSIONS."""
    if MODEL in PRETUNED_BASE:
        grid_path = PRETUNED_GRID_DIR / f"pretrain_grid_n{n}.npz"
        if not grid_path.exists():
            raise FileNotFoundError(f"{grid_path} not found -- run scripts/tuning/optimize_pretrain_grid.py {n} first.")
        with np.load(grid_path) as f:
            if bool(f["smoke"]):
                raise RuntimeError(f"{grid_path.name} is a smoke-test grid, not a result.")
            lr, wd = float(f["best_lr"]), float(f["best_wd"])
        HP_TAG_BY_SIZE[n] = ""
        HP_FILE_BY_SIZE[n] = f"hp_search/pretrain_grid/{grid_path.name}"
        print(f"BNN hyperparameters for n={n}: {HP_FILE_BY_SIZE[n]} (pretraining-only grid best: lr={lr:.3e}, wd={wd:.3e})")
        return lr, wd
    new_path = RESULTS_DIR / f"bde_params_new_n{n}.npz"
    # BDE_HP_VERSION=old forces the original file even when a re-run search exists -- for finishing
    # a run that was started (and partly written) under the old hyperparameters.
    force_old = os.environ.get("BDE_HP_VERSION", "") == "old"
    path = new_path if (new_path.exists() and not force_old) else RESULTS_DIR / f"bde_params_n{n}.npz"
    HP_TAG_BY_SIZE[n] = "hpnew" if path == new_path else ""
    HP_FILE_BY_SIZE[n] = path.name
    print(f"BNN hyperparameters for n={n}: {path.name}")
    if not path.exists():
        print(f"[warn] {path.name} not found -- falling back to default BNN lr/wd for n={n}. "
              f"Run legacy/optimize_bde.py first for tuned values.")
        return DEFAULT_BNN_LR, DEFAULT_BNN_WD
    with np.load(path) as f:
        return float(f["lr"]), float(f["weight_decay"])


def _load_pretuned_temperature(n):
    """final_bnn_temp_pretuned's temperature: scripts/tuning/optimize_temperature_grid.py re-run with PRETUNED_HP=1."""
    path = RESULTS_DIR / f"temperature_search_n{n}_stdprior_pretuned_grid.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run PRETUNED_HP=1 scripts/tuning/optimize_temperature_grid.py {n} first.")
    with np.load(path) as f:
        if not bool(f["complete"]):
            raise RuntimeError(f"{path.name} is an incomplete (timed-out) search -- finish it first.")
        return float(f["best_temperature"])


def _load_searched_temperature(n):
    """Per-size temperature for "bnn_temp_opt" -- see scripts/tuning/optimize_temperature_grid.py."""
    tag = "_hpnew" if HP_TAG_BY_SIZE.get(n) else ""  # searched under the same hyperparameters
    path = RESULTS_DIR / f"temperature_search_n{n}_stdprior{tag}_grid.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/tuning/optimize_temperature_grid.py {n} first.")
    with np.load(path) as f:
        if not bool(f["complete"]):
            raise RuntimeError(f"{path.name} is an incomplete (timed-out) search -- finish it first.")
        return float(f["best_temperature"])


def get_hp(hp_dict, n, default_lr, default_wd):
    return hp_dict.get(n, (default_lr, default_wd))


def raw_path(model, n, repeat):
    """sweep_{model}_n{n}_r{repeat}.npz, or sweep_{model}_hpnew_n{n}_r{repeat}.npz when this
    size's (lr, weight_decay) were loaded from the re-run search -- see HYPERPARAMETER VERSIONS.
    Final-sweep models: FINAL_DIR/{model}_n{n}_r{repeat}.npz (the file records hp_file instead)."""
    if model.startswith("final_"):
        return FINAL_DIR / f"{model}_n{n}_r{repeat}.npz"
    tag = HP_TAG_BY_SIZE.get(n, "")
    return RAW_DIR / (f"sweep_{model}_{tag}_n{n}_r{repeat}.npz" if tag else f"sweep_{model}_n{n}_r{repeat}.npz")


def repeat_done(model, n, repeat):
    if model == "final_bnn_stdnormal_pretuned":   # also writes the two pretrained-network files
        return all(raw_path(m, n, repeat).exists() for m in (model,) + PRETUNED_PRETRAINED_FILES)
    if model in FINAL_SINGLE_OF:   # the ensemble job also writes the single-model file
        return raw_path(model, n, repeat).exists() and raw_path(FINAL_SINGLE_OF[model], n, repeat).exists()
    return raw_path(model, n, repeat).exists()


FINAL_SINGLE_OF = {"final_npe_ens12": "final_npe",
                   "final_bnn_pretrained_ens12": "final_bnn_pretrained"}


def _member_seed(seed, k):
    """Ensemble member k's seed; member 0 keeps the single model's seed (so it IS the single model).
    Offsets of 10,000 cannot collide with the sweep's own seeds (100 * size_index + repeat <= 505)."""
    return seed if k == 0 else seed + 10_000 * k


def _gpu_name(torch_only=False):
    try:
        if not torch_only:
            return str(jax.devices()[0].device_kind)
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception as e:   # recording only -- never fail a fit over this
        return f"unknown ({type(e).__name__})"


def _logsumexp0(a):
    m = np.max(a, axis=0)
    return m + np.log(np.sum(np.exp(a - m), axis=0))


def save_final_draws(bnn_raw, n, repeat):
    """Every kept draw's test-set output, original units: mu, s (n_good, kept, n_test, 2), corr
    (n_good, kept, n_test). Same transforms as bnn._summarize_correlated_raw (softplus + 1e-6 scale,
    tanh correlation). Large (~240 MB at 12 members), hence FINAL_DRAWS_DIR on $PSCRATCH."""
    raw = np.asarray(bnn_raw["draws_raw_test"], dtype=np.float64)
    ymu, ystd = np.asarray(bnn_raw["draws_ymu"]), np.asarray(bnn_raw["draws_ystd"])
    FINAL_DRAWS_DIR.mkdir(parents=True, exist_ok=True)
    path = FINAL_DRAWS_DIR / f"{MODEL}_n{n}_r{repeat}_draws.npz"
    save_raw_atomic(path, mu=(raw[..., 0:2] * ystd + ymu).astype(np.float32),
                    s=((np.logaddexp(0.0, raw[..., 2:4]) + 1e-6) * ystd).astype(np.float32),
                    corr=np.tanh(raw[..., 4]).astype(np.float32), n_train=n, repeat=repeat, model=MODEL)
    return path


def save_raw_atomic(path, **arrays):
    tmp_path = path.parent / (path.stem + ".tmp.npz")
    np.savez_compressed(tmp_path, **arrays)
    tmp_path.rename(path)


def fit_npe_with_retry(Xt_s, yt_s, Xv_s, yv_s, X_test, theta_test, seed, lr, wd,
                        steps, eval_every, patience, tarp_n_draws, n_posterior_samples,
                        tarp_seeds, tarp_lo, tarp_hi, max_attempts=3, retry_sleep_sec=10, **fit_kw):
    """2026-09-13: retry wrapper around npe.fit(), added after re-hitting the exact
    CUBLAS_STATUS_NOT_INITIALIZED failure first seen 2026-09-05/07
    (diagnosed then as transient shared-QOS GPU contention but never
    actually fixed at the code level; a retry wrapper is cheaper than blind resubmission). Retries the SAME (size, repeat)'s
    npe.fit() call in-process up to max_attempts times with a short sleep between attempts if it
    raises a RuntimeError mentioning CUBLAS/CUDA -- ASSUMPTION, not verified to actually help: a
    same-process retry can't move to a different physical node the way a fresh job submission
    could, so this only helps if the contention is a short-lived burst on the SAME node, not a
    persistently bad node. Any other exception (or exhausting max_attempts) propagates normally."""
    import time
    for attempt in range(1, max_attempts + 1):
        try:
            return npe.fit(
                Xt_s, yt_s, Xv_s, yv_s, X_test, theta_test, seed, lr, wd,
                steps, eval_every, patience, tarp_n_draws, n_posterior_samples, tarp_seeds,
                tarp_lo, tarp_hi, **fit_kw,
            )
        except RuntimeError as e:
            is_cuda_err = "CUBLAS" in str(e) or "CUDA" in str(e)
            if not is_cuda_err or attempt == max_attempts:
                raise
            print(f"[warn] npe.fit() attempt {attempt}/{max_attempts} hit a CUDA/CUBLAS error "
                  f"(known transient shared-QOS GPU contention) -- "
                  f"retrying in {retry_sleep_sec}s: {e}")
            time.sleep(retry_sleep_sec)




def fit_bnn_variant(Xb, yb, X_test, theta_test, seed, lr, wd, temperature, tarp_lo, tarp_hi,
                    prior_family="standardnormal", prior_kwargs=None, n_thinning=BNN_N_THINNING,
                    n_samples=BNN_N_SAMPLES, n_kept_samples=None,
                    hidden_layers=None, activation=BNN_ACTIVATION, pretrain_kfold=None, score_pretrained=False):
    """One bnn.fit_correlated() call -- shared by bnn_vanilla/bnn_temp (default standard N(0,1)
    prior, differing only in (lr, wd, temperature)), bnn_studentt (same call, prior_family=
    "student_t" with prior_kwargs={"loc", "scale", "nu"}) and bnn_vanilla_extrasteps (same call
    as bnn_vanilla, n_samples RAW chain steps with exactly n_kept_samples kept). n_samples here
    is the RAW step count -- see bnn.fit_correlated's STEP-COUNT CONVENTION. hidden_layers
    (None = BNN_HIDDEN_LAYERS) and activation: only bnn_elu_h8 sets them. pretrain_kfold: only
    bnn_vanilla_kfold sets it."""
    return bnn.fit_correlated(
        Xb, yb, X_test, theta_test, seed,
        BNN_N_MEMBERS, BNN_HIDDEN_LAYERS if hidden_layers is None else hidden_layers,
        BNN_EPOCHS, BNN_PATIENCE, BNN_VALIDATION_SPLIT,
        n_samples, BNN_WARMUP_STEPS, lr, wd, n_thinning,
        BNN_DESIRED_ENERGY_VAR_START, BNN_DESIRED_ENERGY_VAR_END,
        BNN_MAX_SEED_TRIES, BNN_RHAT_THRESHOLD,
        tarp_lo, tarp_hi, TARP_N_DRAWS, TARP_SEEDS, quiet=True,
        min_good_members=BNN_MIN_GOOD_MEMBERS,
        pretrain_eval_every=BNN_PRETRAIN_EVAL_EVERY, pretrain_min_steps=BNN_PRETRAIN_MIN_STEPS,
        prior_family=prior_family, prior_kwargs=prior_kwargs, temperature=temperature,
        n_kept_samples=n_kept_samples, activation=activation, pretrain_kfold=pretrain_kfold,
        score_pretrained=score_pretrained,
    )


def fit_gibbs_variant(Xb, yb, X_test, theta_test, seed, lr, wd, tarp_lo, tarp_hi):
    return bnn.fit_gibbs_hierarchical(
        Xb, yb, X_test, theta_test, seed,
        BNN_N_MEMBERS, BNN_HIDDEN_LAYERS, BNN_EPOCHS, BNN_PATIENCE, BNN_VALIDATION_SPLIT,
        BNN_N_SAMPLES, BNN_WARMUP_STEPS, lr, wd, BNN_N_THINNING,
        BNN_DESIRED_ENERGY_VAR_START, BNN_DESIRED_ENERGY_VAR_END,
        BNN_MAX_SEED_TRIES, BNN_RHAT_THRESHOLD,
        tarp_lo, tarp_hi, TARP_N_DRAWS, TARP_SEEDS,
        min_good_members=BNN_MIN_GOOD_MEMBERS,
        pretrain_eval_every=BNN_PRETRAIN_EVAL_EVERY, pretrain_min_steps=BNN_PRETRAIN_MIN_STEPS,
        ig_a=GIBBS_IG_A, ig_b=GIBBS_IG_B, tau_init=GIBBS_TAU_INIT,
        n_burnin_sweeps=GIBBS_N_BURNIN_SWEEPS, burnin_warmup_steps=GIBBS_BURNIN_WARMUP_STEPS,
        burnin_n_samples=GIBBS_BURNIN_N_SAMPLES, burnin_n_thinning=GIBBS_BURNIN_N_THINNING,
    )


def fit_rasmussen_retune_variant(Xb, yb, X_test, theta_test, seed, lr, wd, tarp_lo, tarp_hi,
                                 retune_every_samples=RASMUSSEN_RETUNE_EVERY_SAMPLES,
                                 n_thinning=BNN_N_THINNING, n_samples=BNN_N_SAMPLES, resample_tau=True,
                                 eb_global_scale=False, eb_per_step=False, eb_block=None):
    """Defaults are rasmussen_gibbs_retune's own settings. retune_every_samples=None switches
    the periodic retune off (the rasmussen_gibbs_noretune* models); n_thinning sets the raw chain
    steps per kept sample. n_samples here is the KEPT count (n_samples * n_thinning raw steps) --
    the opposite of fit_bnn_variant's."""
    return bnn.fit_rasmussen_gibbs_hierarchical(
        Xb, yb, X_test, theta_test, seed,
        BNN_N_MEMBERS, BNN_HIDDEN_LAYERS, BNN_EPOCHS, BNN_PATIENCE, BNN_VALIDATION_SPLIT,
        n_samples, BNN_WARMUP_STEPS, lr, wd, n_thinning,
        BNN_DESIRED_ENERGY_VAR_START, BNN_DESIRED_ENERGY_VAR_END,
        BNN_MAX_SEED_TRIES, BNN_RHAT_THRESHOLD,
        tarp_lo, tarp_hi, TARP_N_DRAWS, TARP_SEEDS,
        min_good_members=BNN_MIN_GOOD_MEMBERS,
        pretrain_eval_every=BNN_PRETRAIN_EVAL_EVERY, pretrain_min_steps=BNN_PRETRAIN_MIN_STEPS,
        ig_a=GIBBS_IG_A, ig_b=GIBBS_IG_B, tau_init=GIBBS_TAU_INIT,
        tau_save_stride=RASMUSSEN_TAU_SAVE_STRIDE,
        retune_every_samples=retune_every_samples,
        retune_tune1_steps=RASMUSSEN_RETUNE_TUNE1_STEPS, retune_tune2_steps=RASMUSSEN_RETUNE_TUNE2_STEPS,
        retune_tune3_steps=RASMUSSEN_RETUNE_TUNE3_STEPS, resample_tau=resample_tau,
        eb_global_scale=eb_global_scale, eb_adapt_segments=EB_ADAPT_SEGMENTS,
        eb_per_step=eb_per_step, tau_init_from_pretrained=(eb_per_step or eb_block is not None),
        eb_block=eb_block,
    )


def _bnn_variant_arrays(bnn_raw, bnn_f, diag, lr, wd, temperature, X_test, theta_test, Xb, yb):
    """Flattens one bnn.fit_correlated()/fit_gibbs_hierarchical() result into save_raw_atomic
    kwargs -- shared by bnn_vanilla/bnn_temp/gibbs (no prefix needed now, one model per file).

    2026-09-13: expanded to save EVERYTHING scripts/bde_correlated_perchain_test.py's own
    per-chain analysis notebooks consume, not just the pooled summary fields -- per direct request
    ("make sure you are saving enough info as is needed for tarp, 1st and 2nd moment, variance
    between runs so we can error bound, basically everything saved already for a bde_perchain
    analysis nb and then some"). The per-CHAIN decomposition (chain_mean/chain_aleatoric/
    chain_within_spread/chain_corr, between_chain_std/within_chain_std) is what lets a later
    analysis put an error bar on any pooled metric from a SINGLE (n_train, repeat) fit already
    (chain-to-chain spread), on top of the N_REPEATS axis's own independent-subsample spread --
    two distinct, complementary sources of variance, per this project's standing per-chain-spread
    convention."""
    epoch, trainloss, valloss, stop_epoch = bnn.pad_pretrain_loss_history(bnn_raw["pretrain_loss_history"])
    out = {
        "lr": lr, "wd": wd, "temperature": temperature,
        "X_test": np.asarray(X_test).reshape(len(X_test), -1), "theta_test": np.asarray(theta_test),
        "X_train": np.asarray(Xb).reshape(len(Xb), -1), "theta_train": np.asarray(yb),
        "mean": bnn_raw["mean"], "aleatoric": bnn_raw["aleatoric"],
        "epistemic": bnn_raw["epistemic"], "corr": bnn_raw["corr"],
        "log_prob_true": bnn_raw["log_prob_true"], "sbc_rank": bnn_raw["sbc_rank"],
        "theta_samples": bnn_raw["theta_samples"], "f": bnn_f,
        # Per-chain decomposition -- error-bounding within a single (n_train, repeat) fit.
        "chain_mean": bnn_raw["chain_mean"], "chain_aleatoric": bnn_raw["chain_aleatoric"],
        "chain_within_spread": bnn_raw["chain_within_spread"], "chain_corr": bnn_raw["chain_corr"],
        "chain_theta_samples": bnn_raw["chain_theta_samples"],
        "between_chain_std": bnn_raw["between_chain_std"], "within_chain_std": bnn_raw["within_chain_std"],
        # Training-set fit-quality checks (is it fitting the data it was shown at all).
        "train_mean": bnn_raw["train_mean"], "train_corr": bnn_raw["train_corr"],
        "train_chain_mean": bnn_raw["train_chain_mean"], "train_chain_corr": bnn_raw["train_chain_corr"],
        "train_log_prob_true": bnn_raw["train_log_prob_true"],
        # Pre-MCLMC point estimate + pretraining loss curves.
        "pretrain_train_mean": bnn_raw["pretrain_train_mean"], "pretrain_train_corr": bnn_raw["pretrain_train_corr"],
        "pretrain_loss_epoch": epoch, "pretrain_loss_trainloss": trainloss,
        "pretrain_loss_valloss": valloss, "pretrain_loss_stop_epoch": stop_epoch,
        # Mixing diagnostics, pooled + per-test-point.
        "rhat_mean": diag["rhat_mean"], "rhat_max": diag["rhat_max"],
        "rhat_sigma_max": diag["rhat_sigma_max"], "rhat_rho_max": diag["rhat_rho_max"],
        "rhat_full": diag.get("rhat_full", np.zeros((0, 0))), "ess_full": diag.get("ess_full", np.zeros((0, 0))),
        "ess_mean": diag["ess_mean"], "ess_min": diag["ess_min"],
        "n_members_used": diag["n_members_used"], "mixing_ok": diag["mixing_ok"],
        "seeds_used": np.asarray(diag["seeds_used"]),
        "member_fail_rate_across_attempts": diag["member_fail_rate_across_attempts"],
        "seed_attempts": diag["seed_attempts"],
        "members_tried_across_attempts": diag["members_tried_across_attempts"],
        "members_failed_across_attempts": diag["members_failed_across_attempts"],
    }
    if "gibbs_tau_final" in diag:
        out["gibbs_tau_final"] = diag["gibbs_tau_final"]
        out["gibbs_tau_history"] = diag["gibbs_tau_history"]
        out["gibbs_ig_a"] = diag["gibbs_ig_a"]
        out["gibbs_ig_b"] = diag["gibbs_ig_b"]
    if "rasmussen_tau_trace" in diag:
        out["rasmussen_tau_trace"] = diag["rasmussen_tau_trace"]
        out["rasmussen_tau_save_stride"] = diag["rasmussen_tau_save_stride"]
        out["rasmussen_ig_a"] = diag["rasmussen_ig_a"]
        out["rasmussen_ig_b"] = diag["rasmussen_ig_b"]
        out["rasmussen_retune_every_samples"] = diag["rasmussen_retune_every_samples"]
        out["rasmussen_retune_tune1_steps"] = diag["rasmussen_retune_tune1_steps"]
        out["rasmussen_retune_tune2_steps"] = diag["rasmussen_retune_tune2_steps"]
        out["rasmussen_retune_tune3_steps"] = diag["rasmussen_retune_tune3_steps"]
        out["rasmussen_L_history"] = diag["rasmussen_L_history"]
        out["rasmussen_step_size_history"] = diag["rasmussen_step_size_history"]
    for _k in ("rasmussen_eb_adapt_segments", "rasmussen_eb_tau_history", "rasmussen_eb_L_history",
               "rasmussen_eb_step_size_history", "rasmussen_eb_tau_init",
               "rasmussen_ebblock_mode", "rasmussen_ebblock_tau_history", "rasmussen_ebblock_gamma_history",
               "rasmussen_ebblock_tau_target_history", "rasmussen_ebblock_n_gamma_clamped",
               "rasmussen_ebblock_n_step_capped", "rasmussen_ebblock_group_sizes",
               "rasmussen_ebblock_damping", "rasmussen_ebblock_max_factor"):
        if _k in diag:
            out[_k] = diag[_k]
    return out




def fit_npe_ensemble(Xt_s, yt_s, Xv_s, yv_s, X_test, theta_test, seed, lr, wd, tarp_lo, tarp_hi, k_members):
    """k_members NPE flows, same split, seeds _member_seed(seed, k). Returns (single_raw, single_f,
    ens_raw, ens_f); member 0 is exactly the single NPE. Ensemble = equal-weight mixture of the flows:
    log density by log-mean-exp, TARP/SBC from an equal share of each flow's own draws, mean/std from
    the mixture's first two moments."""
    members, single_f = [], None
    for k in range(k_members):
        raw_k, f_k = fit_npe_with_retry(
            Xt_s, yt_s, Xv_s, yv_s, X_test, theta_test, _member_seed(seed, k), lr, wd,
            NPE_STEPS, NPE_EVAL_EVERY, NPE_PATIENCE, TARP_N_DRAWS, NPE_N_POSTERIOR_SAMPLES, TARP_SEEDS, tarp_lo, tarp_hi)
        if k == 0:
            single_f = f_k
        members.append(raw_k)
    lp_k = np.stack([m["log_prob_true"] for m in members]).astype(np.float64)
    per = -(-TARP_N_DRAWS // k_members)
    samples = np.concatenate([np.asarray(m["theta_samples"])[:per] for m in members], 0)[:TARP_N_DRAWS]
    theta_np = np.asarray(theta_test)
    m_mean, m_std = np.stack([m["mean"] for m in members]), np.stack([m["std"] for m in members])
    ens_raw = dict(mean=m_mean.mean(0), std=np.sqrt((m_std ** 2).mean(0) + m_mean.var(0)),
                   log_prob_true=_logsumexp0(lp_k) - np.log(k_members),
                   sbc_rank=metrics.sbc_marginal_rank(samples, theta_np),
                   member_mean=m_mean, member_std=m_std, member_log_prob_true=lp_k)
    ens_f = metrics.compute_tarp_f_stats(samples, theta_np, tarp_lo, tarp_hi, TARP_SEEDS)
    return members[0], single_f, ens_raw, ens_f


def main():
    import time
    t0 = time.time()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL in FINAL_MODELS:   # 2026-10-29 fix: the summary folder was never created (first two final jobs failed at save)
        FINAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Model: {MODEL}  Sizes: {SIZES}  N_REPEATS: {N_REPEATS}")
    if MODEL in MODEL_PURPOSE:
        print(MODEL_PURPOSE[MODEL])
    n_chain_steps, n_kept = chain_budget(MODEL)
    if n_chain_steps is not None:
        print(f"Chain budget per member (post-warmup): {n_chain_steps} raw steps, {n_kept} kept samples")

    X_train, theta_train, X_test, theta_test = data.load_dataset()
    tarp_lo, tarp_hi = data.TARP_LO, data.TARP_HI

    npe_hp = (_load_sized_hp_npz(RESULTS_DIR / "npe_params.npz", "NPE")
              if MODEL in ("npe", "final_npe_ens12") + tuple(NPE_HEAD_MODELS) else {})

    n_files_written = 0
    for n in tqdm(SIZES, desc=f"{MODEL} sweep"):
        run = ALL_SIZES.index(n)  # NOT the position in SIZES -- see SEED CORRECTION in the docstring
        if MODEL in ("npe", "final_npe_ens12") + tuple(NPE_HEAD_MODELS):
            npe_lr, npe_wd = get_hp(npe_hp, n, DEFAULT_NPE_LR, DEFAULT_NPE_WD)
            HP_FILE_BY_SIZE[n] = "npe_params.npz" if n in npe_hp else "default"
        elif MODEL in ("bnn_vanilla", "bnn_temp", "gibbs", "bnn_studentt",
                       "bnn_vanilla_extrasteps", "bnn_wideprior", "bnn_temp_opt",
                       "bnn_studentt_extrasteps", "bnn_gaussprior", "bnn_elu_h8", "bnn_vanilla_kfold") + RASMUSSEN_MODELS + FINAL_BNN_MODELS \
                       + FINAL_PRETRAIN_MODELS \
                or MANUAL_TEMPERATURE is not None:
            bnn_tuned_lr, bnn_tuned_wd = _load_bnn_hp(n)
        if MANUAL_TEMPERATURE is not None:
            searched_temperature = MANUAL_TEMPERATURE
            print(f"{MODEL}: n={n} manual temperature T={searched_temperature}")
        if MODEL == "bnn_temp_opt":
            searched_temperature = _load_searched_temperature(n)
            print(f"bnn_temp_opt: n={n} searched temperature T={searched_temperature}")

        for repeat in REPEATS:
            if repeat_done(MODEL, n, repeat):
                print(f"Skipping model={MODEL} n={n}, run={repeat+1}/{N_REPEATS} (already saved)")
                continue

            seed = 100 * run + repeat
            Xb, yb = data.subsample_train(X_train, theta_train, n, repeat)
            t_fit = time.time()

            if MODEL in FINAL_PRETRAIN_MODELS:
                ens_raw, ens_f, single_raw, single_f, pdiag = bnn.fit_pretrained_ensemble(
                    Xb, yb, X_test, theta_test, seed, BNN_N_MEMBERS, BNN_HIDDEN_LAYERS, BNN_EPOCHS, BNN_PATIENCE,
                    BNN_VALIDATION_SPLIT, bnn_tuned_lr, bnn_tuned_wd, tarp_lo, tarp_hi, TARP_N_DRAWS, TARP_SEEDS,
                    quiet=True, pretrain_eval_every=BNN_PRETRAIN_EVAL_EVERY, pretrain_min_steps=BNN_PRETRAIN_MIN_STEPS,
                    activation=BNN_ACTIVATION, warmup_steps=BNN_WARMUP_STEPS, n_samples=BNN_N_SAMPLES,
                    n_thinning=BNN_N_THINNING, desired_energy_var_start=BNN_DESIRED_ENERGY_VAR_START,
                    desired_energy_var_end=BNN_DESIRED_ENERGY_VAR_END)
                common = dict(n_train=n, repeat=repeat, seed=seed, lr=bnn_tuned_lr, wd=bnn_tuned_wd,
                              X_test=np.asarray(X_test).reshape(len(X_test), -1), theta_test=np.asarray(theta_test),
                              X_train=np.asarray(Xb).reshape(len(Xb), -1), theta_train=np.asarray(yb),
                              hp_file=HP_FILE_BY_SIZE.get(n, ""), gpu_name=_gpu_name(),
                              fit_sec=time.time() - t_fit, n_members_total=BNN_N_MEMBERS,
                              member_finite=pdiag["member_finite"], pretrain_stop_epoch=pdiag["pretrain_stop_epoch"])
                save_raw_atomic(raw_path(FINAL_SINGLE_OF[MODEL], n, repeat), **common, **single_raw, f=single_f,
                                n_ensemble=1)
                save_raw_atomic(raw_path(MODEL, n, repeat), **common, **ens_raw, f=ens_f,
                                n_ensemble=pdiag["n_members_used"], pretrain_weights=pdiag["pretrain_weights"])
                n_files_written += 2
                print(f"Saved {MODEL} and {FINAL_SINGLE_OF[MODEL]} n={n} r={repeat} ({time.time() - t_fit:.0f} s; "
                      f"{pdiag['n_members_used']}/{BNN_N_MEMBERS} members finite)")
                continue

            if MODEL in FINAL_ENSEMBLE_MODELS:
                Xt_s, yt_s, Xv_s, yv_s = metrics.split(np.asarray(Xb), np.asarray(yb), seed)
                if MODEL == "final_npe_ens12":
                    lr_, wd_ = npe_lr, npe_wd
                    single_raw, single_f, ens_raw, ens_f = fit_npe_ensemble(
                        Xt_s, yt_s, Xv_s, yv_s, X_test, theta_test, seed, lr_, wd_, tarp_lo, tarp_hi, FINAL_N_MEMBERS)
                    single_keys = dict(mean=single_raw["mean"], std=single_raw["std"])
                    gpu = _gpu_name(torch_only=True)
                common = dict(n_train=n, repeat=repeat, seed=seed, lr=lr_, wd=wd_,
                              X_test=np.asarray(X_test).reshape(len(X_test), -1), theta_test=np.asarray(theta_test),
                              X_train=np.asarray(Xb).reshape(len(Xb), -1), theta_train=np.asarray(yb),
                              hp_file=HP_FILE_BY_SIZE.get(n, ""), gpu_name=gpu,
                              fit_sec_ensemble_total=time.time() - t_fit)
                save_raw_atomic(raw_path(FINAL_SINGLE_OF[MODEL], n, repeat), **common, **single_keys, f=single_f,
                                log_prob_true=single_raw["log_prob_true"], sbc_rank=single_raw["sbc_rank"])
                save_raw_atomic(raw_path(MODEL, n, repeat), **common, **ens_raw, f=ens_f,
                                n_ensemble=FINAL_N_MEMBERS,
                                member_seeds=np.array([_member_seed(seed, k) for k in range(FINAL_N_MEMBERS)]))
                n_files_written += 2
                print(f"Saved {MODEL} and {FINAL_SINGLE_OF[MODEL]} n={n} r={repeat} ({time.time() - t_fit:.0f} s)")
                continue

            if MODEL in FINAL_BNN_MODELS:
                final_T = 1.0
                if BASE_MODEL == "final_bnn_temp":
                    final_T = (FINAL_TEMPERATURES[n] if MODEL in ("final_bnn_temp", "final_bnn_temp_oldT_pretuned")
                              else _load_pretuned_temperature(n))
                    print(f"{MODEL} n={n}: temperature {final_T}")
                if BASE_MODEL in ("final_bnn_stdnormal", "final_bnn_temp", "final_bnn_gaussprior"):
                    prior_kw = (dict(prior_family="normal", prior_kwargs={"loc": 0.0, "scale": GAUSSPRIOR_SCALE})
                                if BASE_MODEL == "final_bnn_gaussprior" else {})
                    bnn_raw, bnn_f, diag = fit_bnn_variant(
                        Xb, yb, X_test, theta_test, seed, bnn_tuned_lr, bnn_tuned_wd, final_T, tarp_lo, tarp_hi,
                        score_pretrained=(MODEL == "final_bnn_stdnormal_pretuned"), **prior_kw)
                else:   # retune loop: N(0,1) fixed (= bnn_vanilla_retune_10k) or block EB (= bnn_ebblock_iso_10k)
                    bnn_raw, bnn_f, diag = fit_rasmussen_retune_variant(
                        Xb, yb, X_test, theta_test, seed, bnn_tuned_lr, bnn_tuned_wd, tarp_lo, tarp_hi,
                        retune_every_samples=RASMUSSEN_RETUNE_EVERY_SAMPLES, n_samples=RASMUSSEN_10K_N_KEPT,
                        resample_tau=False, eb_block=("iso" if BASE_MODEL == "final_bnn_ebblock" else None))
                fit_sec = time.time() - t_fit
                pre = diag.pop("pretrained", None)
                if MODEL == "final_bnn_stdnormal_pretuned":   # the BNN without sampling, from this same run
                    if pre is None:
                        print(f"[warn] n={n} r={repeat}: member 0's pretrained output is non-finite -- no pretrained files")
                    else:
                        hist = bnn_raw["pretrain_loss_history"]
                        common = dict(n_train=n, repeat=repeat, seed=seed, lr=bnn_tuned_lr, wd=bnn_tuned_wd,
                                      X_test=np.asarray(X_test).reshape(len(X_test), -1), theta_test=np.asarray(theta_test),
                                      X_train=np.asarray(Xb).reshape(len(Xb), -1), theta_train=np.asarray(yb),
                                      hp_file=HP_FILE_BY_SIZE.get(n, ""), gpu_name=_gpu_name(), source_model=MODEL,
                                      fit_sec=fit_sec, n_members_total=BNN_N_MEMBERS,   # = the BNN run's fit time (pretraining + MCLMC)
                                      member_finite=pre["member_finite"],
                                      pretrain_stop_epoch=np.array([h["stop_epoch"] for h in hist]),
                                      seeds_used=np.asarray(diag["seeds_used"]))
                        save_raw_atomic(raw_path(PRETUNED_PRETRAINED_FILES[1], n, repeat), **common, **pre["single_raw"],
                                        f=pre["single_f"], n_ensemble=1)
                        save_raw_atomic(raw_path(PRETUNED_PRETRAINED_FILES[0], n, repeat), **common, **pre["ens_raw"],
                                        f=pre["ens_f"], n_ensemble=pre["n_members_used"],
                                        pretrain_weights=np.asarray(bnn_raw["pretrain_weights"])[pre["member_finite"]])
                        n_files_written += 2
                draws_path = save_final_draws(bnn_raw, n, repeat)
                save_kwargs = dict(n_train=n, repeat=repeat, seed=seed, n_members=BNN_N_MEMBERS,
                                   n_chain_steps=n_chain_steps, n_kept_samples=n_kept, fit_sec=fit_sec,
                                   gpu_name=_gpu_name(), draws_file=str(draws_path),
                                   prior_normal_scale=(GAUSSPRIOR_SCALE if BASE_MODEL == "final_bnn_gaussprior" else 1.0),
                                   retune=(BASE_MODEL in ("final_bnn_stdnormal_retune", "final_bnn_ebblock")))
                save_kwargs.update(_bnn_variant_arrays(
                    bnn_raw, bnn_f, diag, bnn_tuned_lr, bnn_tuned_wd, final_T, X_test, theta_test, Xb, yb))

            elif MODEL == "npe":
                Xt_s, yt_s, Xv_s, yv_s = metrics.split(np.asarray(Xb), np.asarray(yb), seed)
                npe_raw, npe_f = fit_npe_with_retry(
                    Xt_s, yt_s, Xv_s, yv_s, X_test, theta_test, seed, npe_lr, npe_wd,
                    NPE_STEPS, NPE_EVAL_EVERY, NPE_PATIENCE,
                    TARP_N_DRAWS, NPE_N_POSTERIOR_SAMPLES, TARP_SEEDS, tarp_lo, tarp_hi,
                )
                save_kwargs = dict(
                    n_train=n, repeat=repeat, seed=seed, lr=npe_lr, wd=npe_wd,
                    X_test=np.asarray(X_test).reshape(len(X_test), -1), theta_test=np.asarray(theta_test),
                    X_train=np.asarray(Xb).reshape(len(Xb), -1), theta_train=np.asarray(yb),
                    mean=npe_raw["mean"], std=npe_raw["std"], f=npe_f,
                    log_prob_true=npe_raw["log_prob_true"], sbc_rank=npe_raw["sbc_rank"],
                )

            elif MODEL in NPE_HEAD_MODELS:
                Xt_s, yt_s, Xv_s, yv_s = metrics.split(np.asarray(Xb), np.asarray(yb), seed)
                npe_raw, npe_f = fit_npe_with_retry(
                    Xt_s, yt_s, Xv_s, yv_s, X_test, theta_test, seed, npe_lr, npe_wd,
                    NPE_STEPS, NPE_EVAL_EVERY, NPE_PATIENCE,
                    TARP_N_DRAWS, NPE_N_POSTERIOR_SAMPLES, TARP_SEEDS, tarp_lo, tarp_hi,
                    head="gaussian", gaussian_hidden=NPE_HEAD_MODELS[MODEL],
                )
                save_kwargs = dict(
                    n_train=n, repeat=repeat, seed=seed, lr=npe_lr, wd=npe_wd,
                    npe_head="gaussian", npe_gaussian_hidden=NPE_HEAD_MODELS[MODEL],
                    X_test=np.asarray(X_test).reshape(len(X_test), -1), theta_test=np.asarray(theta_test),
                    X_train=np.asarray(Xb).reshape(len(Xb), -1), theta_train=np.asarray(yb),
                    mean=npe_raw["mean"], std=npe_raw["std"], f=npe_f,
                    log_prob_true=npe_raw["log_prob_true"], sbc_rank=npe_raw["sbc_rank"],
                )

            elif MODEL == "bnn_vanilla":
                bnn_raw, bnn_f, diag = fit_bnn_variant(
                    Xb, yb, X_test, theta_test, seed, bnn_tuned_lr, bnn_tuned_wd, 1.0,
                    tarp_lo, tarp_hi,
                )
                save_kwargs = dict(n_train=n, repeat=repeat, seed=seed)
                save_kwargs.update(_bnn_variant_arrays(
                    bnn_raw, bnn_f, diag, bnn_tuned_lr, bnn_tuned_wd, 1.0, X_test, theta_test, Xb, yb
                ))

            elif MODEL == "bnn_elu_h8":
                bnn_raw, bnn_f, diag = fit_bnn_variant(
                    Xb, yb, X_test, theta_test, seed, bnn_tuned_lr, bnn_tuned_wd, 1.0,
                    tarp_lo, tarp_hi,
                    hidden_layers=SMALL_BNN_HIDDEN_LAYERS, activation=SMALL_BNN_ACTIVATION,
                )
                save_kwargs = dict(n_train=n, repeat=repeat, seed=seed,
                                   hidden_layers=np.asarray(SMALL_BNN_HIDDEN_LAYERS),
                                   activation=SMALL_BNN_ACTIVATION)
                save_kwargs.update(_bnn_variant_arrays(
                    bnn_raw, bnn_f, diag, bnn_tuned_lr, bnn_tuned_wd, 1.0, X_test, theta_test, Xb, yb
                ))

            elif MODEL == "bnn_vanilla_kfold":
                bnn_raw, bnn_f, diag = fit_bnn_variant(
                    Xb, yb, X_test, theta_test, seed, bnn_tuned_lr, bnn_tuned_wd, 1.0,
                    tarp_lo, tarp_hi, pretrain_kfold=PRETRAIN_KFOLD_K,
                )
                save_kwargs = dict(n_train=n, repeat=repeat, seed=seed, pretrain_kfold=PRETRAIN_KFOLD_K,
                                   pretrain_fold_val_idx=np.stack(bnn.kfold_val_indices(n, PRETRAIN_KFOLD_K)[:BNN_N_MEMBERS]))
                save_kwargs.update(_bnn_variant_arrays(
                    bnn_raw, bnn_f, diag, bnn_tuned_lr, bnn_tuned_wd, 1.0, X_test, theta_test, Xb, yb
                ))

            elif MODEL == "bnn_temp":
                bnn_raw, bnn_f, diag = fit_bnn_variant(
                    Xb, yb, X_test, theta_test, seed, bnn_tuned_lr, bnn_tuned_wd,
                    BNN_TEMP_TEMPERATURE, tarp_lo, tarp_hi,
                )
                save_kwargs = dict(n_train=n, repeat=repeat, seed=seed)
                save_kwargs.update(_bnn_variant_arrays(
                    bnn_raw, bnn_f, diag, bnn_tuned_lr, bnn_tuned_wd, BNN_TEMP_TEMPERATURE,
                    X_test, theta_test, Xb, yb
                ))

            elif MODEL == "gibbs":
                bnn_raw, bnn_f, diag = fit_gibbs_variant(
                    Xb, yb, X_test, theta_test, seed, bnn_tuned_lr, bnn_tuned_wd, tarp_lo, tarp_hi,
                )
                save_kwargs = dict(n_train=n, repeat=repeat, seed=seed)
                save_kwargs.update(_bnn_variant_arrays(
                    bnn_raw, bnn_f, diag, bnn_tuned_lr, bnn_tuned_wd, 1.0, X_test, theta_test, Xb, yb
                ))

            elif MODEL == "rasmussen_gibbs_retune":
                bnn_raw, bnn_f, diag = fit_rasmussen_retune_variant(
                    Xb, yb, X_test, theta_test, seed, bnn_tuned_lr, bnn_tuned_wd, tarp_lo, tarp_hi,
                )
                save_kwargs = dict(n_train=n, repeat=repeat, seed=seed)
                save_kwargs.update(_bnn_variant_arrays(
                    bnn_raw, bnn_f, diag, bnn_tuned_lr, bnn_tuned_wd, 1.0, X_test, theta_test, Xb, yb
                ))

            elif MODEL in RASMUSSEN_10K_MODELS:
                bnn_raw, bnn_f, diag = fit_rasmussen_retune_variant(
                    Xb, yb, X_test, theta_test, seed, bnn_tuned_lr, bnn_tuned_wd, tarp_lo, tarp_hi,
                    retune_every_samples=(RASMUSSEN_RETUNE_EVERY_SAMPLES
                                          if MODEL in RASMUSSEN_RETUNE_MODELS else None),
                    n_samples=RASMUSSEN_10K_N_KEPT, resample_tau=(MODEL not in FIXED_TAU_MODELS),
                    eb_global_scale=(MODEL in EB_MODELS), eb_per_step=(MODEL in EB_STEP_MODELS),
                    eb_block=EB_BLOCK_MODELS.get(MODEL),
                )
                save_kwargs = dict(n_train=n, repeat=repeat, seed=seed, n_thinning=BNN_N_THINNING,
                                   resample_tau=(MODEL not in FIXED_TAU_MODELS),
                                   n_chain_steps=n_chain_steps, n_kept_samples=n_kept)
                save_kwargs.update(_bnn_variant_arrays(
                    bnn_raw, bnn_f, diag, bnn_tuned_lr, bnn_tuned_wd, 1.0, X_test, theta_test, Xb, yb
                ))

            elif MODEL == "bnn_vanilla_extrasteps":
                bnn_raw, bnn_f, diag = fit_bnn_variant(
                    Xb, yb, X_test, theta_test, seed, bnn_tuned_lr, bnn_tuned_wd, 1.0,
                    tarp_lo, tarp_hi, n_samples=BNN_EXTRASTEPS_N_STEPS,
                    n_kept_samples=BNN_EXTRASTEPS_N_KEPT,
                )
                save_kwargs = dict(n_train=n, repeat=repeat, seed=seed,
                                   n_chain_steps=n_chain_steps, n_kept_samples=n_kept)
                save_kwargs.update(_bnn_variant_arrays(
                    bnn_raw, bnn_f, diag, bnn_tuned_lr, bnn_tuned_wd, 1.0, X_test, theta_test, Xb, yb
                ))

            elif MODEL in ("rasmussen_gibbs_noretune", "rasmussen_gibbs_noretune_extrasteps"):
                n_thinning = (RASMUSSEN_EXTRASTEPS_N_THINNING
                              if MODEL == "rasmussen_gibbs_noretune_extrasteps" else BNN_N_THINNING)
                bnn_raw, bnn_f, diag = fit_rasmussen_retune_variant(
                    Xb, yb, X_test, theta_test, seed, bnn_tuned_lr, bnn_tuned_wd, tarp_lo, tarp_hi,
                    retune_every_samples=None, n_thinning=n_thinning,
                )
                save_kwargs = dict(n_train=n, repeat=repeat, seed=seed, n_thinning=n_thinning)
                save_kwargs.update(_bnn_variant_arrays(
                    bnn_raw, bnn_f, diag, bnn_tuned_lr, bnn_tuned_wd, 1.0, X_test, theta_test, Xb, yb
                ))

            elif MODEL == "bnn_temp_opt" or MANUAL_TEMPERATURE is not None:
                bnn_raw, bnn_f, diag = fit_bnn_variant(
                    Xb, yb, X_test, theta_test, seed, bnn_tuned_lr, bnn_tuned_wd, searched_temperature,
                    tarp_lo, tarp_hi,
                )
                save_kwargs = dict(n_train=n, repeat=repeat, seed=seed)
                save_kwargs.update(_bnn_variant_arrays(
                    bnn_raw, bnn_f, diag, bnn_tuned_lr, bnn_tuned_wd, searched_temperature,
                    X_test, theta_test, Xb, yb
                ))

            elif MODEL == "bnn_studentt_extrasteps":
                bnn_raw, bnn_f, diag = fit_bnn_variant(
                    Xb, yb, X_test, theta_test, seed, bnn_tuned_lr, bnn_tuned_wd, 1.0,
                    tarp_lo, tarp_hi, prior_family="student_t",
                    prior_kwargs={"loc": 0.0, "scale": STUDENTT_SCALE, "nu": STUDENTT_NU},
                    n_samples=BNN_EXTRASTEPS_N_STEPS, n_kept_samples=BNN_EXTRASTEPS_N_KEPT,
                )
                save_kwargs = dict(n_train=n, repeat=repeat, seed=seed,
                                   studentt_nu=STUDENTT_NU, studentt_scale=STUDENTT_SCALE,
                                   n_chain_steps=n_chain_steps, n_kept_samples=n_kept)
                save_kwargs.update(_bnn_variant_arrays(
                    bnn_raw, bnn_f, diag, bnn_tuned_lr, bnn_tuned_wd, 1.0, X_test, theta_test, Xb, yb
                ))

            elif MODEL in ("bnn_wideprior", "bnn_gaussprior"):
                _prior_scale = WIDEPRIOR_SCALE if MODEL == "bnn_wideprior" else GAUSSPRIOR_SCALE
                bnn_raw, bnn_f, diag = fit_bnn_variant(
                    Xb, yb, X_test, theta_test, seed, bnn_tuned_lr, bnn_tuned_wd, 1.0,
                    tarp_lo, tarp_hi, prior_family="normal",
                    prior_kwargs={"loc": 0.0, "scale": _prior_scale},
                )
                save_kwargs = dict(n_train=n, repeat=repeat, seed=seed,
                                   prior_normal_scale=_prior_scale)
                save_kwargs.update(_bnn_variant_arrays(
                    bnn_raw, bnn_f, diag, bnn_tuned_lr, bnn_tuned_wd, 1.0, X_test, theta_test, Xb, yb
                ))

            elif MODEL == "bnn_studentt":
                bnn_raw, bnn_f, diag = fit_bnn_variant(
                    Xb, yb, X_test, theta_test, seed, bnn_tuned_lr, bnn_tuned_wd, 1.0,
                    tarp_lo, tarp_hi, prior_family="student_t",
                    prior_kwargs={"loc": 0.0, "scale": STUDENTT_SCALE, "nu": STUDENTT_NU},
                )
                save_kwargs = dict(n_train=n, repeat=repeat, seed=seed,
                                   studentt_nu=STUDENTT_NU, studentt_scale=STUDENTT_SCALE)
                save_kwargs.update(_bnn_variant_arrays(
                    bnn_raw, bnn_f, diag, bnn_tuned_lr, bnn_tuned_wd, 1.0, X_test, theta_test, Xb, yb
                ))

            if n in HP_FILE_BY_SIZE:
                save_kwargs["hp_file"] = HP_FILE_BY_SIZE[n]
            save_raw_atomic(raw_path(MODEL, n, repeat), **save_kwargs)
            n_files_written += 1
            print(f"Saved model={MODEL} n={n}, run={repeat+1}/{N_REPEATS} -> {raw_path(MODEL, n, repeat).name}")

    elapsed = time.time() - t0
    print(f"Raw results saved to: {RAW_DIR}")

    if OUT_OVERRIDE:
        print(f"RUN_EXPERIMENT_OUT_DIR set ({OUT_OVERRIDE}): no experiment-log entry")
        return
    experiment_log.log_run(
        script="scripts/fit/run_experiment.py",
        config=dict(
            model=MODEL, sizes=SIZES, n_repeats=N_REPEATS, repeats=REPEATS,
            final_n_members=(FINAL_N_MEMBERS if MODEL in FINAL_MODELS else None),
            final_temperatures=({str(n): FINAL_TEMPERATURES[n] for n in SIZES}
                                if MODEL in ("final_bnn_temp", "final_bnn_temp_oldT_pretuned")
                                else {str(n): _load_pretuned_temperature(n) for n in SIZES}
                                if MODEL == "final_bnn_temp_pretuned" else None),
            final_draws_dir=(str(FINAL_DRAWS_DIR) if MODEL in FINAL_BNN_MODELS else None),
            bnn_temp_temperature=BNN_TEMP_TEMPERATURE if MODEL == "bnn_temp" else None,
            gibbs_ig_a=GIBBS_IG_A if MODEL in ("gibbs",) + RASMUSSEN_MODELS else None,
            gibbs_ig_b=GIBBS_IG_B if MODEL in ("gibbs",) + RASMUSSEN_MODELS else None,
            n_thinning=((RASMUSSEN_EXTRASTEPS_N_THINNING
                         if MODEL == "rasmussen_gibbs_noretune_extrasteps" else BNN_N_THINNING)
                        if MODEL in RASMUSSEN_MODELS else None),
            n_chain_steps=n_chain_steps, n_kept_samples=n_kept,
            hidden_layers=(SMALL_BNN_HIDDEN_LAYERS if MODEL == "bnn_elu_h8" else None),
            activation=(SMALL_BNN_ACTIVATION if MODEL == "bnn_elu_h8" else None),
            pretrain_kfold=(PRETRAIN_KFOLD_K if MODEL == "bnn_vanilla_kfold" else None),
            npe_head=("gaussian" if MODEL in NPE_HEAD_MODELS else None),
            npe_gaussian_hidden=NPE_HEAD_MODELS.get(MODEL),
            retune=("off" if MODEL in RASMUSSEN_NORETUNE_MODELS else None),
            prior_normal_scale=(WIDEPRIOR_SCALE if MODEL == "bnn_wideprior"
                                else GAUSSPRIOR_SCALE if MODEL == "bnn_gaussprior" else None),
            manual_temperature=MANUAL_TEMPERATURE,
            resample_tau=((MODEL not in FIXED_TAU_MODELS) if MODEL in RASMUSSEN_MODELS else None),
            eb_global_scale=(True if MODEL in EB_MODELS else None),
            eb_per_step=(True if MODEL in EB_STEP_MODELS else None),
            eb_block=EB_BLOCK_MODELS.get(MODEL),
            eb_adapt_segments=(EB_ADAPT_SEGMENTS if MODEL in EB_MODELS else None),
            hp_files={str(k): v for k, v in HP_FILE_BY_SIZE.items()} or None,
            searched_temperatures=({str(n): _load_searched_temperature(n) for n in SIZES}
                                   if MODEL == "bnn_temp_opt" else None),
            studentt_nu=STUDENTT_NU if MODEL in ("bnn_studentt", "bnn_studentt_extrasteps") else None,
            studentt_scale=STUDENTT_SCALE if MODEL in ("bnn_studentt", "bnn_studentt_extrasteps") else None,
            rasmussen_retune_every_samples=RASMUSSEN_RETUNE_EVERY_SAMPLES if MODEL in RASMUSSEN_RETUNE_MODELS else None,
            rasmussen_retune_tune1_steps=RASMUSSEN_RETUNE_TUNE1_STEPS if MODEL in RASMUSSEN_RETUNE_MODELS else None,
            rasmussen_retune_tune2_steps=RASMUSSEN_RETUNE_TUNE2_STEPS if MODEL in RASMUSSEN_RETUNE_MODELS else None,
        ),
        results=dict(n_files_written=n_files_written, elapsed_sec=elapsed),
        job_id=os.environ.get("SLURM_JOB_ID"),
        dataset=DATASET,
        output_path=str(FINAL_DIR if MODEL in FINAL_MODELS else RAW_DIR),
        notes=f"Per-model sweep (2026-09-13 rewrite -- one model per invocation, not one dataset "
              f"size at a time) -- this run: model={MODEL!r}. See this script's own module "
              f"docstring for the model set (gaussian/npe/bnn_vanilla/bnn_temp/gibbs/"
              f"rasmussen_gibbs_retune, added 2026-09-17, and bnn_studentt, added 2026-10-01: a "
              f"fixed Student-t(nu=2a, scale=sqrt(b/a)) prior equal to the Gibbs models' marginal "
              f"weight prior, on bnn_vanilla's own code path) and every flagged assumption "
              f"(T=0.305 transfer, InverseGamma(2,1) hyperprior, N_REPEATS=5, and for "
              f"rasmussen_gibbs_retune specifically: retune cadence/budget validated only at "
              f"n=100/10,000 before this sweep's other four sizes ran)."
              + (f" {MODEL_PURPOSE[MODEL]}" if MODEL in MODEL_PURPOSE else ""),
    )


if __name__ == "__main__":
    main()
