"""Bayesian neural network posterior via bde/MILE (scikit-learn-contrib/bde, Sommer, Robnik, Nozadze,
Seljak & Rugamer, ICLR 2025). Two fitting paths, both usable:

  fit()            -- stock bde/MILE's public API (BdeRegressor), one regressor per parameter
                       (sigma, rho) fit independently, pooled with an explicit corr=0
                       independence assumption. Kept as the simple/original baseline.
  fit_correlated() -- CorrelatedBdeRegressor, a set of subclass overrides (defined below) that
                       fit a single joint bivariate posterior over (sigma, rho) with a real
                       fitted correlation, not an assumption.

Why fit() doesn't fit a joint posterior on its own: confirmed by inspecting bde's source
directly that (1) BdeRegressor declares
single_output=True/multi_output=False at the sklearn-tag level and its own docs require a (n,)
or (n,1) target, and (2) the sampling-phase likelihood
(bde.sampler.probabilistic.ProbabilisticModel.log_likelihood) hardcodes univariate Gaussian
slicing (lvals[...,0:1] mean, lvals[...,1:2] scale) regardless of what `loss=` object is passed
to BdeRegressor — a custom BaseLoss only ever reaches pretraining, never the MCLMC sampling
phase. A genuine joint/correlated fit requires subclassing ProbabilisticModel and bypassing
several other BdeRegressor internals too -- that's what fit_correlated()/CorrelatedBdeRegressor
below do; see the "Correlated (bivariate) BDE subclasses" comment block ahead of that code for
the full override list.

bde's prior is a single flat prior over all weights in both paths, not the vanilla pipeline's
per-layer hierarchical tau prior, so tau_mean/tau_std (fit()'s output only) are NaN placeholders
kept only so that output still matches the .npz schema the sweep's analysis notebook expects.
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import blackjax
import optax
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_leaves, tree_map
from blackjax.util import streaming_average_update
from sklearn.utils._tags import TargetTags
from tqdm import tqdm as _tqdm

import bde.sampler.warmup as _bde_warmup
import bde.sampler.callbacks as _bde_callbacks
from bde import BdeRegressor
from bde.bde_builder import BdeBuilder
from bde.data.utils import validate_fit_data, validate_predict_data
from bde.loss import GaussianNLL
from bde.loss.loss import BaseLoss
from bde.sampler.mile_wrapper import MileWrapper
from bde.sampler.prior import Prior, log_prior_normal
from bde.sampler.probabilistic import ProbabilisticModel
from bde.training.callbacks import EarlyStoppingCallback, EarlyStoppingState, NullCallback

from rbf_sbi import metrics

PARAM_NAMES = ("sigma", "rho")


def log_prior_student_t(loc=0.0, scale=1.0, nu=3.0):
    """Student-t weight prior, not natively in bde's PriorDist (NORMAL/STANDARDNORMAL/LAPLACE
    only) -- mirrors bde.sampler.prior.log_prior_laplace's exact pattern (see that module, read
    directly 2026-09-05). nu=3.0 default matches Fortuin, Garriga-Alonso, Ober, Wenzel, Ratsch,
    Turner, van der Wilk & Aitchison (2021/ICLR 2022, arXiv:2102.06571) Appendix C's own choice
    ("rather heavy tails, while still having a finite mean and variance") -- their Sec 4.2 finding
    (checked directly against the paper's own PDF text, not a summary) is that fully-connected
    networks' SGD-trained weights are heavy-tailed and that Laplace/Student-t priors beat the
    isotropic Gaussian on FCNN test error/NLL in nearly every case, and remove or invert the cold
    posterior effect there -- motivating this as an alternative to this project's own
    heuristic-scale-tuned priors (optimize_prior.py / optimize_prior_width.py), since nu/loc/scale
    here are fixed literature defaults, not something searched/calibrated per this project's data."""
    def log_prior(params):
        return jnp.sum(jax.scipy.stats.t.logpdf(ravel_pytree(params)[0], df=nu, loc=loc, scale=scale))
    return log_prior


def f_init_student_t(loc=0.0, scale=1.0, nu=3.0):
    """Student-t sampler for bde.sampler.prior.Prior's f_init slot -- never actually invoked in
    this project's fit() path (MCMC always starts from the pretrained weights, confirmed in the
    array-valued-normal workaround's docstring below), but Prior's NamedTuple contract requires a
    callable there regardless. Draws via the standard Gaussian/inverse-Gamma scale-mixture
    construction (Fortuin et al. 2021 Appendix C: "the Student-t also arises as the marginal
    distribution over Gaussians with an inverse-Gamma prior over the variances")."""
    def f_init(key, shape, dtype=jnp.float32):
        gkey, nkey = jax.random.split(key)
        g = jax.random.gamma(gkey, nu / 2, shape, dtype) * (2.0 / nu)  # ~ 1/InverseGamma(nu/2,nu/2)
        z = jax.random.normal(nkey, shape, dtype)
        return loc + scale * z / jnp.sqrt(g)
    return f_init


def _silence_bde_progress():
    """bde's internal MCLMC-warmup and sampling progress bars have no public disable=/verbose=
    option (bde.sampler.warmup.WarmupProgress and bde.sampler.callbacks both hardcode a plain
    tqdm(...)). Left on, they flood non-TTY slurm .out logs with one line per step — at full sweep
    settings (warmup_steps=10000, n_samples=6000, x2 params x6 sizes x5 repeats) that's enormous.
    Monkeypatches the two modules' tqdm references to disable=True; tqdm's own disable mode makes
    update()/refresh()/close() no-ops without raising, so this is safe even though it stops
    reflecting true progress."""
    quiet_tqdm = partial(_tqdm, disable=True)
    _bde_warmup.tqdm = quiet_tqdm
    _bde_callbacks.tqdm = quiet_tqdm


def pad_pretrain_loss_history(pretrain_loss_history):
    """Converts fit_correlated()'s pretrain_loss_history (a list of n_good {epoch, trainloss,
    valloss, stop_epoch} dicts -- see its own docstring, not stacked because members can early-stop
    at different epochs) into padded (n_good, max_len) arrays for a plain rectangular .npz -- NaN
    naturally leaves a gap when plotted, the correct visual for "this member's curve ended here."
    Shared utility (2026-09-13) -- was previously duplicated inline in both
    scripts/bde_correlated_perchain_test.py and scripts/bde_gibbs_test.py.
    Returns (epoch, trainloss, valloss, stop_epoch)."""
    plh = pretrain_loss_history
    max_len = max(len(m["epoch"]) for m in plh)
    epoch = np.full((len(plh), max_len), np.nan)
    trainloss = np.full((len(plh), max_len), np.nan)
    valloss = np.full((len(plh), max_len), np.nan)
    stop_epoch = np.array([m["stop_epoch"] for m in plh])
    for i, m in enumerate(plh):
        L = len(m["epoch"])
        epoch[i, :L] = m["epoch"]
        trainloss[i, :L] = m["trainloss"]
        valloss[i, :L] = m["valloss"]
    return epoch, trainloss, valloss, stop_epoch


def _standardize(X, y):
    """bde's default prior_family='standardnormal' assumes O(1) input/output scale — both X and y
    need standardizing for it to be a reasonable prior, not just for numerical convenience."""
    Xmu, Xstd = X.mean(0), X.std(0) + 1e-8
    ymu, ystd = y.mean(0), y.std(0) + 1e-8
    return (X - Xmu) / Xstd, (y - ymu) / ystd, Xmu, Xstd, ymu, ystd


def _rhat_ess_or_nan(mu_std, rhat_threshold):
    """blackjax.rhat/ess structurally require >=2 chains (`potential_scale_reduction as
    implemented only works for two or more chains`) -- but the member-level retry loops in
    `fit_correlated`/`fit_rasmussen_gibbs_hierarchical` only guard against EXACTLY 0 finite
    members (`n_good == 0` -> RuntimeError), not `n_good == 1`, which both call unconditionally
    into `blackjax.rhat`/`ess` afterward. Added 2026-09-17 after a REAL crash (job 58370119,
    `scripts/run_experiment_strictgate.py`, model=`bnn_temp` (T=0.305) at n_train=10,000): an
    uncaught `AssertionError` from blackjax's own internal chain-count assertion, mid-sweep, after
    only 1/30 repeats -- NOT a threshold/gate rejection (the `rhat_threshold`-based accept/reject
    logic never got a chance to run at all; the crash happens one level below it). Mirrors
    `_fit_one_param`'s own existing graceful-degradation pattern (`n_finite < 2` -> `mixing_ok =
    False`, not a crash) for consistency across this project's two retry-loop designs.

    Returns (rhat, ess, mixing_ok) -- rhat/ess are NaN-filled arrays of the correct per-test-point
    shape when there are <2 chains, so every downstream summary (`rhat_mean`/`rhat_max`/etc. via
    `np.mean`/`np.max` over an all-NaN array) still runs unchanged and correctly reports NaN /
    `mixing_ok=False` rather than crashing the whole sweep repeat."""
    if mu_std.shape[0] < 2:
        shape = mu_std.shape[2:]
        return np.full(shape, np.nan), np.full(shape, np.nan), False
    rhat = np.asarray(blackjax.rhat(jnp.asarray(mu_std), chain_axis=0, sample_axis=1))
    ess = np.asarray(blackjax.ess(jnp.asarray(mu_std), chain_axis=0, sample_axis=1))
    mixing_ok = bool(np.max(rhat) <= rhat_threshold)
    return rhat, ess, mixing_ok


def _fit_one_param(X_train_std, y_col_std, X_test_std, bde_kwargs, seed, max_seed_tries, rhat_threshold):
    """Fits a single-parameter BdeRegressor. bde's public API trains all n_members in one opaque
    call — there's no per-member retry hook, so the retry granularity here is "reseed and refit the
    whole ensemble" (unlike the vanilla pipeline's per-chain retry-on-same-subsample convention,
    which bde's API doesn't expose). Retries until >= 2 members are BOTH finite AND well-mixed
    (R-hat <= rhat_threshold) — "finite" != "well-mixed". If every retry is
    exhausted, falls back to the best (most-finite) attempt seen and flags mixing_ok=False rather
    than crashing the whole sweep repeat on one parameter.

    Returns (raw_kept, used_seed, n_total_members, n_good_members, mixing_ok) where raw_kept has
    shape (n_good_members, n_samples, n_test, 2) — last axis is [mean, scale] in standardized y-space.
    """
    best = None  # (raw, finite_mask, trial_seed) with the most finite members seen so far

    for trial_seed in range(seed, seed + max_seed_tries):
        reg = BdeRegressor(**dict(bde_kwargs, seed=trial_seed))
        reg.fit(x=X_train_std, y=y_col_std)
        raw = np.asarray(reg.predict(X_test_std, raw=True))  # (n_members, n_samples, n_test, 2)

        finite = np.array([np.isfinite(raw[m]).all() for m in range(raw.shape[0])])
        n_finite = int(finite.sum())

        if best is None or n_finite > int(best[1].sum()):
            best = (raw, finite, trial_seed)

        if n_finite < 2:
            continue

        mu_finite = jnp.asarray(raw[finite][..., 0])  # (n_finite, n_samples, n_test)
        rhat = np.asarray(blackjax.rhat(mu_finite, chain_axis=0, sample_axis=1))
        if np.max(rhat) <= rhat_threshold:
            return raw[finite], trial_seed, raw.shape[0], n_finite, True

    raw, finite, trial_seed = best
    n_finite = int(finite.sum())
    if n_finite < 2:
        raise RuntimeError(
            f"exhausted {max_seed_tries} seed retries with <2 finite members (best: {n_finite})"
        )
    return raw[finite], trial_seed, raw.shape[0], n_finite, False



# ---------------------------------------------------------------------------
# Correlated (bivariate) BDE subclasses.
#
# bde/MILE's public API (BdeRegressor) is structurally single-output: its
# sampling-phase likelihood hardcodes univariate Gaussian slicing regardless
# of any custom `loss=`, and its sklearn tags declare single_output=True. A
# genuinely correlated (sigma, rho) posterior requires overriding internal
# (underscore-prefixed, undocumented) hooks -- confirmed by reading the
# installed sklearn-contrib-bde source directly. Four
# overrides were needed, not the three originally scoped:
#
#   1. Bde._build_bde       -- swaps in a BdeBuilder subclass whose
#                               _determine_output_dim returns 5 (mu_sigma,
#                               mu_rho, log_s_sigma, log_s_rho, corr_raw)
#                               instead of the stock hardcoded 2.
#   2. Bde._build_log_post  -- swaps in a ProbabilisticModel subclass whose
#                               log_likelihood is a real bivariate Gaussian
#                               (metrics.bivariate_normal_logpdf) instead of
#                               the stock univariate stats.norm.logpdf on
#                               lvals[...,0:1]/lvals[...,1:2].
#   3. loss=                -- BivariateGaussianNLL below, a custom BaseLoss
#                               for the pretraining stage, consuming the
#                               5-wide head. A custom loss is
#                               necessary, not sufficient,
#                               on its own (see docstring above and (1)/(2)).
#   4. BdeRegressor.fit     -- NOT part of the original three-item plan.
#                               Bde._fit() calls
#                               `self.task.validate_loss(loss_obj)`, and
#                               TaskType.validate_loss hardcodes an allow-list
#                               of loss *classes* ({GaussianNLL, Rmse} for
#                               regression, checked by exact class identity,
#                               not isinstance) that BivariateGaussianNLL can
#                               never belong to. CorrelatedBdeRegressor.fit is
#                               a full copy of Bde._fit's body (with that one
#                               validate_loss call dropped) -- overriding
#                               fit() rather than _fit() because
#                               BdeRegressor.fit()'s `super()._fit(x, y)` is
#                               lexically anchored to BdeRegressor and always
#                               resolves to Bde._fit regardless of type(self);
#                               an _fit override sitting between Bde and
#                               BdeRegressor in the MRO would silently never
#                               run (confirmed the hard way -- see git history
#                               / lab notebook). Also chosen over
#                               monkeypatching the shared TaskType enum, which
#                               would mutate global state every other model
#                               fit in the same process reads.
#
# Also overrides __sklearn_tags__ to declare multi_output=True /
# single_output=False. Without this, bde.data.utils.validate_fit_data
# silently discards the second target column: it reads exactly this tag to
# decide whether a 2D y is a genuine multi-output target or a mistaken
# column-vector, and in the latter case keeps only column 0.
#
# All of this lives in ordinary Python subclasses in this file -- nothing in
# the installed sklearn-contrib-bde package itself is touched. Every
# overridden hook above is an underscore-prefixed or otherwise undocumented
# internal, not stable public API; this was verified against the exact
# installed version this project pins (see the blackjax==1.2.2 note in
# requirements.txt for the same reasoning) and may silently break on a
# future bde/sklearn-contrib-bde upgrade.
# ---------------------------------------------------------------------------


class BivariateGaussianNLL(BaseLoss):
    """Pretraining loss for the 5-wide correlated head: (mu_sigma, mu_rho,
    log_s_sigma, log_s_rho, corr_raw). corr_raw is mapped through tanh to
    (-1, 1); log_s_sigma/log_s_rho through softplus (+ min_sigma) to a
    positive scale -- the same map_fn convention bde's own GaussianNLL uses
    for its scale head, applied to both scale heads here.

    SUM over the batch, not mean (2026-08-25 fix, corrected from an earlier -jnp.mean(...) that
    didn't match reference_code.py's nll_loss, which sums, or the fact that this loss's sum-scale
    needs to match _BivariateProbabilisticModel.log_likelihood's jnp.nansum below -- both must be
    on the same scale since one warm-starts weights the other then samples from the full
    log-posterior of. A mean-scaled likelihood is also wrong on its own Bayesian-math grounds
    here: the flat Normal(0,1) prior's magnitude is fixed regardless of n, so only a
    SUM-scaled (i.e. n-growing) likelihood can correctly dominate the prior more as n grows --
    a mean-scaled likelihood would leave every training size equally prior-dominated."""

    def __init__(self, min_sigma=1e-6, map_fn=jax.nn.softplus):
        self.min_sigma = min_sigma
        self.map_fn = map_fn

    @property
    def name(self):
        return "bivariate_gaussian_nll"

    def __call__(self, preds, y_true):
        mu = preds[..., 0:2]
        s = self.map_fn(preds[..., 2:4]) + self.min_sigma
        corr = jnp.tanh(preds[..., 4])
        return -jnp.sum(metrics.bivariate_normal_logpdf(y_true, mu, s, corr))


class MeanSquaredErrorLoss(BaseLoss):
    """(2026-11-03) Pretraining loss for the MEAN-ONLY predictor (fit_mean_ensemble): a 2-wide head (mu_sigma, mu_rho)
    trained on 0.5 * SUM of squared errors in standardised target space -- the bivariate Gaussian NLL with unit
    variances and zero correlation, up to a constant; summed (not averaged) over the batch like BivariateGaussianNLL,
    so the loss / gradient scale and the early-stopping min_delta act on the same scale as the Gaussian predictor's.
    ASSUMPTION: that unit-variance scaling (0.5 x sum) is the matching choice -- not from any paper. Only preds[..., :2]
    are read, so it also works on a wider head."""

    @property
    def name(self):
        return "mean_squared_error_sum"

    def __call__(self, preds, y_true):
        return 0.5 * jnp.sum((y_true - preds[..., 0:2]) ** 2)


class _MinStepsEarlyStoppingCallback(EarlyStoppingCallback):
    """Same bookkeeping as bde's own EarlyStoppingCallback (best-checkpoint tracking,
    epochs-no-improve counting), but never actually marks a member as stopped before `min_steps`
    epochs have run.

    2026-08-26, in response to a real failure mode: bde's
    BdeBuilder.__init__ hardcodes eval_every=1 (validation checked every epoch, not every
    the earlier standalone baseline's eval_every=100), so a nominal patience=5 meant pretraining could
    -- and at n=100, did -- halt after just 5 epochs. With ~80 rows in the 80/20 split, one noisy
    validation evaluation early on can trigger that, leaving the network effectively blind to data
    it never got a real chance to fit. Pre-floor, every other update (best params, epochs_no_improve) still
    runs exactly as the parent would -- only the stop flag itself is suppressed -- so behavior past
    the floor is identical to what the parent callback would have done from that point on."""

    def __init__(self, *, patience, min_delta, eval_every=1, min_steps=0):
        super().__init__(patience=patience, min_delta=min_delta, eval_every=eval_every)
        self.min_steps = min_steps

    def update(self, state, epoch, params_de, val_lvals_de):
        new_state = super().update(state, epoch, params_de, val_lvals_de)
        if epoch < self.min_steps:
            return EarlyStoppingState(
                best_params_de=new_state.best_params_de,
                best_metric_de=new_state.best_metric_de,
                epochs_no_improve_de=new_state.epochs_no_improve_de,
                stopped_de=state.stopped_de,
                stop_epoch_de=state.stop_epoch_de,
            )
        return new_state


class _CorrelatedBdeBuilder(BdeBuilder):
    """Widens the network's output head from bde's hardcoded 2 to 5, and swaps in
    _MinStepsEarlyStoppingCallback above in place of bde's stock EarlyStoppingCallback.
    eval_every/min_steps are read from plain attributes set by CorrelatedBdeRegressor.fit() right
    after construction (see there) rather than sklearn-declared constructor params -- consistent
    with this class already not being part of bde's stable public API (see the module-level
    override notes above)."""

    def _determine_output_dim(self, y):
        # 2026-11-03: output_dim_override (plain attribute set by CorrelatedBdeRegressor._build_bde from the
        # regressor's pretrain_output_dim; default None -> 5, unchanged). 2 = the mean-only predictor.
        return getattr(self, "output_dim_override", None) or 5

    def _create_callback(self):
        if self.patience is None:
            return NullCallback()
        return _MinStepsEarlyStoppingCallback(
            patience=self.patience, min_delta=self.min_delta,
            eval_every=self.eval_every, min_steps=getattr(self, "min_steps", 0),
        )


def kfold_val_indices(n, k, seed=42):
    """(2026-11-02) K-fold validation index sets for per-member pretraining (see
    CorrelatedBdeRegressor._pretrain_kfold): one fixed permutation of range(n) cut into k
    near-equal consecutive folds; returns a list of k sorted index arrays. Independent of the
    member/retry seed, so a re-seeded attempt sees the same folds. seed=42 echoes bde's own fixed
    train_test_split(random_state=42); the folds are NOT that split (no fold equals it)."""
    perm = np.random.default_rng(seed).permutation(n)
    return [np.sort(f) for f in np.array_split(perm, k)]


# Derived from sklearn-contrib-bde 1.0.2 (BdeBuilder.fit_members), (c) 2025 Arvanitis Vyron, Aslanidis Angelos,
# Sommer Emanuel and scikit-learn-contrib contributors; BSD 3-Clause License, see THIRD_PARTY_LICENSES.
def _fit_members_explicit_split(builder, x_train, y_train, x_val, y_val, epochs, optimizer=None, loss=None):
    """(2026-11-02) Line-for-line copy of bde's BdeBuilder.fit_members (installed
    bde/bde_builder.py, sklearn-contrib-bde under blackjax 1.2.2) with ONE change: the
    train/validation split is passed in instead of being drawn inside by
    super().split_train_val(x, y, val_size=self.val_split) (train_test_split, random_state=42).
    That inner call cannot be redirected from outside (super() resolves on the class), so per-member
    splits need this copy. Everything else -- member initialisation, the pmap'ed training loop,
    the early-stopping callback (here _MinStepsEarlyStoppingCallback via _create_callback), best-
    checkpoint selection and the final assignment of member params / params_e -- is bde's own
    code path. Verified 2026-11-02: called with bde's own split it reproduces fit_members
    bit-for-bit on CPU."""
    full_sizes = builder._build_full_sizes(x_train, y_train)
    builder._ensure_member_initialization(full_sizes)
    components = builder._create_training_components(optimizer, loss)
    state = builder._prepare_distributed_state(components)
    builder._reset_history()
    callback = builder._create_callback()
    callback_state = callback.initialize(state.params_de)
    loop_result = builder._training_loop(
        state, callback, callback_state, x_train, y_train, x_val, y_val, epochs,
    )
    callback.stop_epoch_de(loop_result.callback_state, ensemble_size=state.ensemble_size)
    params_e_final = callback.best_params(loop_result.callback_state, ensemble_size=state.ensemble_size)
    for i, m in enumerate(builder.members):
        m.params = tree_map(lambda a: a[i], params_e_final)
    builder.params_e = params_e_final
    return builder.members


class _BivariateProbabilisticModel(ProbabilisticModel):
    """Replaces bde's hardcoded univariate-Gaussian sampling-phase likelihood
    with a real bivariate Gaussian over (sigma, rho), consuming the 5-wide
    head. See the module-level override notes above.

    temperature (2026-08-28, default 1.0 -- untempered, exactly bde's stock
    log_prior + log_likelihood): first exploratory trial for the aleatoric-inflation
    investigation.
    Tempers the LIKELIHOOD term only (log_likelihood / temperature), leaving log_prior
    unscaled -- matches Delaunoy, de la Brassinne Bonardeaux, Mishra-Sharma & Louppe
    (2024)'s Eq 12 tempered-ELBO term (their T multiplies only KL(q||prior); Wenzel et
    al. (2020) Appendix E shows this is equivalent, in an MCMC/log-density setting, to
    scaling the likelihood by 1/T with the prior left untouched) -- NOT Wenzel et al.'s
    own headline "full" cold-posterior formulation, which scales (log_prior +
    log_likelihood) together by 1/T. Confirmed via direct source read (2026-08-28) that
    bde's own log_unnormalized_posterior (bde.sampler.probabilistic.ProbabilisticModel)
    has no temperature/tempering of any kind -- this is a new mechanism, not exposing an
    existing bde knob."""

    def __init__(self, model, params, prior, task, temperature=1.0):
        super().__init__(model, params, prior, task)
        self.temperature = temperature

    def log_likelihood(self, params, x, y, **kwargs):
        lvals = self.model.forward(params, x)
        mu = lvals[..., 0:2]
        s = jax.nn.softplus(lvals[..., 2:4]) + 1e-6
        corr = jnp.tanh(lvals[..., 4])
        return jnp.nansum(metrics.bivariate_normal_logpdf(y, mu, s, corr))

    def log_unnormalized_posterior(self, positions, x, y, **kwargs):
        return self.log_prior(positions) + self.log_likelihood(positions, x, y, **kwargs) / self.temperature


def _map_init_hierarchical_tau(W_pytree, hyper_log_tau_mu, hyper_log_tau_sigma, tau_link, tau_max,
                                u_tau_init_guess, n_newton_steps=50):
    """"MAP init" (2026-09-13): per-weight MAP solve for u_tau given the ALREADY-PRETRAINED,
    FIXED physical weight W_i, used to build a better starting point for MCLMC than today's
    "same tau_init for every weight regardless of its own pretrained magnitude" scheme (every
    weight currently starts from an identical u_tau_init, so raw_init = W_i/tau_init lands at
    wildly different -- and often atypical/tail -- values of its own N(0,1) prior depending purely
    on how big W_i happens to be).

    Does NOT change the prior itself -- log_prior/log_likelihood (what MCLMC actually samples
    from) are completely untouched, still raw~N(0,1) i.i.d. and u_tau~N(hyper_log_tau_mu,
    hyper_log_tau_sigma) i.i.d., zero-mean throughout. This only picks a smarter INITIAL POINT for
    the chain, exactly analogous to routinely starting HMC/NUTS/MCLMC at or near a MAP estimate --
    standard practice that doesn't bias what a well-mixing chain eventually samples. Real risk,
    not yet checked empirically: if MCLMC's own mixing is poor (this project's chronic problem),
    a better-informed init could make a badly-exploring chain LOOK more settled (a smoother trace)
    without the chain having explored any more of the target than before -- ESS, not trace
    smoothness alone, is what would catch this.

    The objective, solved independently for every weight (no cross terms: raw~N(0,1) and
    u_tau~N(hyper_mu,hyper_sigma) are both i.i.d. across weights, and W_i is held fixed):
        u_tau_i* = argmin_u  (W_i / link(u))^2 / 2  +  ((u - hyper_log_tau_mu) / hyper_log_tau_sigma)^2 / 2
                              \\_____ -log p(raw_i), raw_i := W_i/link(u) _____/   \\_____ -log p(u_tau_i) _____/
    raw_i* := W_i / link(u_tau_i*) exactly reproduces the pretrained physical weight (the network's
    own data fit from pretraining is untouched) -- only the raw/tau SPLIT MCLMC starts from moves.
    Coercive in u for both tau_link choices (checked directly, not assumed): as u moves toward the
    link's zero-saturation side, W_i/link(u) -> +-inf for any W_i != 0 so the raw-term diverges; as
    u -> +inf, either the hyperprior term diverges on its own ("softplus", where link(u) ~ u so the
    raw-term instead -> 0) or the link itself saturates at tau_max while the hyperprior term still
    diverges ("bounded") -- an interior minimum always exists.

    Solved via damped Newton's method, applied identically and independently to every element of
    every leaf via jnp's own elementwise vectorization (no vmap needed): since the summed objective
    is separable across weights, jax.grad of the SUM w.r.t. the full array u gives exactly each
    weight's own elementwise derivative (cross terms are structurally absent), and the same trick
    applied to grad's own sum gives the elementwise second derivative -- the standard trick for
    turning a huge batch of independent 1-D Newton problems into two whole-array autodiff calls per
    iteration instead of a python loop or vmap over ~2245 scalar problems. Hessian floored at a
    small positive epsilon (safety margin against a locally non-convex step, not expected to bind
    given coercivity, but cheap insurance since this runs unattended before any real fit).

    Returns (raw_init, u_tau_init) pytrees, same structure/shape as W_pytree. NOT yet checked
    against a plain gradient-descent solve or a grid search for agreement."""
    def _link(u):
        return jax.nn.softplus(u) if tau_link == "softplus" else tau_max * jax.scipy.stats.norm.cdf(u)

    def _solve_one(w):
        def nll_sum(u):
            raw = w / _link(u)
            return jnp.sum(0.5 * raw ** 2 + 0.5 * jnp.square((u - hyper_log_tau_mu) / hyper_log_tau_sigma))
        grad_fn = jax.grad(nll_sum)

        def hess_diag_fn(u):
            return jax.grad(lambda uu: jnp.sum(grad_fn(uu)))(u)

        def newton_step(u, _):
            g, h = grad_fn(u), hess_diag_fn(u)
            u_new = u - g / jnp.maximum(h, 1e-8)
            return u_new, None

        u_final, _ = jax.lax.scan(
            newton_step, jnp.full_like(w, u_tau_init_guess), xs=None, length=n_newton_steps
        )
        return u_final

    u_tau_init = tree_map(_solve_one, W_pytree)
    raw_init = tree_map(lambda w, u: w / _link(u), W_pytree, u_tau_init)
    return raw_init, u_tau_init


class _HierarchicalBivariateProbabilisticModel(ProbabilisticModel):
    """Genuinely hierarchical, PER-WEIGHT (ARD-style) variant of _BivariateProbabilisticModel
    above. EVERY individual weight w_i gets its OWN scale tau_i -- w_i | tau_i ~ Normal(0,
    tau_i^2), i.i.d. across all ~2245 weights, jointly sampled with the weights by the SAME MCLMC
    chain -- not a single shared/isotropic scale for the whole network. 2026-09-10: rebuilt from
    an earlier, isotropic version (one shared scalar tau for every weight) since the intended
    design is a different Gaussian per weight, not one shared scale. Of the two natural per-weight structures
    (this simple i.i.d. version vs. a horseshoe-style global-times-local scale, both considered --
    Ghosh & Doshi-Velez 2017; Louizos,
    Ullrich & Welling 2017), the simple i.i.d. version was chosen over horseshoe, specifically to avoid adding a 3rd sampled scale layer on top of an
    already dimensionality-doubling change.

    Motivated directly by a real, repeated finding in this project (2026-09-05/09): every pre-data-calibrated prior scale tried so far (Delaunoy et al.'s
    coverage-optimized per-weight prior, and independently the Student-t coverage-optimized
    width) improved its OWN pre-data calibration objective while making downstream fit quality
    (test NLL, MAE) monotonically WORSE -- the calibration objective has no term connecting it to
    post-data fitting capacity. Inferring each tau_i from the SAME likelihood used for everything
    else removes that disconnect: there is one objective (the joint posterior), not two competing
    ones.

    Non-centered parameterization, applied per-weight: the sampled position is {"raw":
    <network-params-shaped pytree>, "u_tau": <SAME-shaped pytree, one u_tau_i per raw_i>}, with a
    FIXED (tau-independent) standard-normal prior on every raw_i and physical weights recovered
    at every likelihood evaluation as physical_i = |u_tau_i| * raw_i (elementwise). This decouples
    each raw_i's posterior geometry from its own tau_i -- the alternative, CENTERED
    parameterization (w_i ~ Normal(0, tau_i) directly, tau_i sampled) is the textbook source of
    "funnel" geometry (sharp, tau-dependent curvature) that HMC/MCLMC-family samplers struggle
    with, which this project's MCLMC is already known to struggle with even in the simpler
    non-hierarchical case (n_train=10,000 mixing failure, R-hat 7-17, mechanism still unconfirmed).
    ASSUMPTION, cited from memory, NOT verified against either paper's text: Papaspiliopoulos, Roberts & Skold (2007), "A General Framework for
    the Parametrization of Hierarchical Models" (Statistical Science) for the reparameterization
    itself; Betancourt & Girolami (2015), "Hamiltonian Monte Carlo for Hierarchical Models" for
    the funnel-avoidance motivation.

    Positivity link tau_i = |u_tau_i| (Half-Normal), NOT exp(u_tau_i) -- 2026-09-10, changed after
    a real failure the exp() version produced:
    tau_i = exp(u_tau_i) makes tau LOG-NORMAL, whose right tail decays only like exp(-(ln t)^2) --
    heavy enough that a handful of weakly-identified-by-the-n=100-likelihood weights' u_tau_i
    random-walked to large values under the sampler and exploded (observed tau_i up to ~1.6e10,
    aleatoric predictive std blowing up to a mean of 2.3 million). Remedy: use a
    LIGHT-tailed prior for the width, since heavy-tailed width priors explode. u_tau_i ~
    Normal(0,1) directly, tau_i = |u_tau_i| -- i.e. tau_i ~ Half-Normal(1), Gelman (2006)'s own
    named recommendation for a weakly-informative scale prior, also cited below -- gives tau_i the
    exact Gaussian (exp(-t^2)) tail its own hyperprior has, with no reshaping at all (contrast the
    softplus(u) link tried in between, which was equally light-tailed asymptotically but distorted
    the prior's shape near 0 -- |u_tau_i| is literally "u_tau_i's own N(0,1) density, folded," the
    most direct reading of "N(0, tau_i) with N(0,1) as hyperprior").
    Non-differentiable at tau=0 (a real, if minor, structural difference from softplus/exp, both
    smooth everywhere) -- HMC/MCLMC-family samplers generally tolerate this fine since it's a
    measure-zero point and jax.numpy.abs's gradient is well-defined a.e.

    2026-09-10, SUPERSEDES the |u_tau_i| link above -- real failure found in job 58163512 (n=100):
    even with the light tail, one chain's u_tau_i for 2 of 2245 weights (both in the final 16->5
    output layer, both weakly-identified -- pretrained |weight| at the 3rd/18th percentile) still
    wandered out to tau_i ~ 8.5-9.2 (posterior STD nearly equal to the mean at those two indices,
    i.e. not a converged high-tau mode, more likely incomplete mixing along those two dimensions),
    which fed through the log_s output heads (softplus, ~linear for large inputs) and inflated the
    POOLED predictive aleatoric std ~150x every other run on disk (48.9/34.3 vs. the 0.2-0.5 range
    everywhere else). |u_tau_i| is light-tailed relative to log-normal, but still UNBOUNDED --
    nothing stops an individual weakly-identified weight's tau from drifting arbitrarily far into
    that light tail. Fix: bound tau outright. New link tau_i = tau_max * Phi(u_tau_i)
    (Phi = standard normal CDF, jax.scipy.stats.norm.cdf), strictly bounded in (0, tau_max) --
    chosen over a hard clip (tau_i = min(|u_tau_i|, tau_max)) specifically because MCLMC is
    gradient-based: a hard clip has EXACTLY ZERO likelihood-gradient once |u_tau_i| > tau_max (only
    the hyperprior's own quadratic term still pulls back), whereas Phi is smooth and has nonzero
    gradient everywhere, which a gradient-based sampler needs to correctly explore the whole space.
    tau_max default 1.0 -- makes this a
    SHRINKAGE-ONLY prior (no weight's prior std can exceed the standard N(0,1) baseline's own),
    which directly forecloses the failure mode above by construction, at the cost of no longer being
    able to express a per-weight prior WIDER than standard. No Jacobian correction needed here
    either, same reasoning as every earlier link (u_tau_i is what MCLMC samples natively; tau is
    always only a deterministic downstream quantity). UNVERIFIED: whether this
    actually fixes mixing/ESS as well as it fixes boundedness -- not yet run at the time this note
    was written.

    Hyperprior: u_tau_i ~ Normal(hyper_log_tau_mu, hyper_log_tau_sigma) i.i.d. for every i -- same
    parameter NAMES as the earlier exp()-based version (kept stable across every relink so far to
    avoid touching every call site's kwarg names); with this link, u_tau_i's Normal density IS
    tau_i's own hyperprior in the most literal sense (folded at 0), not a reparameterized stand-in
    for a differently-shaped distribution the way log-normal (exp link) or softplus-normal were.
    Gelman (2006), "Prior distributions for variance parameters in hierarchical models" (Bayesian
    Analysis) is the standard reference for weakly-informative group-level-scale hyperpriors,
    cited here from memory of its general conclusion (favor a weak proper, light-tailed prior
    over the old conjugate inverse-gamma default OR a heavy-tailed alternative) -- not a verified
    quote.

    Real, stated-up-front risk this change does NOT resolve on its own: per-weight tau still
    DOUBLES the sampled dimensionality (~2245 weights -> ~4490 total) relative to the earlier
    isotropic version, which already struggled to mix well (R-hat 1.1-1.3,
    2026-09-10). The light-tailed link directly targets the numerical-explosion failure mode
    observed with exp(); it is not expected, by itself, to fix the underlying dimensionality-driven
    mixing difficulty -- that remains open.

    `params`/`prior` passed to the parent ProbabilisticModel.__init__ are for informational
    bookkeeping only (n_params count; prior=None is safe here since log_prior below is fully
    overridden and the parent's own log_prior -> self.prior.log_prior(...) path is never reached)."""

    def __init__(self, model, params, task, temperature=1.0,
                 hyper_log_tau_mu=0.0, hyper_log_tau_sigma=1.0, tau_max=1.0,
                 tau_link="bounded"):
        super().__init__(model, params, None, task)
        self.temperature = temperature
        self.hyper_log_tau_mu = hyper_log_tau_mu
        self.hyper_log_tau_sigma = hyper_log_tau_sigma
        self.tau_max = tau_max
        # 2026-09-11: tau_link selects the positivity link applied to the sampled unconstrained
        # u_tau -- "bounded" (default) is the current production link, tau_i = tau_max*Phi(u_tau_i)
        # (see class docstring's "Positivity link" section). "softplus" reinstates the PRIOR link
        # (tau_i = softplus(u_tau_i), superseded 2026-09-10 after the |u_tau_i| link's explosion --
        # see docstring) as an explicit, opt-in choice for direct comparison (e.g. an MCLMC NLL-trace
        # diagnostic run) -- NOT a reversion of the production default, which stays "bounded" unless
        # this is set explicitly. No Jacobian correction needed for either link (same reasoning as
        # every link tried: MCLMC samples u_tau natively, tau is always a deterministic downstream
        # quantity, never itself sampled) -- log_prior below is IDENTICAL for both links.
        if tau_link not in ("bounded", "softplus"):
            raise ValueError(f"tau_link must be 'bounded' or 'softplus', got {tau_link!r}")
        self.tau_link = tau_link

    def log_prior(self, positions):
        raw_flat = ravel_pytree(positions["raw"])[0]
        u_tau_flat = ravel_pytree(positions["u_tau"])[0]
        raw_lp = jnp.sum(jax.scipy.stats.norm.logpdf(raw_flat, loc=0.0, scale=1.0))
        # Density of the Normal hyperprior evaluated directly on the SAMPLED unconstrained
        # variable u_tau -- no Jacobian correction needed (same as every earlier link tried):
        # MCLMC samples u_tau natively, and tau is only ever a deterministic downstream quantity
        # for the likelihood/reporting, never itself sampled. Identical for every tau_link choice.
        tau_lp = jnp.sum(jax.scipy.stats.norm.logpdf(
            u_tau_flat, loc=self.hyper_log_tau_mu, scale=self.hyper_log_tau_sigma
        ))
        return raw_lp + tau_lp

    def log_likelihood(self, positions, x, y, **kwargs):
        # Elementwise per-weight product -- positions["u_tau"] mirrors positions["raw"]'s exact
        # pytree structure/shape (one u_tau_i per raw_i). See self.tau_link / __init__'s docstring
        # note for the two supported links.
        if self.tau_link == "softplus":
            link_fn = lambda t: jax.nn.softplus(t)
        else:
            link_fn = lambda t: self.tau_max * jax.scipy.stats.norm.cdf(t)
        physical = tree_map(
            lambda t, r: link_fn(t) * r,
            positions["u_tau"], positions["raw"],
        )
        lvals = self.model.forward(physical, x)
        mu = lvals[..., 0:2]
        s = jax.nn.softplus(lvals[..., 2:4]) + 1e-6
        corr = jnp.tanh(lvals[..., 4])
        return jnp.nansum(metrics.bivariate_normal_logpdf(y, mu, s, corr))

    def log_unnormalized_posterior(self, positions, x, y, **kwargs):
        return self.log_prior(positions) + self.log_likelihood(positions, x, y, **kwargs) / self.temperature


class _CenteredHierarchicalBivariateProbabilisticModel(ProbabilisticModel):
    """CENTERED per-weight hierarchical variant -- direct empirical baseline (2026-09-10):
    _HierarchicalBivariateProbabilisticModel's own docstring calls this the
    textbook source of "funnel" geometry that non-centering was chosen to avoid, but that claim
    (Papaspiliopoulos, Roberts & Skold 2007; Betancourt & Girolami 2015, both cited there from
    memory, not verified against the paper text) has never actually been tested empirically in
    THIS project's own MCLMC setup. This class exists to test it directly: does non-centering
    demonstrably help here, or was avoiding it only ever a theoretical precaution?

    Every weight w_i is sampled DIRECTLY with density Normal(0, tau_i) -- no fixed-unit-scale
    "raw" auxiliary variable, no multiplicative decoupling. Sampled position: {"w": <weight-shaped
    pytree>, "u_tau": <SAME-shaped pytree>}. tau_i = tau_max*Phi(u_tau_i), bounded in (0, tau_max)
    -- the SAME link as the non-centered class (see its docstring's "Positivity link" section,
    2026-09-10 supersession, for the full explosion investigation and why a bounded/smooth link was
    chosen over |u_tau_i| and over a hard clip), used here deliberately so centering-vs-non-centering
    is the ONLY difference between this class and _HierarchicalBivariateProbabilisticModel --
    changing the link here too would confound the two questions (link-shape effects vs.
    centered-funnel effects).

    log_prior evaluates Normal(0, tau_i).logpdf(w_i) directly -- tau_i enters the WEIGHT's own
    prior density (not a separate multiplier applied downstream), which is exactly the
    sharp/tau-dependent curvature (small tau_i implies a very narrow, steep density in w_i) that
    produces funnel geometry. log_likelihood needs no raw*tau product -- positions["w"] IS the
    physical weight already.

    `params`/`prior` passed to the parent ProbabilisticModel.__init__ are for informational
    bookkeeping only, same as the non-centered class."""

    def __init__(self, model, params, task, temperature=1.0,
                 hyper_log_tau_mu=0.0, hyper_log_tau_sigma=1.0, tau_max=1.0):
        super().__init__(model, params, None, task)
        self.temperature = temperature
        self.hyper_log_tau_mu = hyper_log_tau_mu
        self.hyper_log_tau_sigma = hyper_log_tau_sigma
        self.tau_max = tau_max

    def log_prior(self, positions):
        w_flat = ravel_pytree(positions["w"])[0]
        u_tau_flat = ravel_pytree(positions["u_tau"])[0]
        tau_flat = self.tau_max * jax.scipy.stats.norm.cdf(u_tau_flat)
        # tau_i is the WEIGHT's own prior scale here (centered) -- contrast with the non-centered
        # class, where raw's prior scale is always a fixed 1.0 regardless of tau.
        w_lp = jnp.sum(jax.scipy.stats.norm.logpdf(w_flat, loc=0.0, scale=tau_flat))
        tau_lp = jnp.sum(jax.scipy.stats.norm.logpdf(
            u_tau_flat, loc=self.hyper_log_tau_mu, scale=self.hyper_log_tau_sigma
        ))
        return w_lp + tau_lp

    def log_likelihood(self, positions, x, y, **kwargs):
        lvals = self.model.forward(positions["w"], x)
        mu = lvals[..., 0:2]
        s = jax.nn.softplus(lvals[..., 2:4]) + 1e-6
        corr = jnp.tanh(lvals[..., 4])
        return jnp.nansum(metrics.bivariate_normal_logpdf(y, mu, s, corr))

    def log_unnormalized_posterior(self, positions, x, y, **kwargs):
        return self.log_prior(positions) + self.log_likelihood(positions, x, y, **kwargs) / self.temperature


def _bivariate_nll(model, w, x, y):
    """Plain (unpenalized) negative log-likelihood of the bivariate Gaussian head at physical
    weights w -- same math as _BivariateProbabilisticModel/_HierarchicalBivariateProbabilisticModel
    's own log_likelihood (softplus scale heads, tanh-correlation head), sign-flipped into a
    MINIMIZED quantity for direct use with a gradient-based optimizer, not MCMC. Shared by both
    the M-step loss and the E-step Hessian below so the two stay consistent with each other."""
    lvals = model.forward(w, x)
    mu = lvals[..., 0:2]
    s = jax.nn.softplus(lvals[..., 2:4]) + 1e-6
    corr = jnp.tanh(lvals[..., 4])
    return -jnp.nansum(metrics.bivariate_normal_logpdf(y, mu, s, corr))


def fit_evidence_hierarchical(model, w_init, x, y, n_outer_iters=20, weight_opt_steps=500,
                               weight_opt_lr=1e-3, tau_init=1.0, tau_floor=1e-4,
                               hessian_jitter=1e-6):
    """MacKay (1992), "A Practical Bayesian Framework for Backpropagation Networks" (Neural
    Computation) -- the evidence framework / type-II maximum likelihood approach, implemented here
    (2026-09-13) as a direct-request ALTERNATIVE to this project's existing hierarchical-MCMC path
    (_HierarchicalBivariateProbabilisticModel, which jointly SAMPLES weights and every weight's own
    prior width tau_i via MCLMC). Does not touch MCLMC/bde's sampling machinery at all -- a
    genuinely separate inference algorithm, not a variant of the MCMC one, per explicit request to
    compare the two as two different methods rather than test a shared code path.

    Alternates, per outer iteration:
      (a) M-STEP (weights): given the CURRENT per-weight tau_i (equivalently precision
          alpha_i = 1/tau_i^2), find MAP weights
              w* = argmin_w  NLL_likelihood(w; x, y)  +  0.5 * sum_i alpha_i * w_i^2
          via Adam (weight_opt_steps steps, fresh optimizer state every outer iteration, warm-started
          from the previous outer iteration's w*).
      (b) E-STEP ("evidence" update): given w* and the LAPLACE APPROXIMATION to the posterior
          covariance at that MAP point, update every tau_i via MacKay's classic per-hyperparameter
          evidence formula (same update independently rediscovered as ARD -- Neal (1996), *Bayesian
          Learning for Neural Networks* -- and as the Relevance Vector Machine -- Tipping (2001),
          "Sparse Bayesian Learning and the Relevance Vector Machine," JMLR):
              H = Hessian of NLL_likelihood(w) at w=w* (curvature of the LIKELIHOOD term only)
              Sigma = (H + diag(alpha))^{-1}                    (Laplace-approximate posterior covariance)
              gamma_i = 1 - alpha_i * Sigma_ii                  ("effective number of parameters" per weight,
                                                                  in [0, 1]: how much weight i's own value is
                                                                  actually pinned down by the data vs. still
                                                                  determined by its prior)
              tau_i_new = sqrt(w_i*^2 / gamma_i)
          gamma_i is clipped to [tau_floor^2, 1] before use, both to keep the update well-defined
          when Sigma_ii's finite-precision estimate drifts slightly outside (0, 1/alpha_i) and to
          floor how small a single update step can drive tau_i.
    ASSUMPTION, not yet verified against MacKay (1992)'s own text: the exact update
    formula above (cited here from the general ARD/evidence-framework literature, e.g. as restated
    in Bishop's PRML Sec 7.2.2 for the closely-related RVM) -- verify against the original paper.

    Uses the FULL (dense) Hessian of NLL_likelihood at w*, not a diagonal or Gauss-Newton
    approximation. DELIBERATE CHOICE, flagged as an assumption: this project's network is small
    enough (~2245 weights for the [16,16,16] architecture used elsewhere) that an exact O(n^2)
    dense Hessian and its O(n^3) inversion are both cheap relative to this project's existing
    2245-dimensional MCLMC compute budget, so there's no scale-driven need for the cheaper
    diagonal/Gauss-Newton/K-FAC approximations most large-scale Laplace-approximation-for-NNs work
    uses out of necessity (e.g. Daxberger et al. (2021), "Laplace Redux," NeurIPS). Revisit this
    choice if the architecture ever grows past the point a dense n x n Hessian/inversion is
    affordable -- NOT yet checked how large that is on this project's actual GPU budget, since this
    has not been run for real yet.

    REAL, QUALITATIVE difference from the MCMC-hierarchical path, flagged directly: gamma_i -> 0
    for a weight the data poorly constrains drives tau_i -> tau_floor, i.e. this can (and, per the
    ARD/RVM literature's own well-known behavior, likely will) PRUNE some weights to an effectively
    fixed, near-zero prior width -- forcing that weight toward exactly 0 in every subsequent M-step.
    This is the classic "automatic relevance determination" sparsification effect, not a bug, but a
    qualitatively different outcome from the MCMC path's purely probabilistic (never exactly zero)
    shrinkage. Not yet measured how much pruning actually happens on this project's real
    architecture/data -- flag in any comparison against the MCMC-hierarchical baseline, don't treat
    a lower effective parameter count as automatically better or worse without checking fit quality.

    Produces a POINT estimate (w*) plus a per-weight WIDTH (tau) at convergence, not posterior
    samples -- unlike the MCMC path's actual draws. A Laplace-approximate posterior-predictive
    (draw w ~ N(w*, Sigma) and push through model.forward) is the natural way to get something
    comparable to the MCMC path's ensemble/uncertainty outputs, but is NOT implemented here yet --
    scope this once there's a concrete need to compare against the existing per-chain analysis
    notebooks, rather than building it speculatively now.

    Not yet integrated into CorrelatedBdeRegressor/fit_correlated or any CLI script -- this is the
    core algorithm only, smoke-tested at toy scale but NOT yet run
    against real project data.

    Args:
      model: any object exposing model.forward(w, x) -> (..., 5) array (physical weights, network
        architecture) -- e.g. a pretrained bde ensemble member (self._bde.members[i]), passed as-is,
        NEVER mutated by this function (a fresh w_init pytree is optimized independently).
      w_init: pytree, initial physical weights (e.g. that same member's already-pretrained
        .params) -- the M-step's very first outer iteration warm-starts from here, exactly
        analogous to how the MCMC path starts MCLMC from the pretrained point.
      x, y: training data, same convention as every other log_likelihood in this module.
      n_outer_iters: number of (M-step, E-step) alternations.
      weight_opt_steps / weight_opt_lr: Adam steps / learning rate for each M-step.
      tau_init: initial per-weight tau (scalar, broadcast to every weight) before the first E-step.
      tau_floor: minimum tau_i (and minimum sqrt(gamma_i) allowed in the E-step's clip) -- see the
        pruning note above.
      hessian_jitter: added to the Hessian's diagonal before inversion, pure numerical safety
        against a marginally-indefinite curvature estimate at a not-fully-converged M-step optimum.

    Returns a dict: {"w": final MAP weights (same pytree structure as w_init), "tau": final
    per-weight tau as a FLAT array (ravel_pytree(w_init) order -- caller unravels if a pytree shape
    is needed, same convention as this module's other flattening helpers), "history": a list of
    n_outer_iters dicts {outer_iter, weighted_nll (M-step's final loss value),
    data_nll (NLL_likelihood alone, unpenalized -- lets a caller see fit quality separately from
    the prior penalty that also moves during optimization), tau_mean, tau_std, tau_min, tau_max,
    n_near_floor (count of tau_i within 10x tau_floor, informational only)} for convergence
    inspection, "Sigma": the (n_weights, n_weights) Laplace-approximate posterior covariance from
    the FINAL outer iteration's E-step -- (H + diag(alpha))^{-1} at the final (w, tau) -- exposed
    so a caller can draw approximate posterior-predictive samples w ~ N(w, Sigma) without
    recomputing the Hessian a second time.}"""
    w_flat_init, unravel = ravel_pytree(w_init)
    n_weights = w_flat_init.shape[0]
    tau = jnp.full((n_weights,), tau_init)
    w = w_init
    history = []

    for outer in range(n_outer_iters):
        # ---- M-step: MAP weights given the current tau (Adam, fresh optimizer state each outer
        # iteration, warm-started from the previous outer iteration's w). ----
        alpha_flat = 1.0 / jnp.square(tau)
        alpha_pytree = unravel(alpha_flat)

        def loss_fn(w_):
            data_nll = _bivariate_nll(model, w_, x, y)
            penalty = 0.5 * sum(
                jnp.sum(a_leaf * jnp.square(w_leaf))
                for w_leaf, a_leaf in zip(tree_leaves(w_), tree_leaves(alpha_pytree))
            )
            return data_nll + penalty, data_nll

        opt = optax.adam(weight_opt_lr)
        opt_state = opt.init(w)

        def opt_step(carry, _):
            w_, opt_state_ = carry
            (loss_val, data_nll_val), grads = jax.value_and_grad(loss_fn, has_aux=True)(w_)
            updates, opt_state_ = opt.update(grads, opt_state_, w_)
            w_ = optax.apply_updates(w_, updates)
            return (w_, opt_state_), (loss_val, data_nll_val)

        (w, opt_state), (loss_trace, data_nll_trace) = jax.lax.scan(
            opt_step, (w, opt_state), xs=None, length=weight_opt_steps
        )
        weighted_nll_final = float(loss_trace[-1])
        data_nll_final = float(data_nll_trace[-1])

        # ---- E-step: evidence update for tau given the Laplace approximation at this w. ----
        w_flat = ravel_pytree(w)[0]
        H = jax.hessian(lambda wf: _bivariate_nll(model, unravel(wf), x, y))(w_flat)
        precision = H + jnp.diag(alpha_flat) + hessian_jitter * jnp.eye(n_weights)
        Sigma = jnp.linalg.inv(precision)
        Sigma_diag = jnp.diag(Sigma)
        gamma = jnp.clip(1.0 - alpha_flat * Sigma_diag, tau_floor ** 2, 1.0)
        tau = jnp.sqrt(jnp.maximum(jnp.square(w_flat) / gamma, tau_floor ** 2))

        history.append(dict(
            outer_iter=outer, weighted_nll=weighted_nll_final, data_nll=data_nll_final,
            tau_mean=float(tau.mean()), tau_std=float(tau.std()),
            tau_min=float(tau.min()), tau_max=float(tau.max()),
            n_near_floor=int(jnp.sum(tau < 10 * tau_floor)),
        ))

    return {"w": w, "tau": tau, "history": history, "Sigma": Sigma}


class CorrelatedBdeRegressor(BdeRegressor):
    """BdeRegressor variant fitting a single joint (sigma, rho) bivariate
    posterior instead of two independent univariate ones. See the
    module-level override notes above for what's overridden and why."""

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags = TargetTags(
            required=True, one_d_labels=False, two_d_labels=True,
            positive_only=False, multi_output=True, single_output=False,
        )
        return tags

    def _build_bde(self):
        hidden_layers = self.hidden_layers if self.hidden_layers is not None else [4, 4]
        self._resolved_hidden_layers = list(hidden_layers)
        self._bde = _CorrelatedBdeBuilder(
            hidden_sizes=self._resolved_hidden_layers,
            patience=self.patience, val_split=self.validation_split,
            n_members=self.n_members, lr=self.lr, weight_decay=self.weight_decay,
            task=self.task, seed=self.seed, act_fn=self.activation,
        )
        self._bde.output_dim_override = getattr(self, "pretrain_output_dim", None)   # 2026-11-03, default None -> 5
        self.members_ = self._bde.members

    def _build_log_post(self, x, y, prior_family, prior_kwargs=None):
        prior_obj = (
            prior_family if isinstance(prior_family, Prior)
            else prior_family.get_prior(**(prior_kwargs or {}))
        )
        proto = self._bde.members[0]
        pm = _BivariateProbabilisticModel(
            model=proto, params=proto.params, prior=prior_obj, task=self.task,
            temperature=getattr(self, "temperature", 1.0),
        )
        return partial(pm.log_unnormalized_posterior, x=x, y=y)

    def _build_log_post_hierarchical(self, x, y):
        """See _HierarchicalBivariateProbabilisticModel's docstring. prior_family/prior_kwargs are
        not used in this branch -- the prior is fully determined by hyper_log_tau_mu/
        hyper_log_tau_sigma (plain attributes set by _fit_correlated below, same pattern as
        temperature/pretrain_eval_every)."""
        proto = self._bde.members[0]
        hm = _HierarchicalBivariateProbabilisticModel(
            model=proto, params=proto.params, task=self.task,
            temperature=getattr(self, "temperature", 1.0),
            hyper_log_tau_mu=getattr(self, "hyper_log_tau_mu", 0.0),
            hyper_log_tau_sigma=getattr(self, "hyper_log_tau_sigma", 1.0),
            tau_max=getattr(self, "tau_max", 1.0),
            tau_link=getattr(self, "tau_link", "bounded"),
        )
        return partial(hm.log_unnormalized_posterior, x=x, y=y)

    def _build_log_post_hierarchical_centered(self, x, y):
        """See _CenteredHierarchicalBivariateProbabilisticModel's docstring -- the CENTERED
        (w_i ~ Normal(0, tau_i) directly) empirical-baseline counterpart to
        _build_log_post_hierarchical above. Same hyper_log_tau_mu/hyper_log_tau_sigma attribute
        pattern."""
        proto = self._bde.members[0]
        hm = _CenteredHierarchicalBivariateProbabilisticModel(
            model=proto, params=proto.params, task=self.task,
            temperature=getattr(self, "temperature", 1.0),
            hyper_log_tau_mu=getattr(self, "hyper_log_tau_mu", 0.0),
            hyper_log_tau_sigma=getattr(self, "hyper_log_tau_sigma", 1.0),
            tau_max=getattr(self, "tau_max", 1.0),
        )
        return partial(hm.log_unnormalized_posterior, x=x, y=y)

    def _draw_samples(self, logpost, rng_keys_e, init_positions_e, L_e, step_e, sqrt_diag_e):
        """2026-10-01: optional exact kept-sample count, off by default (n_kept_samples unset/None
        -> bde's own _draw_samples, unchanged). bde's MileWrapper.sample_batched runs n_samples RAW
        chain steps and thins afterwards via `[::n_thinning]` (confirmed by direct source read and
        a toy call: num_samples=1000, thinning=10 -> 100 kept), so the kept count is
        ceil(n_samples / n_thinning) and only integer steps-per-kept-sample ratios are reachable.
        When the plain attribute n_kept_samples is set (by _fit_correlated, same pattern as
        temperature/pretrain_eval_every), n_thinning is IGNORED: the chain still runs exactly
        n_samples raw steps, and exactly n_kept_samples of them are kept at evenly spaced indices
        ending on the final step -- e.g. n_samples=125000, n_kept_samples=10000 keeps every 12th or
        13th step alternately (12.5 on average). Peak memory is the same as bde's own path, which
        also materializes every raw position before slicing."""
        n_kept = getattr(self, "n_kept_samples", None)
        if n_kept is None:
            return super()._draw_samples(logpost, rng_keys_e, init_positions_e, L_e, step_e, sqrt_diag_e)
        if not 0 < n_kept <= self.n_samples:
            raise ValueError(
                f"n_kept_samples ({n_kept}) must be in (0, n_samples={self.n_samples}].")
        positions_eT, _, _ = MileWrapper(logpost).sample_batched(
            rng_keys_e=rng_keys_e, init_positions_e=init_positions_e,
            num_samples=self.n_samples, thinning=1,
            L_e=L_e, step_e=step_e, sqrt_diag_e=sqrt_diag_e, store_states=True,
        )
        idx = (np.arange(1, n_kept + 1) * self.n_samples) // n_kept - 1
        return tree_map(lambda a: a[:, idx, ...], positions_eT)

    def _pretrain_kfold(self, x, y, k, optimizer, loss_obj):
        """(2026-11-02) Per-member K-fold pretraining, replacing bde's single shared 80/20 split
        (every member trained on the same (1 - val_split) n points and early-stopped on the same
        val_split n, bde's fixed train_test_split(random_state=42)). Member m is pretrained on all
        points outside fold m and early-stopped on fold m (kfold_val_indices(n, k)); with k = 5 the
        per-member sizes equal the default 80/20 (40 / 10 at n=50), so only WHICH points are held
        out changes. Requires n_members <= k (members use folds 0..n_members-1).

        Each member gets a FRESH builder with the same seed, so member m starts from exactly the
        initial weights it has in the default path (bde creates members lazily from self.seed;
        re-calling fit_members on one builder would instead start from already-trained weights).
        The pretraining settings (eval_every / min_delta / min_steps, optimizer, loss, epochs) are
        those of the default path. Afterwards the last builder is kept and its members' params,
        params_e (the MCLMC start, stacked) and history ("Model{i}" -> member i's own run) are set
        to the per-member results, i.e. the state fit_members would leave. Costs n_members
        pretrainings instead of one (each trains all members; only member m's result is kept).
        ASSUMPTION: members re-seeded by _fit_correlated's retry loop use the fold of their index
        within the new attempt."""
        n = int(x.shape[0])
        M = int(self.n_members)
        if M > k:
            raise ValueError(f"pretrain_kfold={k} needs n_members <= k, got n_members={M}")
        folds = kfold_val_indices(n, k)
        member_params, member_hist, val_idx = [], {}, []
        for m in range(M):
            self._build_bde()   # fresh builder, same seed -> same initial weights as the default path
            self._bde.eval_every = getattr(self, "pretrain_eval_every", 100)
            self._bde.min_delta = getattr(self, "pretrain_min_delta", 1e-4)
            self._bde.min_steps = getattr(self, "pretrain_min_steps", 500)
            va = folds[m]
            tr = np.setdiff1d(np.arange(n), va)
            _fit_members_explicit_split(self._bde, x[tr], y[tr], x[va], y[va], epochs=self.epochs,
                                        optimizer=optimizer, loss=loss_obj)
            member_params.append(self._bde.members[m].params)
            member_hist[f"Model{m}"] = self._bde.history[f"Model{m}"]
            val_idx.append(va)
        for m, mem in enumerate(self._bde.members):
            mem.params = member_params[m]
        self._bde.params_e = tree_map(lambda *ps: jnp.stack(ps, axis=0), *member_params)
        self._bde.history = member_hist
        self.pretrain_fold_val_idx_ = np.stack(val_idx) if len({len(v) for v in val_idx}) == 1 else val_idx

    # Derived from sklearn-contrib-bde 1.0.2 (Bde._fit), (c) 2025 Arvanitis Vyron, Aslanidis Angelos, Sommer Emanuel
    # and scikit-learn-contrib contributors; BSD 3-Clause License, see THIRD_PARTY_LICENSES.
    def fit(self, x, y):
        """Copied from Bde._fit (installed bde/bde.py), with the
        self.task.validate_loss(loss_obj) call dropped. See override note (4)
        in the module-level comment above for why.

        Overrides fit(), not _fit(): BdeRegressor.fit() calls super()._fit(x, y),
        and that super() is lexically anchored to BdeRegressor -- it always
        resolves to Bde._fit regardless of type(self), so an override of _fit
        placed between Bde and BdeRegressor in the MRO (i.e. here) would never
        actually be reached. Overriding fit() itself sidesteps that."""
        x_np, y_np = validate_fit_data(self, x, y)
        self.n_features_in_ = x_np.shape[1]
        y_prepared = self._prepare_targets(y_np)
        loss_obj = self.loss if isinstance(self.loss, BaseLoss) else BivariateGaussianNLL()
        resolved_prior_family = self._resolve_prior_family()

        x_checked = jnp.asarray(x_np)
        y_checked = jnp.asarray(y_prepared)

        self._resolved_step_size_init = (
            self.step_size_init if self.step_size_init is not None else self.lr
        )

        self._build_bde()
        # 2026-08-26: override bde's hardcoded eval_every=1/min_delta=1e-9 (BdeBuilder.__init__)
        # to match gaussian_predictor.py's pretraining cadence (eval_every=100, min_delta=1e-4),
        # plus a hard floor (min_steps) on how early training is allowed to stop at all -- see
        # _MinStepsEarlyStoppingCallback's docstring above for the failure mode this fixes. Set as
        # plain attributes on the builder (not sklearn-declared constructor params) since these are
        # set right before fit_members() reads them via _create_callback(), and defaulted here so
        # CorrelatedBdeRegressor still works if pretrain_eval_every/pretrain_min_steps were never
        # set on self (e.g. a future direct instantiation outside _fit_correlated below).
        self._bde.eval_every = getattr(self, "pretrain_eval_every", 100)
        self._bde.min_delta = getattr(self, "pretrain_min_delta", 1e-4)
        self._bde.min_steps = getattr(self, "pretrain_min_steps", 500)
        # 2026-09-07: optional exponential LR decay for pretraining, off by default (None ->
        # exact old constant-self.lr behavior). Motivated by a real n=10,000 diagnostic: pretraining
        # under the tuned lr (from optimize_bde.py's pre-2026-09-04-rewrite search, never
        # re-validated at this n_train) never triggers early stopping and val loss is still
        # improving, linearly, at the fixed BNN_EPOCHS cutoff -- i.e. pretraining is plausibly just
        # under-converged, not capacity-limited. bde's own pretraining is full-batch gradient
        # descent (confirmed via direct read of bde/bde_builder.py:_training_loop -- one pstep call
        # over the entire x_train/y_train per epoch, no minibatching), so there is no SGD-noise for
        # a decay schedule to smooth away; the point here is purely to allow faster early movement
        # (a higher initial lr) while still landing exactly on the already-tuned lr (assumed to be a
        # reasonable value near convergence, since it came from a real validation-loss search) by
        # the end of training. lr_decay_multiplier=None or falsy leaves self.lr untouched (constant,
        # matching every existing saved result exactly).
        lr_decay_multiplier = getattr(self, "lr_decay_multiplier", None)
        if lr_decay_multiplier:
            lr_for_opt = optax.exponential_decay(
                init_value=self.lr * lr_decay_multiplier,
                transition_steps=self.epochs,
                decay_rate=1.0 / lr_decay_multiplier,
                end_value=self.lr,
            )
        else:
            lr_for_opt = self.lr
        # 2026-11-02: pretrain_kfold (default None -> the unchanged single bde split below). When an
        # int K is set, each member is pretrained on its own fold split -- see _pretrain_kfold.
        pretrain_kfold = getattr(self, "pretrain_kfold", None)
        if pretrain_kfold:
            self._pretrain_kfold(x_checked, y_checked, int(pretrain_kfold),
                                 optax.adamw(lr_for_opt, self.weight_decay), loss_obj)
        else:
            self._bde.fit_members(
                x=x_checked, y=y_checked, optimizer=optax.adamw(lr_for_opt, self.weight_decay),
                epochs=self.epochs, loss=loss_obj,
            )

        # Diagnostic (2026-08-25): each member's PRETRAINED (pre-MCLMC) point-estimate prediction
        # on the training set it was just fit on -- captured here, before the prior/MCLMC phase
        # ever touches the weights, specifically to localize a "predicts one constant value"
        # failure to pretraining itself vs. something the MCLMC/prior phase does afterward.
        # self._bde.members[i].params holds member i's just-pretrained weights at this exact point
        # (fit_members mutates each member's own .params in place; MCLMC sampling below draws
        # separate position samples, it does not touch member.params again).
        self.pretrain_train_preds_ = np.stack(
            [np.asarray(m.forward(m.params, x_checked)) for m in self._bde.members], axis=0
        )  # (n_members, n_train, 5) -- raw 5-wide head output, standardized y-space, no MCMC axis
        # 2026-11-02: pretrain_eval_x (plain attribute, default None -> unchanged: pretrain_test_preds_ = None). When
        # set (standardized test inputs), the pretrained networks' outputs on it are captured here, at the same
        # point as pretrain_train_preds_ (before warmup / MCLMC), the same way fit_pretrained_ensemble evaluates
        # them (validate_predict_data, then each member's forward). Lets one BNN run also score "the BNN without
        # sampling" -- see fit_correlated(score_pretrained=True).
        _pre_x = getattr(self, "pretrain_eval_x", None)
        self.pretrain_test_preds_ = None if _pre_x is None else np.stack(
            [np.asarray(m.forward(m.params, jnp.asarray(validate_predict_data(self, _pre_x))))
             for m in self._bde.members], axis=0)  # (n_members, n_test, 5)

        # 2026-09-05: raw flattened pretrained WEIGHT vector per member (not the network's output
        # -- the actual weights), captured at the same point as pretrain_train_preds_ above (i.e.
        # before MCLMC/the prior ever touch them). Added to directly check Fortuin, Garriga-Alonso,
        # Ober, Wenzel, Ratsch, Turner, van der Wilk & Aitchison (2021/ICLR 2022, arXiv:2102.06571)
        # Sec 3's own methodology against THIS project's SGD-pretrained weights -- they found FCNN
        # weights trained via SGD are heavy-tailed relative to a Gaussian, which motivated their
        # heavy-tailed-prior recommendation; this lets that be checked directly rather than assumed
        # to transfer from their MNIST/CIFAR setting. ravel_pytree order matches every other
        # flattened-weight convention already used in this project (optimize_prior.py etc.).
        self.pretrain_weights_ = np.stack(
            [np.asarray(ravel_pytree(m.params)[0]) for m in self._bde.members], axis=0
        )  # (n_members, n_weights)

        # 2026-09-04: bde's own BdeBuilder._training_loop already records a full per-member,
        # per-epoch train/val loss history internally (self._bde.history, a dict keyed
        # "Model0".."Model{n_members-1}", each {epoch, trainloss, valloss, stop_epoch} -- confirmed
        # by direct source read of the installed package's bde/bde_builder.py) but nothing in this
        # project previously read it back out. Captured here (same place/pattern as
        # pretrain_train_preds_ above) so a caller can plot the pretraining loss curve directly --
        # added in response to a hyperparameter-search question (does a tuned lr/wd actually
        # diverge/overtrain during pretraining, not just "what's the final validation NLL") that
        # the search objective alone couldn't answer. Converted to plain numpy/int (off the
        # jnp/traced values bde stores) so this survives being pickled/passed around normally.
        self.pretrain_loss_history_ = {
            k: {"epoch": np.asarray(v["epoch"]), "trainloss": np.asarray(v["trainloss"]),
                "valloss": np.asarray(v["valloss"]), "stop_epoch": int(v["stop_epoch"])}
            for k, v in self._bde.history.items()
        }

        # 2026-11-02: pretrain_only (default False -> unchanged). When True, fit() stops here: the members hold
        # their pretrained weights and the pretraining diagnostics above are set, but there is no warmup and no
        # MCLMC. Used by fit_pretrained_ensemble ("the BNN without sampling"); because it is this same fit() up
        # to this point, the networks are exactly those a full fit with the same arguments starts MCLMC from.
        if bool(getattr(self, "pretrain_only", False)):
            return self

        # 2026-09-09: hierarchical (jointly-sampled weight-prior-scale) path -- see
        # _HierarchicalBivariateProbabilisticModel's docstring. hierarchical_prior/tau_init are
        # plain attributes set by _fit_correlated below, same pattern as temperature/
        # pretrain_eval_every above. prior_family/prior_kwargs are ignored entirely in this branch.
        hierarchical = bool(getattr(self, "hierarchical_prior", False))
        # 2026-09-10: CENTERED empirical-baseline variant (see
        # _CenteredHierarchicalBivariateProbabilisticModel's docstring) -- added to
        # test whether non-centering actually helps THIS project's MCLMC setup. Mutually exclusive
        # with `hierarchical` above (checked by fit_correlated()'s own sentinel resolution).
        hierarchical_centered = bool(getattr(self, "hierarchical_centered", False))
        tau_init = float(getattr(self, "tau_init", 0.5))
        tau_max = float(getattr(self, "tau_max", 1.0))
        # 2026-09-11: tau_link selects which positivity link's INVERSE is needed to hit tau_init
        # exactly at u_tau_init -- only applies to the non-centered class (hierarchical_centered
        # always uses the bounded/Phi link, unchanged). "bounded" (default, production): link is
        # tau_i = tau_max*Phi(u_tau_i), inverted via Phi^-1 (jax.scipy.special.ndtri); tau_init/
        # tau_max clipped away from the open interval's endpoints since Phi^-1(0)/Phi^-1(1) are
        # +-inf. Default tau_init=0.5 is deliberately tau_max's MIDPOINT so the default init lands
        # at u_tau_init=0 exactly, the hyperprior's own mode. "softplus" (2026-09-11, reinstated as
        # an explicit opt-in -- see _HierarchicalBivariateProbabilisticModel.__init__'s docstring):
        # link is tau_i = softplus(u_tau_i), inverted via the numerically-stable
        # u = tau_init + log1p(-exp(-tau_init)) (softplus^-1, valid for tau_init > 0).
        tau_link = getattr(self, "tau_link", "bounded")
        if hierarchical and tau_link == "softplus":
            u_tau_init = float(tau_init + np.log1p(-np.exp(-tau_init)))
        elif hierarchical or hierarchical_centered:
            u_tau_init = float(jax.scipy.special.ndtri(np.clip(tau_init / tau_max, 1e-6, 1 - 1e-6)))
        else:
            u_tau_init = None
        if hierarchical_centered:
            logpost_one = self._build_log_post_hierarchical_centered(x_checked, y_checked)
            # Centered init position per member: w = pretrained physical weights directly (no
            # raw/tau decomposition to invert -- this IS the physical weight), u_tau = u_tau_init
            # everywhere. Starts MCLMC from the same point the non-hierarchical path would, same
            # as the non-centered variant's own init.
            params_e = tree_map(
                lambda *ps: jnp.stack(ps, axis=0), *[m.params for m in self._bde.members]
            )
            u_tau_init_e = tree_map(lambda p: jnp.full_like(p, u_tau_init), params_e)
            self._bde.params_e = {"w": params_e, "u_tau": u_tau_init_e}
        elif hierarchical:
            logpost_one = self._build_log_post_hierarchical(x_checked, y_checked)
            # Augmented init position per member: raw = pretrained physical weights / tau_init
            # (non-centered -- see class docstring), u_tau = u_tau_init. tau_init=1.0 (this
            # function's default, set by _fit_correlated) makes raw_init == the pretrained weights
            # exactly, i.e. this starts MCLMC from the SAME point the non-hierarchical path would.
            # Set via bde.warmup.warmup_bde's own `getattr(bde, "params_e", None)` override hook
            # (confirmed via direct source read, 2026-09-09) -- when present, it's used as the
            # per-member init position in place of stacking each member.params directly, and is
            # fully pytree-generic (confirmed both warmup_bde and MileWrapper.sample_batched only
            # ever use tree_map/tree_leaves on positions, no hardcoded structure assumption beyond
            # what init_positions_e's own pytree provides).
            # 2026-09-10: PER-WEIGHT tau (see class docstring) -- u_tau_init_e now mirrors
            # raw_init_e's exact pytree/shape (one u_tau_init per weight, not one scalar per
            # member) via tree_map(jnp.full_like), same tau_init value everywhere at init since
            # every weight starts from the same hyperprior.
            params_e = tree_map(
                lambda *ps: jnp.stack(ps, axis=0), *[m.params for m in self._bde.members]
            )
            # 2026-09-13 "MAP init" (opt-in, default off -- see _map_init_hierarchical_tau's
            # docstring): replaces the fixed-tau_init scheme above (same tau_init for every weight
            # regardless of its own pretrained magnitude) with a per-weight MAP solve for u_tau_i
            # given that weight's own already-pretrained, FIXED physical value -- does not change
            # the prior itself, only the starting point handed to MCLMC.
            if bool(getattr(self, "map_init_tau", False)):
                raw_init_e, u_tau_init_e = _map_init_hierarchical_tau(
                    params_e, hyper_log_tau_mu=self.hyper_log_tau_mu,
                    hyper_log_tau_sigma=self.hyper_log_tau_sigma, tau_link=tau_link,
                    tau_max=tau_max, u_tau_init_guess=u_tau_init,
                )
            else:
                raw_init_e = tree_map(lambda p: p / tau_init, params_e)
                u_tau_init_e = tree_map(lambda p: jnp.full_like(p, u_tau_init), params_e)
            self._bde.params_e = {"raw": raw_init_e, "u_tau": u_tau_init_e}
        else:
            logpost_one = self._build_log_post(
                x_checked, y_checked, resolved_prior_family, self.prior_kwargs
            )
        init_positions_e, tuned = self._warmup_sampler(logpost_one)

        num_chains = tree_leaves(init_positions_e)[0].shape[0]
        rng_keys_e = self._generate_rng_keys(num_chains)
        L_e, step_e, sqrt_diag_e = self._normalize_tuned_parameters(tuned, num_chains)

        # 2026-09-08: MCLMC's own tuned per-member sampler hyperparameters -- L (expected number of
        # integrator steps between full momentum decoherence, MCLMC's continuous analogue of HMC's
        # trajectory length / NUTS's tree depth), step_size, and sqrt_diag_cov (per-dimension
        # diagonal preconditioner). bde's own base `Bde._fit` (confirmed via direct source read,
        # installed bde/bde.py lines 446-450) computes these via warmup and immediately discards
        # them as local variables -- never stored on self, never exposed after fit() returns. Same
        # discard pattern existed here (this method is copied from Bde._fit) until now. Captured for
        # the n_train-scaling mixing-failure investigation: lets a caller
        # directly compare what the adaptation converged to at n_train=100 (good mixing) vs.
        # n_train=10,000 (bad mixing) under an otherwise-identical config, rather than only inferring
        # sampler behavior indirectly from R-hat/ESS/prediction-space spread.
        self.tuned_L_ = np.asarray(L_e)
        self.tuned_step_size_ = np.asarray(step_e)
        self.tuned_sqrt_diag_cov_ = np.asarray(sqrt_diag_e)

        samples = self._draw_samples(
            logpost_one, rng_keys_e, init_positions_e, L_e, step_e, sqrt_diag_e
        )
        # 2026-09-09/10: hierarchical path -- `samples` here is the augmented pytree
        # (n_members, n_samples, ...) per leaf, {"raw", "u_tau"} for the non-centered variant or
        # {"w", "u_tau"} for the centered one, u_tau PER-WEIGHT in both cases (mirrors the
        # weight-shaped pytree's exact shape -- see the relevant class docstring). Convert to
        # physical weights BEFORE storing into positions_eT_, so every downstream consumer
        # (predict()'s BdePredictor, which calls member.forward(position, x) directly and is NOT
        # overridden here) sees exactly the same pytree structure/shape it always has -- nothing
        # downstream of this point needs to know a hierarchical scale was ever sampled.
        if hierarchical or hierarchical_centered:
            # u_tau's own posterior (as physical tau=|u_tau|) is kept separately, as a per-weight
            # MEAN/STD SUMMARY across MCMC samples (n_members, n_weights) each -- NOT the full
            # (n_members, n_samples, n_weights) per-draw array, which at this project's scale
            # (2245 weights x 10000 samples x 3 chains x 4 bytes) would be ~270MB per run for a
            # quantity whose main use is diagnostic plotting, not a downstream consumer needing
            # every draw. Flattens each pytree leaf's trailing (per-weight) dims and concatenates
            # along the last axis -- avoids needing ravel_pytree's unravel machinery, which isn't
            # naturally vmappable across the leading (n_members, n_samples) axes already present
            # on every leaf. Shared by both hierarchical variants (both use the "u_tau" key
            # identically).
            def _flatten_trailing(pytree_of_arrays):
                leaves = tree_leaves(pytree_of_arrays)
                flat_leaves = [leaf.reshape(leaf.shape[0], leaf.shape[1], -1) for leaf in leaves]
                return jnp.concatenate(flat_leaves, axis=-1)  # (n_members, n_samples, n_weights)

            # 2026-09-11 BUG FIX: this used to hardcode tau=|u_tau| here (the value in BOTH this
            # diagnostic tau summary AND, more seriously, the ACTUAL PHYSICAL-WEIGHT
            # RECONSTRUCTION two lines below) -- a leftover from the |u_tau_i| link that was never
            # updated when the link changed to tau_max*Phi(u_tau_i) (2026-09-10) or when the
            # tau_link="softplus" option was added (2026-09-11). log_prior/log_likelihood (what
            # MCLMC actually samples FROM) already used the correct tau_link-selected transform the
            # whole time -- so the CHAIN itself sampled from the right density -- but every
            # downstream consumer of `samples`/`positions_eT_` (predict(), R-hat/ESS, every saved
            # prediction) for the NON-CENTERED hierarchical path was reconstructing physical
            # weights with the WRONG transform, silently substituting a different (and for the
            # current bounded/Phi link, differently-SHAPED and non-bounded) function of the same
            # raw/u_tau draws. Affects every non-centered hierarchical run since the Phi link
            # landed (2026-09-10) through this fix. The CENTERED path (samples["w"] used directly, no reconstruction)
            # was never affected by the second bug -- only its own tau_flat diagnostic display was.
            _link_fn = (
                (lambda t: jax.nn.softplus(t)) if tau_link == "softplus"
                else (lambda t: tau_max * jax.scipy.stats.norm.cdf(t))
            )
            tau_flat = _link_fn(_flatten_trailing(samples["u_tau"]))  # (n_members, n_samples, n_weights)
            self.hierarchical_tau_mean_ = np.asarray(jnp.mean(tau_flat, axis=1))  # (n_members, n_weights)
            self.hierarchical_tau_std_ = np.asarray(jnp.std(tau_flat, axis=1))    # (n_members, n_weights)

            if hierarchical_centered:
                # Centered: samples["w"] IS the physical weight already -- no raw*tau product.
                samples = samples["w"]
            else:
                samples = tree_map(
                    lambda t, r: _link_fn(t) * r, samples["u_tau"], samples["raw"]
                )
        self.positions_eT_ = tree_map(lambda arr: jnp.asarray(arr), samples)
        self.is_fitted_ = True
        return self


def _summarize_correlated_raw(raw, ystd, ymu, theta_true=None):
    """Shared by the test-set and training-set diagnostic paths in fit_correlated() below --
    destandardizes a raw (n_good, n_samples, n_pts, 5) joint BDE draw array into per-point
    mean/aleatoric/epistemic/corr (pooled across chains+samples) and a per-chain decomposition,
    and -- if theta_true is given -- the posterior-predictive log-density at the true parameter
    per point (its negative mean is the average per-item predictive loss, the same NLPD
    convention used for log_prob_true elsewhere in this project)."""
    mu_std = raw[..., 0:2]                                              # (n_good, n_samples, n_pts, 2)
    s_std = np.asarray(jax.nn.softplus(jnp.asarray(raw[..., 2:4]))) + 1e-6
    corr_std = np.asarray(jnp.tanh(jnp.asarray(raw[..., 4])))           # (n_good, n_samples, n_pts)

    mean_all = mu_std * ystd + ymu    # (n_good, n_samples, n_pts, 2)
    scale_all = s_std * ystd          # correlation is scale-invariant -- no destandardization needed

    chain_mean = mean_all.mean(axis=1)                                    # (n_good, n_pts, 2)
    chain_aleatoric = np.sqrt((scale_all ** 2).mean(axis=1))              # (n_good, n_pts, 2)
    chain_within_spread = np.sqrt(np.maximum(mean_all.var(axis=1), 0))    # (n_good, n_pts, 2)
    chain_corr = corr_std.mean(axis=1)                                    # (n_good, n_pts)

    K_mu = mean_all.reshape(-1, mean_all.shape[-2], mean_all.shape[-1])      # (K, n_pts, 2)
    K_s = scale_all.reshape(-1, scale_all.shape[-2], scale_all.shape[-1])    # (K, n_pts, 2)
    K_corr = corr_std.reshape(-1, corr_std.shape[-1])                       # (K, n_pts)
    mu, sd, corr = jnp.asarray(K_mu), jnp.asarray(K_s), jnp.asarray(K_corr)

    out = dict(
        mean=np.asarray(mu.mean(0)), aleatoric=np.sqrt(np.asarray((sd**2).mean(0))),
        epistemic=np.sqrt(np.asarray(mu.var(0))), corr=np.asarray(corr.mean(0)),
        chain_mean=chain_mean, chain_aleatoric=chain_aleatoric,
        chain_within_spread=chain_within_spread, chain_corr=chain_corr,
    )

    if theta_true is not None:
        theta_j = jnp.asarray(np.asarray(theta_true))
        log_comp = metrics.bivariate_normal_logpdf(theta_j[None], mu, sd, corr)  # (K, n_pts)
        out["log_prob_true"] = np.asarray(
            jax.scipy.special.logsumexp(log_comp, axis=0) - jnp.log(mu.shape[0])
        )

    return out


def _summarize_pretrained(raw, raw_train, pretrain_train, ystd, ymu, theta_test_np, yb_np, seed,
                          tarp_lo, tarp_hi, tarp_n_draws, tarp_seeds):
    """(2026-11-02) Summary of a set of pretrained networks (no MCLMC) in fit_correlated's bnn_raw schema: raw /
    raw_train are (n_members, 1, n_points, 5) -- one "draw" per member -- so the predictive is the equal-weight
    mixture of the members' Gaussians and the epistemic term is the spread of their means. The block below is
    fit_correlated's own post-sampling summary (as reproduced in fit_rasmussen_gibbs_hierarchical), with the
    per-member draw axis of length 1; TARP / SBC draws use the same keys (PRNGKey(seed), PRNGKey(seed + 999))."""
    n_good = raw.shape[0]
    test_summary = _summarize_correlated_raw(raw, ystd, ymu, theta_true=theta_test_np)
    train_summary = _summarize_correlated_raw(raw_train, ystd, ymu, theta_true=yb_np)
    chain_mean, chain_aleatoric = test_summary["chain_mean"], test_summary["chain_aleatoric"]
    chain_within_spread = test_summary["chain_within_spread"]

    mu_std = raw[..., 0:2]
    s_std = np.asarray(jax.nn.softplus(jnp.asarray(raw[..., 2:4]))) + 1e-6
    corr_std = np.asarray(jnp.tanh(jnp.asarray(raw[..., 4])))
    mean_all, scale_all = mu_std * ystd + ymu, s_std * ystd

    n_samples_axis = mean_all.shape[1]
    take_per_chain = max(1, min(n_samples_axis, tarp_n_draws // max(n_good, 1)))
    key_chain = jax.random.PRNGKey(seed + 999)
    chain_sample_idx = jax.random.choice(
        key_chain, n_samples_axis, (take_per_chain,), replace=(n_samples_axis < take_per_chain))
    _, sk_chain = jax.random.split(key_chain)
    chain_theta_samples = np.asarray(jax.vmap(metrics.sample_posterior_predictive)(
        jax.random.split(sk_chain, n_good), jnp.asarray(mean_all)[:, chain_sample_idx],
        jnp.asarray(scale_all)[:, chain_sample_idx], jnp.asarray(corr_std)[:, chain_sample_idx]))

    mu = jnp.asarray(mean_all.reshape(-1, mean_all.shape[-2], mean_all.shape[-1]))
    sd = jnp.asarray(scale_all.reshape(-1, scale_all.shape[-2], scale_all.shape[-1]))
    corr = jnp.asarray(corr_std.reshape(-1, corr_std.shape[-1]))
    key = jax.random.PRNGKey(seed)
    K = mu.shape[0]
    # Always tarp_n_draws predictive draws: with K = n_members (12) < tarp_n_draws, fit_correlated's min(tarp_n_draws, K)
    # would give only K draws per test dataset; drawing members with replacement samples the K-Gaussian mixture.
    idx = jax.random.choice(key, K, (tarp_n_draws,), replace=(K < tarp_n_draws))
    key, sk = jax.random.split(key)
    theta_samples = np.asarray(metrics.sample_posterior_predictive(sk, mu[idx], sd[idx], corr[idx]))
    f = metrics.compute_tarp_f_stats(theta_samples, theta_test_np, tarp_lo, tarp_hi, tarp_seeds)
    sbc_rank = metrics.sbc_marginal_rank(theta_samples, theta_test_np)
    out = dict(
        mean=test_summary["mean"], aleatoric=test_summary["aleatoric"], epistemic=test_summary["epistemic"],
        corr=test_summary["corr"], log_prob_true=test_summary["log_prob_true"], sbc_rank=sbc_rank,
        theta_samples=theta_samples, chain_mean=chain_mean, chain_aleatoric=chain_aleatoric,
        chain_within_spread=chain_within_spread, chain_corr=test_summary["chain_corr"],
        between_chain_std=np.std(chain_mean, axis=0),
        within_chain_std=np.sqrt(np.mean(chain_aleatoric ** 2 + chain_within_spread ** 2, axis=0)),
        chain_theta_samples=chain_theta_samples,
        train_mean=train_summary["mean"], train_corr=train_summary["corr"],
        train_log_prob_true=train_summary["log_prob_true"],
        pretrain_train_mean=np.asarray(pretrain_train[..., 0:2]) * ystd + ymu,
        pretrain_train_corr=np.asarray(jnp.tanh(jnp.asarray(pretrain_train[..., 4]))),
    )
    return out, f


def fit_pretrained_ensemble(Xb, yb, X_test, theta_test, seed, n_members, hidden_layers, epochs, patience,
                            validation_split, lr, weight_decay, tarp_lo, tarp_hi, tarp_n_draws, tarp_seeds,
                            quiet=True, pretrain_eval_every=100, pretrain_min_steps=500, activation="soft_sign",
                            warmup_steps=50000, n_samples=10000, n_thinning=10,
                            desired_energy_var_start=0.5, desired_energy_var_end=0.1):
    """(2026-11-02) "The BNN without sampling": exactly fit_correlated's pretraining -- same standardisation, same
    CorrelatedBdeRegressor arguments and attributes, same seed, the same fit() code via pretrain_only=True -- with
    no warmup and no MCLMC, then the pretrained networks scored on the test set. Pretraining does not use the
    weight prior, so for a given (n, repeat, seed) these are the networks every stock-MILE BNN variant
    (N(0,1), data-free width, tempered) starts MCLMC from. warmup_steps / n_samples / n_thinning / energy targets
    are only passed to the constructor (unused) so its arguments match fit_correlated's exactly.

    Returns (ens_raw, ens_f, single_raw, single_f, diag): the equal-weight mixture of all finite members, and
    member 0 alone (a single Gaussian network; epistemic = 0), both in fit_correlated's bnn_raw schema, plus
    pretrain_loss_history / pretrain_weights and member bookkeeping. Members with non-finite test or training
    outputs are dropped from the ensemble and counted in diag (no retry: pretraining is the deterministic part)."""
    if quiet:
        _silence_bde_progress()
    Xb_np = np.asarray(Xb).reshape(len(Xb), -1)
    yb_np = np.asarray(yb)
    Xte_np = np.asarray(X_test).reshape(len(X_test), -1)
    theta_test_np = np.asarray(theta_test)
    Xtr, ytr, Xmu, Xstd, ymu, ystd = _standardize(Xb_np, yb_np)
    Xte = (Xte_np - Xmu) / Xstd

    bde_kwargs = dict(
        hidden_layers=hidden_layers, n_members=n_members, loss=BivariateGaussianNLL(),
        epochs=epochs, validation_split=validation_split, lr=lr, weight_decay=weight_decay,
        warmup_steps=warmup_steps, n_samples=n_samples, n_thinning=n_thinning, patience=patience,
        desired_energy_var_start=desired_energy_var_start, desired_energy_var_end=desired_energy_var_end,
        activation=activation, prior_family="standardnormal", prior_kwargs=None,
    )
    reg = CorrelatedBdeRegressor(**dict(bde_kwargs, seed=seed))
    reg.pretrain_eval_every = pretrain_eval_every
    reg.pretrain_min_steps = pretrain_min_steps
    reg.temperature = 1.0
    reg.lr_decay_multiplier = None
    reg.hierarchical_prior = False
    reg.hierarchical_centered = False
    reg.pretrain_kfold = None
    reg.pretrain_only = True
    reg.fit(x=Xtr, y=ytr)

    x_te = jnp.asarray(validate_predict_data(reg, Xte))
    raw_test = np.stack([np.asarray(m.forward(m.params, x_te)) for m in reg._bde.members])[:, None]
    pretrain_train = np.asarray(reg.pretrain_train_preds_)                  # (n_members, n_train, 5)
    raw_train = pretrain_train[:, None]
    finite = np.isfinite(raw_test).all(axis=(1, 2, 3)) & np.isfinite(raw_train).all(axis=(1, 2, 3))
    if not finite.any():
        raise RuntimeError("fit_pretrained_ensemble: every pretrained member is non-finite")
    if not finite[0]:
        raise RuntimeError("fit_pretrained_ensemble: member 0 (the single network) is non-finite")
    good = np.where(finite)[0]
    ens_raw, ens_f = _summarize_pretrained(raw_test[good], raw_train[good], pretrain_train[good], ystd, ymu,
                                           theta_test_np, yb_np, seed, tarp_lo, tarp_hi, tarp_n_draws, tarp_seeds)
    single_raw, single_f = _summarize_pretrained(raw_test[:1], raw_train[:1], pretrain_train[:1], ystd, ymu,
                                                 theta_test_np, yb_np, seed, tarp_lo, tarp_hi, tarp_n_draws, tarp_seeds)
    hist = reg.pretrain_loss_history_
    diag = dict(
        n_members_used=int(len(good)), member_finite=finite, members_failed=int((~finite).sum()),
        pretrain_loss_history=[hist[f"Model{i}"] for i in good],
        pretrain_stop_epoch=np.array([hist[f"Model{i}"]["stop_epoch"] for i in range(len(finite))]),
        pretrain_weights=np.asarray(reg.pretrain_weights_)[good],
    )
    return ens_raw, ens_f, single_raw, single_f, diag


def fit_mean_ensemble(Xb, yb, X_test, seed, n_members, hidden_layers, epochs, patience, validation_split, lr,
                      weight_decay, quiet=True, pretrain_eval_every=100, pretrain_min_steps=500, activation="soft_sign",
                      warmup_steps=50000, n_samples=10000, n_thinning=10,
                      desired_energy_var_start=0.5, desired_energy_var_end=0.1):
    """(2026-11-03) MEAN-ONLY predictor: fit_pretrained_ensemble's pretraining (the Gaussian predictor = the BNN
    without sampling) with exactly the same standardisation, CorrelatedBdeRegressor arguments / attributes, seed, split,
    optimiser (full-batch AdamW, lr, wd), early stopping (eval every pretrain_eval_every, patience, min_delta 1e-4,
    pretrain_min_steps floor) and network, EXCEPT a 2-wide output head (mu_sigma, mu_rho; pretrain_output_dim=2) and
    MeanSquaredErrorLoss instead of BivariateGaussianNLL. Member i is initialised from seed + i as in bde, so its
    hidden layers start from the same draw as the Gaussian predictor's member i ASSUMING bde's per-layer init does not
    depend on the output width (the last layer necessarily differs). No sampling, no predictive distribution.

    Returns dict: member_mean (n_members, n_test, 2) and member_train_mean (n_members, n_train, 2) in theta units,
    member_finite, pretrain_stop_epoch (-1 = hit the epoch cap), pretrain_loss_history (per member)."""
    if quiet:
        _silence_bde_progress()
    Xb_np = np.asarray(Xb).reshape(len(Xb), -1)
    yb_np = np.asarray(yb)
    Xte_np = np.asarray(X_test).reshape(len(X_test), -1)
    Xtr, ytr, Xmu, Xstd, ymu, ystd = _standardize(Xb_np, yb_np)
    Xte = (Xte_np - Xmu) / Xstd
    bde_kwargs = dict(
        hidden_layers=hidden_layers, n_members=n_members, loss=MeanSquaredErrorLoss(),
        epochs=epochs, validation_split=validation_split, lr=lr, weight_decay=weight_decay,
        warmup_steps=warmup_steps, n_samples=n_samples, n_thinning=n_thinning, patience=patience,
        desired_energy_var_start=desired_energy_var_start, desired_energy_var_end=desired_energy_var_end,
        activation=activation, prior_family="standardnormal", prior_kwargs=None,
    )
    reg = CorrelatedBdeRegressor(**dict(bde_kwargs, seed=seed))
    reg.pretrain_eval_every = pretrain_eval_every
    reg.pretrain_min_steps = pretrain_min_steps
    reg.temperature = 1.0
    reg.lr_decay_multiplier = None
    reg.hierarchical_prior = False
    reg.hierarchical_centered = False
    reg.pretrain_kfold = None
    reg.pretrain_only = True
    reg.pretrain_output_dim = 2
    reg.fit(x=Xtr, y=ytr)
    x_te = jnp.asarray(validate_predict_data(reg, Xte))
    out_test = np.stack([np.asarray(m.forward(m.params, x_te)) for m in reg._bde.members])   # (M, n_test, 2)
    out_train = np.asarray(reg.pretrain_train_preds_)                                         # (M, n_train, 2)
    assert out_test.shape[-1] == 2 and out_train.shape[-1] == 2, (out_test.shape, out_train.shape)
    finite = np.isfinite(out_test).all(axis=(1, 2)) & np.isfinite(out_train).all(axis=(1, 2))
    hist = reg.pretrain_loss_history_
    return dict(
        member_mean=out_test * ystd + ymu, member_train_mean=out_train * ystd + ymu, member_finite=finite,
        pretrain_stop_epoch=np.array([hist[f"Model{i}"]["stop_epoch"] for i in range(len(finite))]),
        pretrain_loss_history=[hist[f"Model{i}"] for i in range(len(finite))],
    )


def _predict_raw_per_member(reg, x):
    """reg.predict(x, raw=True), one ensemble member at a time. bde's own BdePredictor vmaps the
    same forward function (members[0].forward) over (member, sample) in a single call, so every
    intermediate activation carries the member axis too -- at 10,000 kept samples x 10,000
    training points x 3 members one hidden layer is 17.88 GiB, which ran out of GPU memory
    (2026-10-01, job 59163366, bnn_vanilla_extrasteps at n=10,000). Same computation and same
    (n_members, n_samples, n_points, 5) result, a third of the peak memory; this is what
    fit_rasmussen_gibbs_hierarchical's own per-member evaluation already does. Deliberately NOT
    jax.jit-wrapped: bde's predictor runs this vmap un-jitted, and a jitted version was measured
    to differ from it by float32 rounding (mean 5e-7, max 4.6e-4), whereas this matches it
    bit-for-bit."""
    x_j = jnp.asarray(validate_predict_data(reg, x))
    forward = reg._bde.members[0].forward
    one_member = lambda pos_T: jax.vmap(lambda p: forward(p, x_j))(pos_T)
    n_members = tree_leaves(reg.positions_eT_)[0].shape[0]
    return np.stack(
        [np.asarray(one_member(tree_map(lambda a: a[m], reg.positions_eT_))) for m in range(n_members)],
        axis=0)


def _fit_correlated(X_train_std, y_std, X_test_std, bde_kwargs, seed, max_seed_tries,
                     min_good_members=3, pretrain_eval_every=100, pretrain_min_steps=500,
                     temperature=1.0, lr_decay_multiplier=None,
                     hierarchical_prior=False, hyper_log_tau_mu=0.0, hyper_log_tau_sigma=1.0,
                     tau_init=0.5, tau_max=1.0, hierarchical_centered=False, tau_link="bounded",
                     map_init_tau=False, n_kept_samples=None, pretrain_kfold=None, score_pretrained=False):
    """Analogous to _fit_one_param but for a single joint CorrelatedBdeRegressor fitting (sigma,
    rho) together. Accumulates finite members ACROSS separately-reseeded attempts into a running
    pool -- each attempt refits a fresh bde_kwargs['n_members']-member ensemble with a new seed,
    and every member that comes back finite (on both the test set AND the training set -- a member
    finite on one but not the other would be an inconsistent read of the same underlying MCMC
    samples) is kept in the pool regardless of which attempt produced it. Stops as soon as the pool
    reaches min_good_members, or once max_seed_tries attempts have run.

    2026-08-25, changed per explicit correction: the original version reseeded and refit the WHOLE
    ensemble on every attempt and only accepted an attempt if it alone came back with
    >= min_good_members members simultaneously finite and well-mixed (max R-hat <= a threshold) --
    "keep running 3-chain runs until it gets a run that has 3 that work." The intended behavior is
    "keep running chains until you have 3," i.e. accumulate good chains one attempt's worth at a
    time rather than discarding an attempt's finite members just because the same attempt also
    produced a non-finite one. Mixing (R-hat) is no longer part of the accept/continue decision at
    all, also per explicit request -- fit_correlated() still computes and reports it on the final
    pooled chains as a diagnostic, it just doesn't gate which members are kept or how long the
    search runs.

    Returns (raw_kept, raw_train_kept, pretrain_train_kept, pretrain_loss_history_kept,
    pretrain_weights_kept, tuned_L_kept, tuned_step_size_kept, tuned_sqrt_diag_cov_kept,
    seeds_used, n_good, extra, hierarchical_tau_mean_kept, hierarchical_tau_std_kept) where
    raw_kept/raw_train_kept have shape
    (n_good, n_samples, n_{test,train}, 5)
    -- one row per accumulated finite member, in the order collected -- last axis is [mu_sigma,
    mu_rho, log_s_sigma, log_s_rho, corr_raw] in standardized y-space. pretrain_train_kept has shape
    (n_good, n_train, 5) -- same last-axis layout, but the PRETRAINED (pre-MCLMC) point estimate
    per kept member (see CorrelatedBdeRegressor.fit's pretrain_train_preds_), no samples axis since
    it's a single point estimate, not a posterior draw. pretrain_loss_history_kept (2026-09-04) is a
    list of n_good {epoch, trainloss, valloss, stop_epoch} dicts, one per kept member -- see
    CorrelatedBdeRegressor.fit's pretrain_loss_history_ docstring. pretrain_weights_kept (2026-09-05)
    is (n_good, n_weights) -- the raw flattened pretrained weight vector per kept member (ravel_pytree
    order), see CorrelatedBdeRegressor.fit's pretrain_weights_ docstring. tuned_L_kept/
    tuned_step_size_kept are (n_good,) and tuned_sqrt_diag_cov_kept is (n_good, n_weights) --
    MCLMC's own tuned per-member sampler hyperparameters, one row per kept member (2026-09-08), see
    CorrelatedBdeRegressor.fit's tuned_L_/tuned_step_size_/tuned_sqrt_diag_cov_ docstring. seeds_used
    is a list of the trial
    seed that produced each kept member, parallel to raw_kept's first axis (chains in the pool can
    come from different seeds). extra is a dict of retry-level bookkeeping (seed_attempts,
    members_tried_across_attempts, members_failed_across_attempts) covering every member from every
    attempt made, not just the ones kept -- the "keep the fail rate logged" requirement.

    pretrain_eval_every/pretrain_min_steps (2026-08-26) are forwarded to each CorrelatedBdeRegressor
    instance as plain attributes (read by CorrelatedBdeRegressor.fit(), see there) -- override bde's
    own hardcoded pretraining early-stopping cadence to match gaussian_predictor.py's schedule plus
    a hard floor on how early pretraining can stop at all; see _MinStepsEarlyStoppingCallback.

    temperature (2026-08-28, default 1.0) is likewise forwarded as a plain attribute, read by
    _build_log_post via _BivariateProbabilisticModel -- see that class's docstring for exactly what
    gets tempered (likelihood only, Delaunoy-et-al-equivalent) and why. Applies only to the MCLMC
    phase, not pretraining -- pretraining's loss (BivariateGaussianNLL) is untouched by this.

    lr_decay_multiplier (2026-09-07, default None -- exact old constant-lr pretraining behavior)
    is likewise forwarded as a plain attribute, read by CorrelatedBdeRegressor.fit() -- see there
    for the exponential-decay schedule this builds and the n=10,000 under-convergence diagnostic
    that motivated it. Applies only to pretraining, not MCLMC.

    hierarchical_prior/hyper_log_tau_mu/hyper_log_tau_sigma/tau_init (2026-09-09, default False --
    exact old fixed-prior-scale behavior): when True, bypasses bde_kwargs['prior_family'] entirely
    and instead samples a PER-WEIGHT weight-prior scale jointly with the weights via
    _HierarchicalBivariateProbabilisticModel (see its docstring for the non-centered
    parameterization and hyperprior -- 2026-09-10: rebuilt from an earlier isotropic/shared-scale
    version). Forwarded as plain attributes, same pattern as
    temperature/pretrain_eval_every. Returns extra good_hierarchical_tau_mean/
    good_hierarchical_tau_std (n_good, n_weights) arrays -- the per-weight posterior mean/std of
    physical tau=exp(log_tau) across MCMC samples per kept member (NOT the full per-sample array,
    a deliberate storage-size decision -- see CorrelatedBdeRegressor.fit's docstring) when True,
    else (None, None).
    """
    good_raw, good_raw_train, good_pretrain_train, good_pretrain_loss_history = [], [], [], []
    good_pretrain_weights, seeds_used = [], []
    good_pretrain_test = []   # 2026-11-02, score_pretrained only
    good_tuned_L, good_tuned_step_size, good_tuned_sqrt_diag_cov = [], [], []
    _any_hierarchical = hierarchical_prior or hierarchical_centered
    good_hierarchical_tau_mean = [] if _any_hierarchical else None
    good_hierarchical_tau_std = [] if _any_hierarchical else None
    seed_attempts = 0
    members_tried_across_attempts = 0
    members_failed_across_attempts = 0

    trial_seed = seed
    while len(good_raw) < min_good_members and seed_attempts < max_seed_tries:
        seed_attempts += 1
        reg = CorrelatedBdeRegressor(**dict(bde_kwargs, seed=trial_seed))
        reg.pretrain_eval_every = pretrain_eval_every
        reg.pretrain_min_steps = pretrain_min_steps
        reg.temperature = temperature
        reg.lr_decay_multiplier = lr_decay_multiplier
        reg.hierarchical_prior = hierarchical_prior
        reg.hierarchical_centered = hierarchical_centered
        reg.hyper_log_tau_mu = hyper_log_tau_mu
        reg.hyper_log_tau_sigma = hyper_log_tau_sigma
        reg.tau_init = tau_init
        reg.tau_max = tau_max
        reg.tau_link = tau_link
        reg.map_init_tau = map_init_tau
        reg.n_kept_samples = n_kept_samples  # see CorrelatedBdeRegressor._draw_samples
        reg.pretrain_kfold = pretrain_kfold  # 2026-11-02, see CorrelatedBdeRegressor._pretrain_kfold
        reg.pretrain_eval_x = X_test_std if score_pretrained else None   # 2026-11-02, see CorrelatedBdeRegressor.fit
        reg.fit(x=X_train_std, y=y_std)
        raw = _predict_raw_per_member(reg, X_test_std)         # (n_members, n_samples, n_test, 5)
        raw_train = _predict_raw_per_member(reg, X_train_std)  # (n_members, n_samples, n_train, 5)
        pretrain_train = reg.pretrain_train_preds_                  # (n_members, n_train, 5)

        finite = np.array([
            np.isfinite(raw[m]).all() and np.isfinite(raw_train[m]).all() for m in range(raw.shape[0])
        ])
        members_tried_across_attempts += raw.shape[0]
        members_failed_across_attempts += raw.shape[0] - int(finite.sum())

        for m in np.flatnonzero(finite):
            good_raw.append(raw[m])
            good_raw_train.append(raw_train[m])
            good_pretrain_train.append(pretrain_train[m])
            good_pretrain_loss_history.append(reg.pretrain_loss_history_[f"Model{m}"])
            good_pretrain_weights.append(reg.pretrain_weights_[m])
            if score_pretrained:
                good_pretrain_test.append(reg.pretrain_test_preds_[m])
            good_tuned_L.append(reg.tuned_L_[m])
            good_tuned_step_size.append(reg.tuned_step_size_[m])
            good_tuned_sqrt_diag_cov.append(reg.tuned_sqrt_diag_cov_[m])
            if _any_hierarchical:
                good_hierarchical_tau_mean.append(reg.hierarchical_tau_mean_[m])
                good_hierarchical_tau_std.append(reg.hierarchical_tau_std_[m])
            seeds_used.append(trial_seed)
            if len(good_raw) >= min_good_members:
                break

        trial_seed += 1

    n_good = len(good_raw)
    extra = dict(seed_attempts=seed_attempts,
                 members_tried_across_attempts=members_tried_across_attempts,
                 members_failed_across_attempts=members_failed_across_attempts)
    if score_pretrained and n_good > 0:   # 2026-11-02: (n_good, n_test, 5); popped by fit_correlated, never saved in diag
        extra["pretrain_test_kept"] = np.stack(good_pretrain_test, axis=0)
    if n_good == 0:
        raise RuntimeError(f"exhausted {max_seed_tries} seed attempts with 0 finite members")

    raw_kept = np.stack(good_raw, axis=0)
    raw_train_kept = np.stack(good_raw_train, axis=0)
    pretrain_train_kept = np.stack(good_pretrain_train, axis=0)
    # good_pretrain_loss_history: list of n_good {epoch, trainloss, valloss, stop_epoch} dicts, one
    # per kept member (2026-09-04) -- kept as a plain list, not stacked, since each member can have
    # a different length (early-stopped at a different epoch); see CorrelatedBdeRegressor.fit's
    # pretrain_loss_history_ docstring above for provenance.
    # pretrain_weights_kept (2026-09-05): (n_good, n_weights) -- every member has the SAME weight
    # count (fixed architecture), so this stacks cleanly unlike the loss history above.
    pretrain_weights_kept = np.stack(good_pretrain_weights, axis=0)
    # tuned_L/step_size/sqrt_diag_cov_kept (2026-09-08): MCLMC's own tuned per-member sampler
    # hyperparameters -- see CorrelatedBdeRegressor.fit's docstring for what each one is/why this
    # was added. L and step_size stack to (n_good,) scalars; sqrt_diag_cov stacks to (n_good,
    # n_weights) same as pretrain_weights_kept above (same fixed architecture across members).
    tuned_L_kept = np.stack(good_tuned_L, axis=0)
    tuned_step_size_kept = np.stack(good_tuned_step_size, axis=0)
    tuned_sqrt_diag_cov_kept = np.stack(good_tuned_sqrt_diag_cov, axis=0)
    # 2026-09-10: (n_good, n_weights) per-weight posterior mean/std of physical tau=|u_tau|
    # across MCMC samples, per kept member, only when hierarchical_prior or hierarchical_centered
    # is True -- None otherwise. Every member has the SAME weight count (fixed architecture), so
    # this stacks cleanly.
    hierarchical_tau_mean_kept = (
        np.stack(good_hierarchical_tau_mean, axis=0) if _any_hierarchical else None
    )
    hierarchical_tau_std_kept = (
        np.stack(good_hierarchical_tau_std, axis=0) if _any_hierarchical else None
    )
    return (raw_kept, raw_train_kept, pretrain_train_kept, good_pretrain_loss_history,
            pretrain_weights_kept, tuned_L_kept, tuned_step_size_kept, tuned_sqrt_diag_cov_kept,
            seeds_used, n_good, extra, hierarchical_tau_mean_kept, hierarchical_tau_std_kept)


def _array_normal_prior(loc, scale):
    """Array-valued-prior workaround (see fit_correlated's own docstring for the underlying bde
    bug this works around) -- needed here because fit_gibbs_hierarchical's burn-in sweeps
    construct CorrelatedBdeRegressor directly (to read back positions_eT_), not via
    fit_correlated's own inline workaround."""
    return Prior(
        name="Normal",
        f_init=lambda key, shape, dtype=jnp.float32: jnp.zeros(shape, dtype=dtype),  # never invoked
        log_prior=log_prior_normal(loc=jnp.asarray(loc), scale=jnp.asarray(scale)),
    )


def _flatten_trailing_pytree(pytree_of_arrays, leading_shape):
    """Flattens every leaf's trailing (per-weight) dims and concatenates across leaves, keeping
    leading_shape (e.g. (n_members, n_samples)) intact."""
    leaves = tree_leaves(pytree_of_arrays)
    flat_leaves = [np.asarray(leaf).reshape(*leading_shape, -1) for leaf in leaves]
    return np.concatenate(flat_leaves, axis=-1)


def fit_gibbs_hierarchical(Xb, yb, X_test, theta_test, seed,
        n_members, hidden_layers, epochs, patience, validation_split,
        n_samples, warmup_steps, lr, weight_decay, n_thinning,
        desired_energy_var_start, desired_energy_var_end,
        max_seed_tries, rhat_threshold,
        tarp_lo, tarp_hi, tarp_n_draws, tarp_seeds,
        min_good_members=3, pretrain_eval_every=100, pretrain_min_steps=500,
        ig_a=2.0, ig_b=1.0, tau_init=1.0,
        n_burnin_sweeps=15, burnin_warmup_steps=2000, burnin_n_samples=500, burnin_n_thinning=1):
    """Block Gibbs sampler for the per-weight hierarchical prior -- see scripts/bde_gibbs_test.py's
    module docstring for the full algorithm description, the CENTERED-parameterization rationale,
    and the citation correction (classical normal-inverse-gamma conjugate model; Neal (1996)'s
    BNN-ARD-via-Gibbs as the closest -- still unverified from memory -- precedent, NOT Park &
    Casella's Bayesian Lasso). Factored out of that script (2026-09-13) into this library function
    so scripts/run_experiment.py's per-model sweep can call the same, already-validated logic
    rather than duplicating it.

    Two phases: (1) n_burnin_sweeps cheap sweeps (n_members=1, burnin_warmup_steps/burnin_n_samples,
    small relative to production) alternating ordinary MCLMC weight-sampling at a FIXED per-weight
    tau (reusing the existing prior_family="normal" array-valued-prior machinery, no new sampling
    code) with an EXACT closed-form conjugate draw for tau given the just-sampled weights
    (tau_i^2 | weights ~ InverseGamma(ig_a + K/2, ig_b + sum(w_i_k^2)/2), K = pooled draws that
    sweep); (2) one production-quality bnn.fit_correlated() call (the given n_members/warmup_steps/
    n_samples -- the FULL production budget, not the burn-in one) at the converged (fixed) tau.

    SIMPLIFICATION, same as the standalone script: each burn-in sweep reruns pretraining+MCLMC from
    scratch rather than continuing one evolving chain.

    Returns (bnn_raw, bnn_f, diag) -- SAME schema as fit_correlated's own return (diag gains three
    extra keys: gibbs_tau_final (n_weights,), gibbs_tau_history (n_burnin_sweeps+1, n_weights),
    gibbs_ig_a/gibbs_ig_b) -- so callers that already know how to consume a fit_correlated() result
    (e.g. scripts/run_experiment.py's _bnn_variant_arrays) need only a small extra branch for the
    tau diagnostics, not a separate code path."""
    Xb_np = np.asarray(Xb).reshape(len(Xb), -1)
    yb_np = np.asarray(yb)
    Xtr, ytr, Xmu, Xstd, ymu, ystd = _standardize(Xb_np, yb_np)

    probe = CorrelatedBdeRegressor(
        hidden_layers=hidden_layers, n_members=1, loss=BivariateGaussianNLL(),
        epochs=1, validation_split=validation_split, lr=lr, weight_decay=weight_decay,
        warmup_steps=200, n_samples=10, n_thinning=1, patience=1,
        desired_energy_var_start=desired_energy_var_start, desired_energy_var_end=desired_energy_var_end,
        activation="soft_sign", prior_family="standardnormal", prior_kwargs=None, seed=seed,
    )
    probe.pretrain_eval_every, probe.pretrain_min_steps = 1, 1
    probe.hierarchical_prior = False
    probe.fit(jnp.asarray(Xtr), jnp.asarray(ytr))
    n_weights = ravel_pytree(probe._bde.members[0].params)[0].shape[0]

    tau = jnp.full((n_weights,), tau_init)
    tau_history = [np.asarray(tau)]

    for sweep in range(n_burnin_sweeps):
        reg = CorrelatedBdeRegressor(
            hidden_layers=hidden_layers, n_members=1, loss=BivariateGaussianNLL(),
            epochs=epochs, validation_split=validation_split, lr=lr, weight_decay=weight_decay,
            warmup_steps=burnin_warmup_steps, n_samples=burnin_n_samples, n_thinning=burnin_n_thinning,
            patience=patience,
            desired_energy_var_start=desired_energy_var_start, desired_energy_var_end=desired_energy_var_end,
            activation="soft_sign", prior_family=_array_normal_prior(jnp.zeros(n_weights), tau),
            prior_kwargs=None, seed=seed + sweep,
        )
        reg.pretrain_eval_every = pretrain_eval_every
        reg.pretrain_min_steps = pretrain_min_steps
        reg.temperature = 1.0
        reg.lr_decay_multiplier = None
        reg.hierarchical_prior = False
        reg.fit(jnp.asarray(Xtr), jnp.asarray(ytr))

        n_kept_samples = tree_leaves(reg.positions_eT_)[0].shape[1]
        w_samples = _flatten_trailing_pytree(reg.positions_eT_, leading_shape=(1, n_kept_samples))
        w_samples = w_samples.reshape(-1, n_weights)
        K = w_samples.shape[0]

        sumsq = np.sum(w_samples ** 2, axis=0)
        a_post = ig_a + K / 2.0
        b_post = ig_b + sumsq / 2.0

        key = jax.random.PRNGKey(1000 * (seed + 1) + sweep)
        g = jax.random.gamma(key, a_post, shape=(n_weights,)) / jnp.asarray(b_post)
        tau2_new = 1.0 / g
        tau = jnp.sqrt(jnp.maximum(tau2_new, 1e-12))
        tau_history.append(np.asarray(tau))

    tau_final = tau
    tau_history = np.stack(tau_history, axis=0)

    bnn_raw, bnn_f, diag = fit_correlated(
        Xb, yb, X_test, theta_test, seed,
        n_members, hidden_layers, epochs, patience, validation_split,
        n_samples, warmup_steps, lr, weight_decay, n_thinning,
        desired_energy_var_start, desired_energy_var_end,
        max_seed_tries, rhat_threshold,
        tarp_lo, tarp_hi, tarp_n_draws, tarp_seeds, quiet=True,
        min_good_members=min_good_members,
        pretrain_eval_every=pretrain_eval_every, pretrain_min_steps=pretrain_min_steps,
        prior_family="normal", prior_kwargs={"loc": np.zeros(n_weights), "scale": np.asarray(tau_final)},
        temperature=1.0, lr_decay_multiplier=None,
    )
    diag = dict(diag)
    diag["gibbs_tau_final"] = np.asarray(tau_final)
    diag["gibbs_tau_history"] = tau_history
    diag["gibbs_ig_a"] = ig_a
    diag["gibbs_ig_b"] = ig_b
    return bnn_raw, bnn_f, diag


def _warm_step_size_L_adapt(kernel, dim, state, params, adaptive_state, streaming_avg,
                             tune1_steps, tune2_steps, desired_energy_var, trust_in_estimate,
                             num_effective_samples, rng_key):
    """Reimplementation of bde.sampler.warmup.make_L_step_size_adaptation's tune1/tune2 step-size
    adaptation loop, with BOTH the step-size EMA accumulator (`adaptive_state = (time, x_average,
    step_size_max)`) AND the position-variance streaming average used for `L` (`streaming_avg`)
    threaded through as explicit input/output instead of being reset to a blank slate at the start
    of every call -- confirmed via direct source read that bde's own version resets both internally,
    unconditionally, every time it's invoked.

    History (2026-09-17, same day): first implementation only carried the OUTPUT (L, step_size)
    across retunes, not adaptive_state -- crashed on real n=100 compute (job 58465410, 2 of 3
    members non-finite), traced at toy scale to step_size whipsawing (0.10 -> 2.35 -> 4.02 -> 4.79
    -> 0.27) consistent with an EMA rebuilt from scratch every call with only ~200 raw steps. Fixed
    by carrying adaptive_state -- toy-scale behavior improved (no more whipsaw), but a SECOND real
    n=100 run (job 58468761) still crashed the same way. Direct question raised whether `L` was
    being tracked too -- it was not: `streaming_avg` was still reset every call, meaning `L` was
    being re-estimated from the variance of just `tune2_steps` (50 by default) samples across all
    `dim` (2245) weight dimensions, EVERY retune, independently -- a far noisier quantity than
    step_size's own scalar EMA, and very plausibly the dominant remaining instability source. Now
    carried, but with the incoming weight DECAYED by the same `decay_rate` used for the step-size
    EMA before continuing accumulation (see call site) -- a deliberate compromise, not literally
    forever-growing memory (which would let L "freeze," barely responding to new data after enough
    retunes, since old weight would come to dominate the plain running average) nor a full reset
    (too noisy at only 50 samples/window) -- ASSUMPTION, not yet independently verified beyond
    fixing the observed real failures, that this specific decay choice is well-calibrated; it reuses
    the existing EMA time-constant only because it's already the project's own reference decay, not
    because it's known to be correct for L's position-variance statistic specifically.
    Each retune's target is technically a different fixed-tau snapshot, so carried variance mixes
    statistics from slightly different targets -- judged an acceptable price for stability given
    tau moves gradually, but flagged directly as a real approximation.

    desired_energy_var is a single constant (no start->end anneal) -- consistent with
    fit_rasmussen_gibbs_hierarchical's own docstring point 5 on why a warm-started retune shouldn't
    re-anneal.

    Returns (state, params (with L and step_size updated), new adaptive_state, new streaming_avg)."""
    decay_rate = (num_effective_samples - 1.0) / (num_effective_samples + 1.0)

    def predictor(previous_state, params, adaptive_state, step_key):
        time, x_average, step_size_max = adaptive_state
        next_state, info = kernel(params.sqrt_diag_cov)(
            rng_key=step_key, state=previous_state, L=params.L, step_size=params.step_size)
        success, state, step_size_max, energy_change = _bde_warmup.handle_nans(
            previous_state, next_state, params.step_size, step_size_max, info.energy_change)
        xi = jnp.square(energy_change) / (dim * desired_energy_var) + 1e-8
        weight = jnp.exp(-0.5 * jnp.square(jnp.log(xi) / (6.0 * trust_in_estimate)))
        x_average = decay_rate * x_average + weight * (xi / jnp.power(params.step_size, 6.0))
        time = decay_rate * time + weight
        step_size_new = jnp.power(x_average / time, -1.0 / 6.0)
        step_size_new = (step_size_new < step_size_max) * step_size_new + \
            (step_size_new > step_size_max) * step_size_max
        params_new = params._replace(step_size=step_size_new)
        return state, params_new, (time, x_average, step_size_max), success

    def step(carry, xs):
        mask, step_key = xs
        state, params, adaptive_state, streaming_avg = carry
        state, params, adaptive_state, success = predictor(state, params, adaptive_state, step_key)
        x = ravel_pytree(state.position)[0]
        streaming_avg = streaming_average_update(
            current_value=jnp.array([x, jnp.square(x)]),
            previous_weight_and_average=streaming_avg,
            weight=(1 - mask) * success * params.step_size,
            zero_prevention=mask,
        )
        return (state, params, adaptive_state, streaming_avg), None

    n_total = tune1_steps + tune2_steps
    keys = jax.random.split(rng_key, max(n_total, 1))
    mask = 1 - jnp.concatenate((jnp.zeros(tune1_steps), jnp.ones(tune2_steps)))
    # Decay the incoming streaming-average weight before continuing -- see docstring's "History"
    # note on why this is neither a full reset nor unbounded memory.
    prev_weight, prev_avg = streaming_avg
    streaming_avg0 = (prev_weight * decay_rate, prev_avg)
    (state, params, adaptive_state, (streaming_weight, average)), _ = jax.lax.scan(
        step, (state, params, adaptive_state, streaming_avg0), (mask, keys))
    streaming_avg = (streaming_weight, average)

    L = params.L
    if tune2_steps != 0:
        x_average, x_squared_average = average[0], average[1]
        variances = x_squared_average - jnp.square(x_average)
        L = jnp.sqrt(jnp.sum(variances))
    params = params._replace(L=L)
    return state, params, adaptive_state, streaming_avg


def fit_rasmussen_gibbs_hierarchical(Xb, yb, X_test, theta_test, seed,
        n_members, hidden_layers, epochs, patience, validation_split,
        n_samples, warmup_steps, lr, weight_decay, n_thinning,
        desired_energy_var_start, desired_energy_var_end,
        max_seed_tries, rhat_threshold,
        tarp_lo, tarp_hi, tarp_n_draws, tarp_seeds,
        min_good_members=3, pretrain_eval_every=100, pretrain_min_steps=500,
        ig_a=2.0, ig_b=1.0, tau_init=1.0, tau_save_stride=10,
        retune_every_samples=None, retune_tune1_steps=200, retune_tune2_steps=50,
        retune_tune3_steps=0, resample_tau=True,
        eb_global_scale=False, eb_adapt_segments=30, eb_tau_floor=1e-3,
        eb_per_step=False, tau_init_from_pretrained=False,
        eb_block=None, eb_block_damping=0.5, eb_block_max_factor=2.0, eb_block_tau0=None):
    """Single continuously-evolving-chain block Gibbs sampler for the per-weight hierarchical
    prior, added 2026-09-15 as a direct-request test of Rasmussen (1995, NIPS 8, "A Practical
    Monte Carlo Implementation of Bayesian Learning") / Neal (1996)'s classical scheme: interleave
    an MCMC step for the weights with a closed-form conjugate Gibbs update for their prior scale
    EVERY step, within ONE evolving chain -- not fit_gibbs_hierarchical's restart-per-sweep
    approximation (see that function's own docstring for the "SIMPLIFICATION" it explicitly flags,
    which this function exists specifically to avoid).

    Checked against the literature (Rasmussen, 1996): Rasmussen's paper alternates HMC weight updates with a closed-form Gamma-conjugate
    Gibbs update of the precision hyperparameters, INTERLEAVED EVERY ITERATION within one
    continuously-evolving chain -- exactly the structure implemented here (MCLMC in place of HMC,
    Inverse-Gamma-on-variance in place of Gamma-on-precision, otherwise the same pattern).

    ALGORITHM, precisely:
      1. Pretrain (Adam, same as every other variant) -- one-time, independent of tau.
      2. Warm up ONCE via bde's own mclmc_find_L_and_step_size, at tau_init, starting from the
         pretrained weights -- exactly as every other variant's initial tuning. L/step_size/
         sqrt_diag_cov from this ONE warmup are then held FIXED for the rest of the run --
         ASSUMPTION, flagged directly: this does NOT re-tune the integrator as tau evolves, a real
         scope-limiting simplification relative to what a fully continuously-adaptive scheme might
         do (re-tuning mid-chain is a much bigger undertaking and was out of scope here).
      3. ONE continuous MCLMC chain per member, n_samples*n_thinning total steps, implemented as a
         nested jax.lax.scan (outer over kept samples, inner over the n_thinning raw steps between
         them) so tau updates happen at EVERY raw step (the literal reading of "interleaved every
         step") while memory stays bounded to O(n_samples), not O(n_samples*n_thinning). At every
         raw step: (a) closed-form tau_i^2 | current weights ~ InverseGamma(ig_a+1/2,
         ig_b+w_i^2/2) -- K=1, using only the CURRENT single weight draw, the literal per-step
         reading of the classical scheme (no pooling across multiple draws, unlike
         fit_gibbs_hierarchical's sweep-level K); (b) rebuild the log-density with the new tau
         (reusing _array_normal_prior/_BivariateProbabilisticModel unchanged -- the CENTERED
         weight~N(0,tau_i^2) prior is IDENTICAL to what fit_gibbs_hierarchical's own MCLMC-at-fixed-
         tau step already uses, only the update cadence differs); (c) recompute (logdensity,
         logdensity_grad) at the CURRENT position under the new target (momentum is left untouched
         -- that continuity, not a fresh momentum draw, is what makes this one evolving chain rather
         than a restart); (d) one MCLMC kernel step.
      4. n_members independent chains (own pretraining seed, own warmup, own continuous run) for
         the same reason every other variant in this project needs an ensemble: R-hat/ESS and the
         epistemic decomposition require genuinely independent chains -- tau is NOT shared/pooled
         across members, each member's tau evolves on its own.
      5. OPTIONAL (2026-09-17, off by default via retune_every_samples=None, for backward
         compatibility): periodic WARM-STARTED re-tuning of (L, step_size), added because tau's
         own drift changes the prior's curvature (~1/tau^2) and therefore the target's geometry,
         while (L, step_size) from the one-time warmup above are otherwise frozen for the entire
         run regardless. When retune_every_samples is set, the continuous chain is split into
         segments of that many KEPT samples each; after every segment, a fresh
         mclmc.MCLMCAdaptationState is retuned STARTING FROM the segment's own current (L,
         step_size, sqrt_diag_cov) rather than bde.sampler.warmup.mclmc_find_L_and_step_size's own
         cold-reset defaults -- direct source read confirmed that function unconditionally discards
         whatever (L, step_size, sqrt_diag_cov) is passed in, so warm-starting requires calling the
         lower-level bde.sampler.warmup.make_L_step_size_adaptation/make_adaptation_L closures
         directly instead (both DO use their given `params` as the starting point, unmodified).
         Because warm-starting begins near the already-correct value, retune_tune1_steps/
         retune_tune2_steps/retune_tune3_steps are deliberately MUCH smaller than the cold warmup's
         own warmup_steps -- ASSUMPTION, not yet independently verified beyond a
         smoke test, that a short warm-started retune converges close to what a full cold retune at
         the same tau would give. retune_tune3_steps defaults to 0 (skip the ESS-based L refinement
         entirely on each retune, keep only the tune1/tune2 position-variance-based L estimate) --
         a deliberate simplification: bde's own make_adaptation_L docstring states its FFT-based ESS
         estimate needs "around 10 effective samples" to be accurate, which a short retune budget is
         not expected to reliably provide. The desired_energy_var schedule for each retune is held
         CONSTANT at desired_energy_var_end (not re-annealed from desired_energy_var_start) -- this
         is a refinement of an already-tuned chain, not a fresh cold tune, so no anneal is used.
         Every retune call within one member reuses the SAME (retune_tune1_steps,
         retune_tune2_steps, retune_tune3_steps) shapes so JAX JIT-compiles the retune function
         once per member and reuses that compiled executable for every subsequent segment in that
         member's run, rather than recompiling per call.
         CORRECTION (2026-09-17, same day): an earlier version of this retune called
         bde.sampler.warmup.make_L_step_size_adaptation directly, which internally resets its own
         EMA accumulator (time, x_average, step_size_max) to a blank slate at the start of EVERY
         call regardless of what params is passed in -- meaning only the OUTPUT (L, step_size) was
         actually being warm-started, not the internal statistic that makes the adaptation stable.
         This caused a real failure on real n=100 compute (job 58465410: 2 of 3 members went
         non-finite mid-chain). Fixed via _warm_step_size_L_adapt, a local reimplementation of the
         same tune1/tune2 loop that threads the EMA accumulator through explicitly across retunes
         (see that function's own docstring for the full reasoning, including why L's own
         position-variance estimate is deliberately NOT carried the same way).

    resample_tau (2026-10-01, default True -- unchanged behavior): when False, tau is never
    redrawn and stays at tau_init for every weight for the whole run, so with tau_init=1.0 the
    prior is exactly N(0,1) on every weight and there is no Gibbs step at all -- the same target
    as fit_correlated's default prior, on this function's sampler. Added to test whether the
    periodic (L, step_size) retune (point 5) helps a plain N(0,1) BNN, i.e. whether its benefit
    has anything to do with tau sampling. Everything else (pretraining, one-time warmup at
    tau_init, the retune machinery, n_samples = KEPT count, no retry on a failed member) is
    unchanged; ig_a/ig_b are then unused.

    eb_global_scale (2026-10-02, default False -- unchanged behavior): EMPIRICAL-BAYES estimate
    of ONE prior scale shared by every weight, by Monte Carlo EM, followed by ordinary sampling
    under the resulting fixed prior N(0, tau*^2). Requires resample_tau=False and
    retune_every_samples set. This OPTIMIZES the prior scale against the training data, unlike
    resample_tau=True, which only redraws per-weight scales from a fixed IG(ig_a, ig_b)
    hyperprior and leaves the weights' marginal prior unchanged.
      - Adaptation phase (unsaved), eb_adapt_segments segments, run BEFORE the kept samples and
        after the one-time warmup at tau_init. Each segment: (E) retune_every_samples * n_thinning
        MCLMC steps at the current tau; (M) tau^2 <- mean of w^2 over ALL weights and over the
        kept positions of the segment's SECOND HALF (the first half is discarded as
        re-equilibration after the previous tau change) -- the closed-form maximizer of
        E[log N(w; 0, tau^2 I)], i.e. the exact EM update for a zero-mean isotropic Gaussian
        prior's scale; then the same warm-started (L, step_size) retune as point 5, at the new
        tau. Cost: eb_adapt_segments * (retune_every_samples * n_thinning + retune_tune1_steps +
        retune_tune2_steps) extra steps per member.
      - Production phase: tau is FROZEN at its final value and the chain continues exactly as the
        resample_tau=False retune path does (n_samples kept, retuned every retune_every_samples),
        so the kept samples target one fixed posterior.
    ASSUMPTIONS, flagged directly: (a) each ensemble member estimates its OWN tau from its own
    chain (members run sequentially here); a single prior would strictly be shared -- check
    rasmussen_eb_tau_history for agreement across members. (b) No damping and no convergence
    test: eb_adapt_segments is a fixed budget, and where the posterior is prior-dominated
    (small n_train) EM contracts slowly, so tau may NOT have converged -- always inspect the
    history before reading a result. (c) "Second half of the segment" is an unvalidated choice of
    re-equilibration window. (d) Biases share the scale with the weights, as under N(0,1).
    (e) eb_tau_floor only guards against a collapse to exactly 0. Empirical Bayes uses the data
    twice (to set the prior and to form the posterior), so it can make the posterior too narrow.
    diag gains rasmussen_eb_tau_history / _eb_L_history / _eb_step_size_history, each
    (n_good, 1 + eb_adapt_segments) with the pre-adaptation value first, and
    rasmussen_eb_adapt_segments; rasmussen_L_history's first entry is then the value entering
    production, not the cold-warmup value.

    eb_per_step / tau_init_from_pretrained (2026-10-02, both default False -- unchanged
    behavior): the PER-STEP empirical-Bayes global scale, the design adopted after the
    block-EM mode above was judged far too expensive (it adds eb_adapt_segments * 1,250 unsaved
    steps; with 150 segments that is 187,500). No adaptation phase and no extra steps at all:
      - tau_init_from_pretrained=True: the one-time warmup runs under N(0, tau0^2) with
        tau0 = sqrt(mean w^2) of THAT member's Adam-pretrained weights (tau_init is ignored).
      - eb_per_step=True: at EVERY raw chain step, before the MCLMC step, the one scale shared by
        all weights is reset to sqrt(mean w^2) of the CURRENT weights (floored at eb_tau_floor) --
        the same slot where resample_tau=True draws per-weight scales, so the chain length is
        exactly n_samples * n_thinning (+ retune steps only if retune_every_samples is set).
        Requires resample_tau=False and eb_global_scale=False. The scale is never frozen.
    ASSUMPTIONS / things NOT verified, flagged directly: (a) because tau is a deterministic
    function of the current weights, the prior's gradient is -d * w / |w|^2 (d = n_weights),
    i.e. the chain behaves as if the prior were p(w) ~ |w|^(-d) -- a scale-free, improper prior
    on the weight norm; this is our own derivation, not checked against a
    reference, and it is exact only for an exact sampler (MCLMC here is unadjusted). (b) There is
    no guarantee the weight norm is stable under that prior at small n_train -- inspect
    rasmussen_eb_tau_history (drift, collapse toward eb_tau_floor, or growth) before reading any
    result. (c) tau is per member, from that member's own chain. (d) Without
    retune_every_samples, (L, step_size) stay as tuned at tau0 while tau moves. (e) A single
    weight vector's mean square stands in for a posterior average (relative noise ~sqrt(2/d),
    about 3% at d=2245).
    diag gains rasmussen_eb_tau_history (n_good, 1 + n_samples): tau0 followed by the scale at
    each kept sample, and rasmussen_eb_tau_init (n_good,).

    eb_block (2026-10-04, default None -- unchanged behavior): BLOCK empirical-Bayes prior scale,
    "iso" (one scale for all weights) or "layer" (one scale per layer; a layer's W and b share
    it). Replaces eb_per_step's every-step reset, which couples the prior to the current sample
    (the chain then behaves as if the prior were |w|^-d, improper), with an update between
    blocks during which the prior is FIXED. Requires resample_tau=False, eb_per_step=False,
    eb_global_scale=False and retune_every_samples set; a block is one retune segment
    (retune_every_samples * n_thinning raw steps), so no steps are added beyond the retune's.
      - Initial prior: each group's scale = RMS of that group's Adam-pretrained weights (floored
        at eb_tau_floor). The one-time warmup/tuning runs under it; no update happens there.
        eb_block_tau0 (2026-10-05, default None -- unchanged behavior): when set, every group
        starts at this scale instead. With eb_block_damping=0.0 the scale then never moves, so
        the run is a FIXED-width retuned chain that still records each block's moments
        (rasmussen_ebblock_gamma_history / _tau_target_history) -- used by
        scripts/eb_fixed_width_residual.py to measure E[|w|^2] / (d tau^2) - 1 at a chosen tau.
      - After each block: per-weight mean m_i and variance v_i over ALL raw steps of the block
        (running sums of w - w_block_start, so not only the kept samples). Per group c of k_c
        weights with current scale tau_c:
            gamma_c = k_c - sum_i v_i / tau_c^2      (clamped to [1, k_c])
            tau_target_c^2 = sum_i m_i^2 / gamma_c
        i.e. MacKay's (1992) evidence-framework update alpha = gamma / |w_MP|^2 with the Laplace
        mean/covariance replaced by the block's sample mean/variance. Then
            log tau_c += clip(eb_block_damping * (log tau_target_c - log tau_c),
                              +/- log eb_block_max_factor),
        floored at eb_tau_floor, followed by the usual warm retune of (L, step_size) under the
        new prior.
    ASSUMPTIONS, flagged directly: (a) the update is our own proposal, written from
    memory of MacKay's formula and not checked against the paper; it shares its fixed point with
    the plain EM update tau^2 = E[w^2] but that equivalence was derived by hand, not
    tested. (b) eb_block_damping=0.5, eb_block_max_factor=2.0 and the gamma clamp have no source.
    (c) A block that mixes poorly underestimates v_i, so gamma is overestimated and tau_target is
    biased low. (d) All kept samples are returned, including those drawn before the scale has
    settled -- early blocks are under a different prior than late ones; restrict to late blocks
    in analysis via rasmussen_ebblock_tau_history. (e) The update after the final block is
    recorded but affects no kept sample. (f) Scales are per member, from that member's own chain.
    (g) Data used twice -- check coverage.
    diag gains rasmussen_ebblock_mode, _tau_history (n_good, 1 + n_segments, n_groups; first
    entry the initial scale), _gamma_history (n_good, n_segments, n_groups; UNCLAMPED),
    _tau_target_history (same shape; before damping/cap), _n_gamma_clamped and _n_step_capped
    (n_good,), _group_sizes, _damping, _max_factor, and rasmussen_eb_tau_init (n_good,; the
    all-weight pretrained RMS).

    SIMPLIFICATION relative to fit_gibbs_hierarchical/fit_correlated's retry logic, flagged
    directly: no retry-on-a-fresh-seed-if-a-member-fails loop. A member is kept if its full raw
    trace is finite, dropped otherwise -- appropriate for an exploratory single run, not carried
    over from the production pipeline's pooled-across-reseeds accumulation.

    Storage: rasmussen_tau_trace is (n_good, n_samples // tau_save_stride, n_weights) -- every
    tau_save_stride'th KEPT sample's full per-weight tau vector, per chain (2026-09-15, direct
    request: "save every 10th tau vector" to bound the ~270MB/run full-trace size flagged
    elsewhere in this project for per-weight per-draw arrays). hierarchical_tau_mean/_std are
    repurposed here (like fit_gibbs_hierarchical) but now carry a REAL posterior std, not a point
    estimate: mean/std of each weight's tau ACROSS the n_samples kept draws of the single
    continuous chain, since tau genuinely varies sample-to-sample here (unlike
    fit_gibbs_hierarchical's fixed-tau production fit).

    Returns (bnn_raw, bnn_f, diag) -- same schema as fit_correlated's own return (diag gains
    rasmussen_tau_trace/rasmussen_ig_a/rasmussen_ig_b, plus (if retune_every_samples is set)
    rasmussen_retune_every_samples/_tune1_steps/_tune2_steps/_tune3_steps and
    rasmussen_L_history/rasmussen_step_size_history -- (n_good, 1+n_segments) per-member traces of
    L/step_size after each retune, first entry always the initial cold-warmup value; bnn_raw is
    unaffected)."""
    Xb_np = np.asarray(Xb).reshape(len(Xb), -1)
    yb_np = np.asarray(yb)
    Xtr, ytr, Xmu, Xstd, ymu, ystd = _standardize(Xb_np, yb_np)
    theta_test_np = np.asarray(theta_test)
    Xte = (np.asarray(X_test).reshape(len(X_test), -1) - Xmu) / Xstd

    Xtr_j, ytr_j, Xte_j = jnp.asarray(Xtr), jnp.asarray(ytr), jnp.asarray(Xte)

    raw_list, raw_train_list = [], []
    pretrain_train_list, pretrain_weights_list, pretrain_loss_history = [], [], []
    tuned_L_list, tuned_step_size_list, tuned_sqrt_diag_cov_list = [], [], []
    tau_mean_list, tau_std_list, tau_trace_list = [], [], []
    retune_L_history_list, retune_step_size_history_list = [], []
    eb_tau_history_list, eb_L_history_list, eb_step_size_history_list = [], [], []
    tau_init_used_list = []
    ebblock_tau_hist_list, ebblock_gamma_hist_list, ebblock_target_hist_list = [], [], []
    ebblock_n_gamma_clamped_list, ebblock_n_step_capped_list = [], []
    ebblock_group_sizes = None
    seeds_used = []

    if eb_block not in (None, "iso", "layer"):
        raise ValueError(f"fit_rasmussen_gibbs_hierarchical: eb_block must be None, 'iso' or 'layer', got {eb_block!r}")
    if eb_block is not None and (resample_tau or eb_per_step or eb_global_scale or retune_every_samples is None):
        raise ValueError(
            "fit_rasmussen_gibbs_hierarchical: eb_block requires resample_tau=False, eb_per_step=False, "
            "eb_global_scale=False and retune_every_samples set (the block is one retune segment).")

    if eb_global_scale and (resample_tau or retune_every_samples is None):
        raise ValueError(
            "fit_rasmussen_gibbs_hierarchical: eb_global_scale=True requires resample_tau=False "
            "and retune_every_samples set (the EM update replaces the per-weight Gibbs draw, and "
            "(L, step_size) must be retuned as the prior scale moves).")
    if eb_per_step and (resample_tau or eb_global_scale):
        raise ValueError(
            "fit_rasmussen_gibbs_hierarchical: eb_per_step=True requires resample_tau=False and "
            "eb_global_scale=False (three mutually exclusive ways of setting tau).")
    if retune_every_samples is not None and n_samples % retune_every_samples != 0:
        raise ValueError(
            f"fit_rasmussen_gibbs_hierarchical: n_samples ({n_samples}) must be divisible by "
            f"retune_every_samples ({retune_every_samples})."
        )

    for m in range(n_members):
        seed_m = seed + m

        reg = CorrelatedBdeRegressor(
            hidden_layers=hidden_layers, n_members=1, loss=BivariateGaussianNLL(),
            epochs=epochs, validation_split=validation_split, lr=lr, weight_decay=weight_decay,
            warmup_steps=1, n_samples=1, n_thinning=1, patience=patience,
            desired_energy_var_start=desired_energy_var_start, desired_energy_var_end=desired_energy_var_end,
            activation="soft_sign", prior_family="standardnormal", prior_kwargs=None, seed=seed_m,
        )
        reg.pretrain_eval_every = pretrain_eval_every
        reg.pretrain_min_steps = pretrain_min_steps
        reg._resolved_step_size_init = reg.step_size_init if reg.step_size_init is not None else reg.lr
        reg._build_bde()
        reg._bde.eval_every = reg.pretrain_eval_every
        reg._bde.min_delta = getattr(reg, "pretrain_min_delta", 1e-4)
        reg._bde.min_steps = reg.pretrain_min_steps
        reg._bde.fit_members(
            x=Xtr_j, y=ytr_j, optimizer=optax.adamw(reg.lr, reg.weight_decay),
            epochs=reg.epochs, loss=BivariateGaussianNLL(),
        )
        proto = reg._bde.members[0]
        task = reg.task
        n_weights = ravel_pytree(proto.params)[0].shape[0]

        pretrain_train_list.append(np.asarray(proto.forward(proto.params, Xtr_j)))
        pretrain_weights_list.append(np.asarray(ravel_pytree(proto.params)[0]))
        h = reg._bde.history["Model0"]
        pretrain_loss_history.append({
            "epoch": np.asarray(h["epoch"]), "trainloss": np.asarray(h["trainloss"]),
            "valloss": np.asarray(h["valloss"]), "stop_epoch": int(h["stop_epoch"]),
        })

        # tau0: the given constant, or (tau_init_from_pretrained) this member's pretrained-weight RMS.
        tau_init_m = (float(np.sqrt(np.mean(np.square(pretrain_weights_list[-1]))))
                      if tau_init_from_pretrained else float(tau_init))
        tau_init_vec = jnp.full((n_weights,), tau_init_m)
        if eb_block is not None:
            # Groups: all weights ("iso") or one per layer ("layer"). ravel_pytree flattens bde's
            # [(W, b), ...] as W0, b0, W1, b1, ..., so each layer is one contiguous run.
            group_sizes = (np.asarray([int(W.size + b.size) for W, b in proto.params])
                           if eb_block == "layer" else np.asarray([n_weights]))
            group_ids = np.repeat(np.arange(len(group_sizes)), group_sizes)
            assert group_ids.shape[0] == n_weights, "eb_block: layer sizes do not sum to n_weights"
            ebblock_group_sizes = group_sizes
            w_pre64 = np.asarray(pretrain_weights_list[-1], dtype=np.float64)
            tau_groups = np.maximum(
                np.sqrt(np.bincount(group_ids, weights=w_pre64 ** 2) / group_sizes), eb_tau_floor)
            if eb_block_tau0 is not None:
                tau_groups = np.full(len(group_sizes), float(eb_block_tau0))
            tau_init_vec = jnp.asarray(tau_groups[group_ids], dtype=tau_init_vec.dtype)
        prior_obj0 = _array_normal_prior(jnp.zeros(n_weights), tau_init_vec)
        pm0 = _BivariateProbabilisticModel(
            model=proto, params=proto.params, prior=prior_obj0, task=task, temperature=1.0)
        logpost_at_init = partial(pm0.log_unnormalized_posterior, x=Xtr_j, y=ytr_j)

        ar = _bde_warmup.warmup_bde(
            bde=reg._bde, logpost_one=logpost_at_init,
            step_size_init=reg._resolved_step_size_init,
            desired_energy_var_start=desired_energy_var_start,
            desired_energy_var_end=desired_energy_var_end,
            warmup_steps=warmup_steps,
        )
        state0 = tree_map(lambda a: a[0], ar.state)
        L0 = ar.parameters.L[0]
        step0 = ar.parameters.step_size[0]
        sqrt_diag0 = ar.parameters.sqrt_diag_cov[0]

        def _one_raw_step(carry, key, proto=proto, task=task, n_weights=n_weights):
            state, tau, L_cur, step_cur, sqrt_diag_cur = carry
            key_tau, key_step = jax.random.split(key)
            if resample_tau:
                w_flat, _ = ravel_pytree(state.position)
                a_post = ig_a + 0.5
                b_post = ig_b + 0.5 * jnp.square(w_flat)
                g = jax.random.gamma(key_tau, a_post, shape=w_flat.shape) / b_post
                tau_new = jnp.sqrt(jnp.maximum(1.0 / g, 1e-12))
            elif eb_per_step:
                # One global scale, reset to the current weights' RMS -- see docstring.
                w_flat, _ = ravel_pytree(state.position)
                tau_new = jnp.full_like(tau, jnp.maximum(jnp.sqrt(jnp.mean(jnp.square(w_flat))), eb_tau_floor))
            else:
                tau_new = tau  # fixed at its initial value for the whole run -- see docstring

            prior_obj = _array_normal_prior(jnp.zeros(n_weights), tau_new)
            pm = _BivariateProbabilisticModel(
                model=proto, params=proto.params, prior=prior_obj, task=task, temperature=1.0)
            logdensity_fn = partial(pm.log_unnormalized_posterior, x=Xtr_j, y=ytr_j)

            new_logdensity, new_grad = jax.value_and_grad(logdensity_fn)(state.position)
            state = state._replace(logdensity=new_logdensity, logdensity_grad=new_grad)

            kernel = blackjax.mcmc.mclmc.build_kernel(
                logdensity_fn=logdensity_fn, sqrt_diag_cov=sqrt_diag_cur,
                integrator=blackjax.mcmc.integrators.isokinetic_mclachlan)
            new_state, _info = kernel(rng_key=key_step, state=state, L=L_cur, step_size=step_cur)
            return (new_state, tau_new, L_cur, step_cur, sqrt_diag_cur), None

        def _one_kept_sample(carry, keys_block):
            carry, _ = jax.lax.scan(_one_raw_step, carry, keys_block)
            state, tau, L_cur, step_cur, sqrt_diag_cur = carry
            return carry, (state.position, tau)

        # eb_block: same step, plus running sums of (w - w_ref) and its square over every raw step
        # of the block (w_ref = the block's starting weights, to keep float32 sums well-conditioned).
        def _one_raw_step_stats(c, key):
            carry, w_ref, s1, s2 = c
            carry, _ = _one_raw_step(carry, key)
            dw = ravel_pytree(carry[0].position)[0] - w_ref
            return (carry, w_ref, s1 + dw, s2 + dw * dw), None

        def _one_kept_sample_stats(c, keys_block):
            c, _ = jax.lax.scan(_one_raw_step_stats, c, keys_block)
            return c, (c[0][0].position, c[0][1])

        # See docstring point 5: warm-started periodic re-tune of (L, step_size), bypassing
        # bde.sampler.warmup.mclmc_find_L_and_step_size's own cold-reset-of-params behavior by
        # calling the lower-level make_L_step_size_adaptation/make_adaptation_L closures directly.
        def _warm_retune(state, tau, L_prev, step_prev, sqrt_diag_prev, adaptive_state_prev,
                          streaming_avg_prev, key, proto=proto, task=task, n_weights=n_weights):
            prior_obj = _array_normal_prior(jnp.zeros(n_weights), tau)
            pm = _BivariateProbabilisticModel(
                model=proto, params=proto.params, prior=prior_obj, task=task, temperature=1.0)
            logdensity_fn = partial(pm.log_unnormalized_posterior, x=Xtr_j, y=ytr_j)

            new_logdensity, new_grad = jax.value_and_grad(logdensity_fn)(state.position)
            state = state._replace(logdensity=new_logdensity, logdensity_grad=new_grad)

            def kernel(sqrt_diag_cov):
                return blackjax.mcmc.mclmc.build_kernel(
                    logdensity_fn=logdensity_fn, sqrt_diag_cov=sqrt_diag_cov,
                    integrator=blackjax.mcmc.integrators.isokinetic_mclachlan)

            params_prev = _bde_warmup.MCLMCAdaptationState(
                L=L_prev, step_size=step_prev, sqrt_diag_cov=sqrt_diag_prev)
            key1, key2 = jax.random.split(key)
            # 2026-09-17: uses the EMA-accumulator-carrying reimplementation (_warm_step_size_L_adapt)
            # instead of bde.sampler.warmup.make_L_step_size_adaptation directly -- see that
            # function's own docstring for why (bde's version resets its internal accumulator to a
            # blank slate every call, which caused real non-finite failures at n=100).
            state, params, adaptive_state_new, streaming_avg_new = _warm_step_size_L_adapt(
                kernel=kernel, dim=n_weights, state=state, params=params_prev,
                adaptive_state=adaptive_state_prev, streaming_avg=streaming_avg_prev,
                tune1_steps=retune_tune1_steps, tune2_steps=retune_tune2_steps,
                desired_energy_var=desired_energy_var_end,
                trust_in_estimate=1.5, num_effective_samples=100, rng_key=key1,
            )
            if retune_tune3_steps > 0:
                state, params = _bde_warmup.make_adaptation_L(
                    kernel(params.sqrt_diag_cov), Lfactor=0.4,
                )(state, params, retune_tune3_steps, key2)
            return (state, params.L, params.step_size, params.sqrt_diag_cov,
                    adaptive_state_new, streaming_avg_new)

        all_keys = jax.random.split(
            jax.random.PRNGKey(seed_m + 777), n_samples * n_thinning
        ).reshape(n_samples, n_thinning, 2)

        L_history_m, step_history_m = [float(L0)], [float(step0)]
        if retune_every_samples is None:
            _, (positions_kept, taus_kept) = jax.lax.scan(
                _one_kept_sample, (state0, tau_init_vec, L0, step0, sqrt_diag0), all_keys)
        else:
            n_segments = n_samples // retune_every_samples
            keys_by_segment = all_keys.reshape(n_segments, retune_every_samples, n_thinning, 2)
            _run_segment_jit = jax.jit(
                lambda carry, keys_seg: jax.lax.scan(_one_kept_sample, carry, keys_seg))
            _warm_retune_jit = jax.jit(_warm_retune)

            carry = (state0, tau_init_vec, L0, step0, sqrt_diag0)
            # EMA accumulator AND position-variance streaming average both carried ACROSS retunes
            # (2026-09-17 fix, extended same day after a second real-compute failure -- see
            # _warm_step_size_L_adapt's docstring for the full history). Kept as concrete jnp
            # arrays (not bare Python floats) so every call to _warm_retune_jit sees the same
            # dtype/shape and JAX reuses the compiled executable rather than retracing on a
            # weak-type mismatch.
            retune_adaptive_state = (jnp.zeros(()), jnp.zeros(()), jnp.array(jnp.inf))
            retune_streaming_avg = (jnp.zeros(()), jnp.array([jnp.zeros(n_weights), jnp.zeros(n_weights)]))
            retune_key = jax.random.PRNGKey(seed_m + 5555)

            # Empirical-Bayes adaptation of one global prior scale (see docstring's
            # eb_global_scale section) -- unsaved segments, each followed by the closed-form EM
            # update of tau and a warm retune at the new tau. Reuses _run_segment_jit with the
            # production segment shape, so nothing extra is compiled.
            eb_tau_hist_m, eb_L_hist_m, eb_step_hist_m = [tau_init_m], [float(L0)], [float(step0)]
            if eb_global_scale:
                adapt_keys = jax.random.split(
                    jax.random.PRNGKey(seed_m + 31337),
                    eb_adapt_segments * retune_every_samples * n_thinning,
                ).reshape(eb_adapt_segments, retune_every_samples, n_thinning, 2)
                _mean_sq_weight = jax.jit(lambda pos: jnp.mean(jnp.square(
                    jax.vmap(lambda p: ravel_pytree(p)[0])(pos)[retune_every_samples // 2:])))
                for seg in range(eb_adapt_segments):
                    carry, (pos_seg, _) = _run_segment_jit(carry, adapt_keys[seg])
                    state_cur, tau_cur, L_cur, step_cur, sqrt_diag_cur = carry
                    tau_new = jnp.maximum(jnp.sqrt(_mean_sq_weight(pos_seg)), eb_tau_floor)
                    tau_vec = jnp.full((n_weights,), tau_new, dtype=tau_cur.dtype)
                    retune_key, sub_key = jax.random.split(retune_key)
                    (state_new, L_new, step_new, sqrt_diag_new, retune_adaptive_state,
                     retune_streaming_avg) = _warm_retune_jit(
                        state_cur, tau_vec, L_cur, step_cur, sqrt_diag_cur,
                        retune_adaptive_state, retune_streaming_avg, sub_key)
                    carry = (state_new, tau_vec, L_new, step_new, sqrt_diag_new)
                    eb_tau_hist_m.append(float(tau_new))
                    eb_L_hist_m.append(float(L_new))
                    eb_step_hist_m.append(float(step_new))
                L_history_m, step_history_m = [float(carry[2])], [float(carry[3])]

            if eb_block is not None:
                _run_segment_stats_jit = jax.jit(
                    lambda c, keys_seg: jax.lax.scan(_one_kept_sample_stats, c, keys_seg))
                ebblock_tau_hist_m, ebblock_gamma_hist_m, ebblock_target_hist_m = [tau_groups.copy()], [], []
                ebblock_n_gamma_clamped_m, ebblock_n_step_capped_m = 0, 0
                n_raw_block = retune_every_samples * n_thinning

            positions_segs, taus_segs = [], []
            for seg in range(n_segments):
                if eb_block is not None:
                    w_ref = ravel_pytree(carry[0].position)[0]
                    (carry, _, s1, s2), (pos_seg, tau_seg) = _run_segment_stats_jit(
                        (carry, w_ref, jnp.zeros_like(w_ref), jnp.zeros_like(w_ref)), keys_by_segment[seg])
                else:
                    carry, (pos_seg, tau_seg) = _run_segment_jit(carry, keys_by_segment[seg])
                positions_segs.append(pos_seg)
                taus_segs.append(tau_seg)
                state_cur, tau_cur, L_cur, step_cur, sqrt_diag_cur = carry
                if eb_block is not None:
                    # Block update of the prior scale(s) -- see docstring's eb_block section.
                    d1 = np.asarray(s1, dtype=np.float64) / n_raw_block
                    m_w = np.asarray(w_ref, dtype=np.float64) + d1
                    v_w = np.maximum(np.asarray(s2, dtype=np.float64) / n_raw_block - d1 ** 2, 0.0)
                    gamma_raw = group_sizes - np.bincount(group_ids, weights=v_w) / tau_groups ** 2
                    gamma = np.clip(gamma_raw, 1.0, group_sizes)
                    tau_target = np.maximum(
                        np.sqrt(np.bincount(group_ids, weights=m_w ** 2) / gamma), eb_tau_floor)
                    step_raw = eb_block_damping * (np.log(tau_target) - np.log(tau_groups))
                    step_log = np.clip(step_raw, -np.log(eb_block_max_factor), np.log(eb_block_max_factor))
                    tau_groups = np.maximum(tau_groups * np.exp(step_log), eb_tau_floor)
                    ebblock_n_gamma_clamped_m += int(np.sum(gamma != gamma_raw))
                    ebblock_n_step_capped_m += int(np.sum(step_log != step_raw))
                    ebblock_tau_hist_m.append(tau_groups.copy())
                    ebblock_gamma_hist_m.append(gamma_raw)
                    ebblock_target_hist_m.append(tau_target)
                    tau_cur = jnp.asarray(tau_groups[group_ids], dtype=tau_cur.dtype)
                retune_key, sub_key = jax.random.split(retune_key)
                (state_new, L_new, step_new, sqrt_diag_new, retune_adaptive_state,
                 retune_streaming_avg) = _warm_retune_jit(
                    state_cur, tau_cur, L_cur, step_cur, sqrt_diag_cur,
                    retune_adaptive_state, retune_streaming_avg, sub_key)
                carry = (state_new, tau_cur, L_new, step_new, sqrt_diag_new)
                L_history_m.append(float(L_new))
                step_history_m.append(float(step_new))
            positions_kept = tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *positions_segs)
            taus_kept = jnp.concatenate(taus_segs, axis=0)
        taus_kept = np.asarray(taus_kept)  # (n_samples, n_weights)

        raw_test_m = np.asarray(jax.vmap(lambda pos: proto.forward(pos, Xte_j))(positions_kept))
        raw_train_m = np.asarray(jax.vmap(lambda pos: proto.forward(pos, Xtr_j))(positions_kept))

        if not (np.all(np.isfinite(raw_test_m)) and np.all(np.isfinite(raw_train_m))):
            continue  # dropped -- see docstring's SIMPLIFICATION note (no retry-on-new-seed here)

        raw_list.append(raw_test_m)
        raw_train_list.append(raw_train_m)
        tuned_L_list.append(float(L0))
        tuned_step_size_list.append(float(step0))
        tuned_sqrt_diag_cov_list.append(np.asarray(sqrt_diag0))
        tau_mean_list.append(taus_kept.mean(axis=0))
        tau_std_list.append(taus_kept.std(axis=0))
        tau_trace_list.append(taus_kept[::tau_save_stride])
        retune_L_history_list.append(np.asarray(L_history_m))
        retune_step_size_history_list.append(np.asarray(step_history_m))
        if eb_per_step:
            eb_tau_history_list.append(np.concatenate([[tau_init_m], taus_kept[:, 0]]))
        tau_init_used_list.append(tau_init_m)
        if eb_block is not None:
            ebblock_tau_hist_list.append(np.stack(ebblock_tau_hist_m, axis=0))
            ebblock_gamma_hist_list.append(np.stack(ebblock_gamma_hist_m, axis=0))
            ebblock_target_hist_list.append(np.stack(ebblock_target_hist_m, axis=0))
            ebblock_n_gamma_clamped_list.append(ebblock_n_gamma_clamped_m)
            ebblock_n_step_capped_list.append(ebblock_n_step_capped_m)
        if eb_global_scale:
            eb_tau_history_list.append(np.asarray(eb_tau_hist_m))
            eb_L_history_list.append(np.asarray(eb_L_hist_m))
            eb_step_size_history_list.append(np.asarray(eb_step_hist_m))
        seeds_used.append(seed_m)

    n_good = len(raw_list)
    if n_good == 0:
        raise RuntimeError(
            "fit_rasmussen_gibbs_hierarchical: every member produced a non-finite trace -- "
            "no retry-on-new-seed logic here, see the docstring's SIMPLIFICATION note.")

    raw = np.stack(raw_list, axis=0)              # (n_good, n_samples, n_test, 5)
    raw_train = np.stack(raw_train_list, axis=0)  # (n_good, n_samples, n_train, 5)
    pretrain_train = np.stack(pretrain_train_list, axis=0)[:n_good]
    pretrain_weights = np.stack(pretrain_weights_list, axis=0)[:n_good]
    tuned_L = np.asarray(tuned_L_list)
    tuned_step_size = np.asarray(tuned_step_size_list)
    tuned_sqrt_diag_cov = np.stack(tuned_sqrt_diag_cov_list, axis=0)
    hierarchical_tau_mean = np.stack(tau_mean_list, axis=0)
    hierarchical_tau_std = np.stack(tau_std_list, axis=0)
    rasmussen_tau_trace = np.stack(tau_trace_list, axis=0)  # (n_good, n_samples // stride, n_weights)
    rasmussen_L_history = np.stack(retune_L_history_list, axis=0)
    rasmussen_step_size_history = np.stack(retune_step_size_history_list, axis=0)

    # From here on: identical to fit_correlated's own post-sampling summarization
    # (_summarize_correlated_raw + the rhat/ess/TARP/SBC block), reproduced rather than refactored
    # out of fit_correlated to avoid touching that already-validated production path.
    test_summary = _summarize_correlated_raw(raw, ystd, ymu, theta_true=theta_test_np)
    train_summary = _summarize_correlated_raw(raw_train, ystd, ymu, theta_true=yb_np)

    pretrain_train_mean = np.asarray(pretrain_train[..., 0:2]) * ystd + ymu
    pretrain_train_corr = np.asarray(jnp.tanh(jnp.asarray(pretrain_train[..., 4])))

    mean, aleatoric, epistemic, corr_mean = (
        test_summary["mean"], test_summary["aleatoric"], test_summary["epistemic"], test_summary["corr"])
    chain_mean, chain_aleatoric = test_summary["chain_mean"], test_summary["chain_aleatoric"]
    chain_within_spread, chain_corr = test_summary["chain_within_spread"], test_summary["chain_corr"]
    bnn_log_prob_true = test_summary["log_prob_true"]

    between_chain_std = np.std(chain_mean, axis=0)
    within_chain_std = np.sqrt(np.mean(chain_aleatoric ** 2 + chain_within_spread ** 2, axis=0))

    mu_std = raw[..., 0:2]
    s_std = np.asarray(jax.nn.softplus(jnp.asarray(raw[..., 2:4]))) + 1e-6
    corr_std = np.asarray(jnp.tanh(jnp.asarray(raw[..., 4])))
    mean_all = mu_std * ystd + ymu
    scale_all = s_std * ystd

    n_samples_axis = mean_all.shape[1]
    take_per_chain = max(1, min(n_samples_axis, tarp_n_draws // max(n_good, 1)))
    key_chain = jax.random.PRNGKey(seed + 999)
    chain_sample_idx = jax.random.choice(
        key_chain, n_samples_axis, (take_per_chain,), replace=(n_samples_axis < take_per_chain))
    mean_chain_j = jnp.asarray(mean_all)[:, chain_sample_idx, :, :]
    scale_chain_j = jnp.asarray(scale_all)[:, chain_sample_idx, :, :]
    corr_chain_j = jnp.asarray(corr_std)[:, chain_sample_idx, :]
    _, sk_chain = jax.random.split(key_chain)
    sk_chains = jax.random.split(sk_chain, n_good)
    chain_theta_samples = np.asarray(jax.vmap(metrics.sample_posterior_predictive)(
        sk_chains, mean_chain_j, scale_chain_j, corr_chain_j))

    K_mu = mean_all.reshape(-1, mean_all.shape[-2], mean_all.shape[-1])
    K_s = scale_all.reshape(-1, scale_all.shape[-2], scale_all.shape[-1])
    K_corr = corr_std.reshape(-1, corr_std.shape[-1])
    mu, sd, corr = jnp.asarray(K_mu), jnp.asarray(K_s), jnp.asarray(K_corr)

    key = jax.random.PRNGKey(seed)
    K = mu.shape[0]
    take = min(tarp_n_draws, K)
    idx = jax.random.choice(key, K, (take,), replace=(K < tarp_n_draws))
    key, sk = jax.random.split(key)
    bnn_theta_samples = np.asarray(metrics.sample_posterior_predictive(sk, mu[idx], sd[idx], corr[idx]))
    bnn_f = metrics.compute_tarp_f_stats(bnn_theta_samples, theta_test_np, tarp_lo, tarp_hi, tarp_seeds)
    bnn_sbc_rank = metrics.sbc_marginal_rank(bnn_theta_samples, theta_test_np)

    bnn_raw = dict(
        mean=mean, aleatoric=aleatoric, epistemic=epistemic, corr=corr_mean,
        log_prob_true=bnn_log_prob_true, sbc_rank=bnn_sbc_rank,
        theta_samples=bnn_theta_samples,
        chain_mean=chain_mean, chain_aleatoric=chain_aleatoric,
        chain_within_spread=chain_within_spread, chain_corr=chain_corr,
        between_chain_std=between_chain_std, within_chain_std=within_chain_std,
        chain_theta_samples=chain_theta_samples,
        train_mean=train_summary["mean"], train_corr=train_summary["corr"],
        train_chain_mean=train_summary["chain_mean"], train_chain_corr=train_summary["chain_corr"],
        train_log_prob_true=train_summary["log_prob_true"],
        pretrain_train_mean=pretrain_train_mean, pretrain_train_corr=pretrain_train_corr,
        pretrain_loss_history=pretrain_loss_history,
        pretrain_weights=pretrain_weights,
        tuned_L=tuned_L, tuned_step_size=tuned_step_size, tuned_sqrt_diag_cov=tuned_sqrt_diag_cov,
        hierarchical_tau_mean=hierarchical_tau_mean, hierarchical_tau_std=hierarchical_tau_std,
    )

    rhat, ess, mixing_ok = _rhat_ess_or_nan(mu_std, rhat_threshold)

    diag = dict(
        rhat_mean=float(np.mean(rhat)), rhat_max=float(np.max(rhat)),
        rhat_sigma_max=float(np.max(rhat[..., 0])), rhat_rho_max=float(np.max(rhat[..., 1])),
        ess_mean=float(np.mean(ess)), ess_min=float(np.min(ess)),
        rhat_full=rhat, ess_full=ess,
        n_members_used=n_good, seeds_used=seeds_used, mixing_ok=mixing_ok,
        temperature=1.0,
        # No retry-on-new-seed loop here (see docstring) -- these mirror fit_correlated's own
        # retry-bookkeeping field NAMES so downstream .npz consumers don't need a special case,
        # but every member here comes from exactly one attempt each.
        seed_attempts=1, members_tried_across_attempts=n_members,
        members_failed_across_attempts=n_members - n_good,
        member_fail_rate_across_attempts=(n_members - n_good) / n_members,
        rasmussen_tau_trace=rasmussen_tau_trace,
        rasmussen_tau_save_stride=tau_save_stride,
        rasmussen_ig_a=ig_a, rasmussen_ig_b=ig_b, rasmussen_resample_tau=bool(resample_tau),
        rasmussen_retune_every_samples=(retune_every_samples if retune_every_samples is not None else 0),
        rasmussen_retune_tune1_steps=retune_tune1_steps,
        rasmussen_retune_tune2_steps=retune_tune2_steps,
        rasmussen_retune_tune3_steps=retune_tune3_steps,
        rasmussen_L_history=rasmussen_L_history,
        rasmussen_step_size_history=rasmussen_step_size_history,
    )
    if eb_per_step:
        diag.update(
            rasmussen_eb_tau_history=np.stack(eb_tau_history_list, axis=0),
            rasmussen_eb_tau_init=np.asarray(tau_init_used_list),
        )
    if eb_block is not None:
        diag.update(
            rasmussen_ebblock_mode=eb_block,
            rasmussen_ebblock_tau_history=np.stack(ebblock_tau_hist_list, axis=0),
            rasmussen_ebblock_gamma_history=np.stack(ebblock_gamma_hist_list, axis=0),
            rasmussen_ebblock_tau_target_history=np.stack(ebblock_target_hist_list, axis=0),
            rasmussen_ebblock_n_gamma_clamped=np.asarray(ebblock_n_gamma_clamped_list),
            rasmussen_ebblock_n_step_capped=np.asarray(ebblock_n_step_capped_list),
            rasmussen_ebblock_group_sizes=np.asarray(ebblock_group_sizes),
            rasmussen_ebblock_damping=eb_block_damping,
            rasmussen_ebblock_max_factor=eb_block_max_factor,
            rasmussen_eb_tau_init=np.asarray(tau_init_used_list),
        )
    if eb_global_scale:
        diag.update(
            rasmussen_eb_adapt_segments=eb_adapt_segments,
            rasmussen_eb_tau_history=np.stack(eb_tau_history_list, axis=0),
            rasmussen_eb_L_history=np.stack(eb_L_history_list, axis=0),
            rasmussen_eb_step_size_history=np.stack(eb_step_size_history_list, axis=0),
        )
    # 2026-10-29 (final sweep): the per-draw test-set outputs (standardized; n_good x kept x n_test x 5)
    # and the destandardization constants, exposed as extra keys so a caller can save every draw
    # (chain-wise / width R-hat). Additive: nothing that existed reads these keys.
    bnn_raw["draws_raw_test"], bnn_raw["draws_ymu"], bnn_raw["draws_ystd"] = raw, ymu, ystd
    return bnn_raw, bnn_f, diag


def fit_correlated(Xb, yb, X_test, theta_test, seed,
        n_members, hidden_layers, epochs, patience, validation_split,
        n_samples, warmup_steps, lr, weight_decay, n_thinning,
        desired_energy_var_start, desired_energy_var_end,
        max_seed_tries, rhat_threshold,
        tarp_lo, tarp_hi, tarp_n_draws, tarp_seeds, quiet=True, min_good_members=3,
        pretrain_eval_every=100, pretrain_min_steps=500,
        prior_family="standardnormal", prior_kwargs=None,
        temperature=1.0, lr_decay_multiplier=None, n_kept_samples=None,
        activation="soft_sign", pretrain_kfold=None, score_pretrained=False):
    """Fits a single joint CorrelatedBdeRegressor over (sigma, rho) -- a real
    fitted correlation, not fit()'s corr=0 independence assumption. See the
    module-level override notes above for what makes this possible.

    activation (2026-11-02): hidden-layer activation, any name in bde's registry
    (bde.models.models.Fnn._get_activation, e.g. "soft_sign", "elu", "relu"). Default
    "soft_sign" = the value hard-coded here before this argument existed, so every
    existing call is unchanged. Added for run_experiment.py's "bnn_elu_h8".

    pretrain_kfold (2026-11-02): None (default, unchanged) or an int K -- each member is
    pretrained on its own K-fold split instead of bde's one shared 80/20 split; see
    CorrelatedBdeRegressor._pretrain_kfold. The MCLMC phase uses all training points either way.
    Added for run_experiment.py's "bnn_vanilla_kfold".

    score_pretrained (2026-11-02): False (default, unchanged). When True, the kept members' PRETRAINED networks
    (the weights MCLMC starts from) are also scored on the test set and returned as diag["pretrained"] = dict(
    ens_raw, ens_f, single_raw, single_f, member_finite, n_members_used) -- the equal-weight mixture of the finite
    pretrained members and member 0 alone, via _summarize_pretrained, i.e. the same schema and TARP / SBC keys as
    fit_pretrained_ensemble. "The BNN without sampling" from the same run, at no extra fit. Differences from
    fit_pretrained_ensemble: the members are the KEPT ones (if a member failed in MCLMC and was replaced from a
    re-seeded attempt, its pretrained network is the replacement's); diag["pretrained"] is None if member 0's
    pretrained output is non-finite (the BNN result is still returned).

    STEP-COUNT CONVENTION (2026-10-01, stated here because it differs from
    fit_rasmussen_gibbs_hierarchical's): n_samples is the RAW post-warmup chain step count (bde's
    own convention), and ceil(n_samples / n_thinning) samples are kept per chain -- n_samples=10000,
    n_thinning=10 is 10,000 steps and 1,000 kept. fit_rasmussen_gibbs_hierarchical instead treats
    n_samples as the KEPT count and runs n_samples * n_thinning steps. n_kept_samples (default
    None -- unchanged behavior) keeps exactly that many evenly spaced samples out of n_samples raw
    steps and ignores n_thinning; see CorrelatedBdeRegressor._draw_samples.

    temperature (2026-08-28, default 1.0 -- untempered, exactly bde's stock behavior):
    first exploratory trial for the open aleatoric-inflation investigation.
    Tempers only the MCLMC-phase likelihood term (Delaunoy et
    al. (2024)'s Eq-12-equivalent variant, NOT Wenzel et al. (2020)'s full-posterior
    tempering -- see _BivariateProbabilisticModel's docstring for the exact math and
    why this variant was picked first). Has no effect on pretraining. Forwarded to
    _fit_correlated as a plain per-instance attribute, same pattern as
    pretrain_eval_every/pretrain_min_steps below.

    prior_family/prior_kwargs (2026-08-28, default 'standardnormal'/None -- bde's own untouched
    fixed Normal(0,1) prior over every weight): for an optimized per-weight prior from
    scripts/optimize_prior.py, pass prior_family="normal", prior_kwargs={"loc": ..., "scale": ...}
    with loc/scale as flat arrays matching the network's ravel_pytree order.

    REAL BUG FOUND (2026-08-28, job 57684892) in bde's own prior.py, worked around here rather than
    upstream: bde.sampler.prior.Prior.from_name (what prior_family.get_prior(**prior_kwargs) calls)
    unconditionally builds BOTH log_prior AND f_init from the same prior_kwargs, and
    f_init_normal's `if not loc == 0.0:` guard assumes a scalar loc -- for our array-valued loc (one
    entry per weight) this raises `ValueError: The truth value of an array with more than one
    element is ambiguous` before CorrelatedBdeRegressor.fit() ever runs, even though f_init's
    RESULT is never actually invoked anywhere in this fit() path (confirmed via source read of
    bde.bde.py/mile_wrapper.py -- MCMC always starts from the pretrained weights, never from a
    fresh f_init draw). log_prior_normal itself has no such bug (confirmed via source read: it
    broadcasts array-valued loc/scale over the flattened weight vector via ordinary array
    broadcasting, exactly as intended). Fix: when prior_family=="normal" and prior_kwargs' loc is
    array-valued (size>1), this function builds the bde.sampler.prior.Prior object directly here
    -- log_prior=log_prior_normal(loc=loc, scale=scale) (the real, correct prior term) paired with
    a trivial no-op f_init (never invoked, so its actual behavior is irrelevant) -- bypassing
    Prior.from_name/get_prior entirely, and passes that Prior object as prior_family instead of the
    string (CorrelatedBdeRegressor._build_log_post already has an isinstance(prior_family, Prior)
    branch for exactly this case; prior_kwargs is ignored once prior_family is a Prior instance).
    Scalar-valued prior_kwargs (e.g. a future prior_family="normal", prior_kwargs={"loc":0.0,
    "scale":2.0}) still go through bde's own normal path unchanged -- only array-valued loc hits
    the bug and this workaround.

    pretrain_eval_every/pretrain_min_steps (2026-08-26, defaults match gaussian_predictor.py's
    schedule): the pretraining phase's validation-check cadence and a hard floor on how many
    pretraining steps must run before early stopping is allowed to trigger at all. See
    _MinStepsEarlyStoppingCallback's docstring above for the failure mode this fixes -- bde's own
    default (eval_every=1) meant patience=5 was effectively 5 epochs, which at low n_train could
    (and did) halt pretraining almost immediately.

    min_good_members (default 3, i.e. every member of a 3-member ensemble): _fit_correlated
    accumulates finite members across separately-reseeded attempts (not "reseed and refit the
    whole ensemble until one attempt alone comes back with this many finite") until this many are
    collected, or max_seed_tries attempts have run -- see _fit_correlated's docstring for the
    2026-08-25 change from the original all-in-one-attempt scheme. Mixing (R-hat) plays no role in
    which members are kept or how long the search runs, also per explicit request; rhat_threshold
    below is used only to compute the `mixing_ok` diagnostic on the final pooled chains, reported
    but not acted on. The full retry history (seed_attempts, members_tried/failed_across_attempts)
    is always in the returned diag.

    Unlike fit()'s per-parameter scale readout (np.abs(raw[...,1]), a rough
    stand-in), this uses the actual softplus transform the likelihood/loss
    apply to their scale heads -- more faithful to the fitted model, and
    available here since the loss/likelihood are both defined in this file.

    Returns (bnn_raw, bnn_f, diag). bnn_raw carries several extra keys fit()'s
    output doesn't have: 'corr' (posterior-mean fitted correlation per test
    point -- fit() assumes corr=0 by construction), 'theta_samples' (the
    (take, n_test, 2) joint posterior-predictive draws used for TARP/SBC,
    exposed for downstream use e.g. a corner plot of one test point), a
    per-chain decomposition on the test set -- 'chain_mean'/'chain_aleatoric'/
    'chain_within_spread'/'chain_corr' (each (n_good_members, n_test, ...),
    one row per surviving ensemble member) plus 'between_chain_std'/
    'within_chain_std' ((n_test, 2), pooled across chains) -- for telling
    between-chain disagreement (what bad R-hat measures) apart from each
    chain's own within-chain spread, instead of only seeing them folded
    together into the pooled 'epistemic' term above -- plus
    'chain_theta_samples' ((n_good_members, take_per_chain, n_test, 2)),
    the per-chain analogue of 'theta_samples' with chain identity preserved,
    for e.g. an overlaid per-chain corner plot or per-chain TARP curve --
    and, finally, first-moment-only quantities computed on the *training* set instead of the test
    set -- 'train_mean'/'train_corr'/'train_chain_mean'/'train_chain_corr'/'train_log_prob_true' --
    a diagnostic for whether the model is fitting the data it was actually shown at all (as opposed
    to merely failing to generalize); not part of the original output schema. 2026-08-26: no
    training-set second moment (train_aleatoric/train_epistemic/train_chain_aleatoric/
    train_chain_within_spread) is computed or saved here -- per explicit decision, the training set
    is only ever evaluated on its first moment; the exact-posterior grid computation that would be
    needed as ground truth for a training-set second moment is comparatively expensive at large
    n_train and was never used for anything else.
    """
    if quiet:
        _silence_bde_progress()

    Xb_np = np.asarray(Xb).reshape(len(Xb), -1)
    yb_np = np.asarray(yb)
    Xte_np = np.asarray(X_test).reshape(len(X_test), -1)
    theta_test_np = np.asarray(theta_test)

    Xtr, ytr, Xmu, Xstd, ymu, ystd = _standardize(Xb_np, yb_np)
    Xte = (Xte_np - Xmu) / Xstd

    # Bug workaround (see docstring above): array-valued prior_kwargs['loc'] crashes bde's own
    # Prior.from_name/get_prior via f_init_normal's scalar-only truthiness guard, even though
    # f_init's result is never used in this fit() path. Build the Prior object directly instead.
    if (prior_family == "normal" and prior_kwargs is not None
            and np.ndim(prior_kwargs.get("loc")) >= 1 and np.size(prior_kwargs.get("loc")) > 1):
        _loc = jnp.asarray(prior_kwargs["loc"])
        _scale = jnp.asarray(prior_kwargs["scale"])
        prior_family = Prior(
            name="Normal",
            f_init=lambda key, shape, dtype=jnp.float32: jnp.zeros(shape, dtype=dtype),  # never invoked
            log_prior=log_prior_normal(loc=_loc, scale=_scale),
        )
        prior_kwargs = None
    elif prior_family == "student_t":
        # 2026-09-05: Student-t isn't in bde's PriorDist registry (NORMAL/STANDARDNORMAL/LAPLACE
        # only) -- build the Prior object directly, same pattern as the array-valued-normal
        # workaround above. See log_prior_student_t's docstring for the literature motivation.
        _stk = prior_kwargs or {}
        prior_family = Prior(
            name="StudentT",
            f_init=f_init_student_t(**_stk),
            log_prior=log_prior_student_t(**_stk),
        )
        prior_kwargs = None

    # 2026-09-09: "hierarchical" sentinel -- jointly-sampled weight-prior scale, see
    # _HierarchicalBivariateProbabilisticModel's docstring. Not a bde.sampler.prior.Prior at all
    # (the position pytree itself is augmented, not just the log-density term), so this is handled
    # entirely via _fit_correlated's own hierarchical_prior=True path below rather than the
    # prior_family/prior_kwargs machinery every other branch above uses. prior_kwargs, when given,
    # may override hyper_log_tau_mu/hyper_log_tau_sigma/tau_max/tau_init (defaults:
    # 0.0/1.0/1.0/0.5*tau_max -- see _HierarchicalBivariateProbabilisticModel's docstring's
    # 2026-09-10 "Positivity link" supersession for why tau is now bounded and tau_init's default
    # changed from 1.0).
    # 2026-09-10: "hierarchical_centered" sentinel -- the CENTERED empirical-baseline counterpart
    # (see _CenteredHierarchicalBivariateProbabilisticModel's docstring), same prior_kwargs
    # overrides, mutually exclusive with "hierarchical".
    hierarchical_prior = prior_family == "hierarchical"
    hierarchical_centered = prior_family == "hierarchical_centered"
    if hierarchical_prior or hierarchical_centered:
        _hk = prior_kwargs or {}
        hyper_log_tau_mu = float(_hk.get("hyper_log_tau_mu", 0.0))
        hyper_log_tau_sigma = float(_hk.get("hyper_log_tau_sigma", 1.0))
        tau_max = float(_hk.get("tau_max", 1.0))
        tau_init = float(_hk.get("tau_init", 0.5 * tau_max))
        # 2026-09-11: tau_link override, matching CorrelatedBdeRegressor's own default ("bounded")
        # when not given -- see _HierarchicalBivariateProbabilisticModel's docstring's "Positivity
        # link" section for the two options.
        tau_link = _hk.get("tau_link", "bounded")
        # 2026-09-13: "MAP init" override (see _map_init_hierarchical_tau's docstring) -- default
        # False = exact old fixed-tau_init init scheme, unchanged.
        map_init_tau = bool(_hk.get("map_init_tau", False))
        prior_family, prior_kwargs = "standardnormal", None  # unused in this branch, kept valid
    else:
        hyper_log_tau_mu, hyper_log_tau_sigma, tau_init, tau_max = 0.0, 1.0, 0.5, 1.0
        tau_link = "bounded"
        map_init_tau = False

    bde_kwargs = dict(
        hidden_layers=hidden_layers, n_members=n_members, loss=BivariateGaussianNLL(),
        epochs=epochs, validation_split=validation_split, lr=lr, weight_decay=weight_decay,
        warmup_steps=warmup_steps, n_samples=n_samples, n_thinning=n_thinning, patience=patience,
        desired_energy_var_start=desired_energy_var_start, desired_energy_var_end=desired_energy_var_end,
        # 2026-08-26: move off bde's own default (relu, never previously overridden here) toward
        # tanh, the activation of an earlier standalone baseline network (an architecture comparison
        # flagged this). bde's activation registry (bde.models.models.Fnn._get_activation) has
        # no true tanh, though: "soft_sign" (x/(1+|x|)) is used instead, a smooth S-shaped
        # saturating nonlinearity with tanh-like gradient behavior (just slightly heavier tails) --
        # explicitly chosen over the registry's "hard_tanh" (clip(x,-1,1)), which despite the name
        # is piecewise-linear and has zero gradient once saturated, unlike tanh.
        # 2026-11-02: now the `activation` argument, default "soft_sign" (unchanged behaviour).
        activation=activation,
        prior_family=prior_family, prior_kwargs=prior_kwargs,
    )

    (raw, raw_train, pretrain_train, pretrain_loss_history, pretrain_weights,
     tuned_L, tuned_step_size, tuned_sqrt_diag_cov, seeds_used, n_good, retry_extra,
     hierarchical_tau_mean, hierarchical_tau_std) = _fit_correlated(
        Xtr, ytr, Xte, bde_kwargs, seed, max_seed_tries, min_good_members,
        pretrain_eval_every, pretrain_min_steps, temperature, lr_decay_multiplier,
        hierarchical_prior, hyper_log_tau_mu, hyper_log_tau_sigma, tau_init, tau_max,
        hierarchical_centered, tau_link, map_init_tau, n_kept_samples=n_kept_samples,
        pretrain_kfold=pretrain_kfold, score_pretrained=score_pretrained,
    )
    pretrain_test = retry_extra.pop("pretrain_test_kept", None)   # 2026-11-02, score_pretrained only
    # raw: (n_good, n_samples, n_test, 5); raw_train: (n_good, n_samples, n_train, 5) -- both standardized y-space
    # pretrain_train: (n_good, n_train, 5) -- pretrained (pre-MCLMC) point estimate, no samples axis

    test_summary = _summarize_correlated_raw(raw, ystd, ymu, theta_true=theta_test_np)
    train_summary = _summarize_correlated_raw(raw_train, ystd, ymu, theta_true=yb_np)

    # Diagnostic (2026-08-25): destandardize the pretrained-only point estimate the same way
    # _summarize_correlated_raw does for mu/corr, so a caller can check whether the collapse to a
    # near-constant prediction is already present right after pretraining, before MCLMC/the prior
    # ever touch the weights. No aleatoric/epistemic split here -- one point per member, not a
    # posterior sample.
    pretrain_train_mean = np.asarray(pretrain_train[..., 0:2]) * ystd + ymu       # (n_good, n_train, 2)
    pretrain_train_corr = np.asarray(jnp.tanh(jnp.asarray(pretrain_train[..., 4])))  # (n_good, n_train)

    mean, aleatoric, epistemic, corr_mean = (
        test_summary["mean"], test_summary["aleatoric"], test_summary["epistemic"], test_summary["corr"]
    )
    chain_mean, chain_aleatoric = test_summary["chain_mean"], test_summary["chain_aleatoric"]
    chain_within_spread, chain_corr = test_summary["chain_within_spread"], test_summary["chain_corr"]
    bnn_log_prob_true = test_summary["log_prob_true"]

    # Exact law-of-total-variance split of the pooled epistemic term above: epistemic**2 ==
    # mean_chains(chain_within_spread**2) + between_chain_std**2 (population variance, ddof=0,
    # matching mu.var(0)) -- holds because every chain contributes the same number of samples to
    # the pool.
    between_chain_std = np.std(chain_mean, axis=0)                                          # (n_test, 2)
    within_chain_std = np.sqrt(np.mean(chain_aleatoric ** 2 + chain_within_spread ** 2, axis=0))  # (n_test, 2)

    # Per-chain posterior-predictive draws -- same idea as bnn_theta_samples below (drawn from
    # each test point's bivariate Gaussian mixture component), but sampled separately per chain
    # instead of pooling across chains first, so chain identity is preserved. Needed for any
    # downstream per-chain plot (e.g. an overlaid corner plot showing each chain's own posterior
    # for one test point) that the pooled theta_samples can't provide, since pooling discards
    # which chain each draw came from.
    mu_std = raw[..., 0:2]
    s_std = np.asarray(jax.nn.softplus(jnp.asarray(raw[..., 2:4]))) + 1e-6
    corr_std = np.asarray(jnp.tanh(jnp.asarray(raw[..., 4])))
    mean_all = mu_std * ystd + ymu
    scale_all = s_std * ystd

    n_good_ = mean_all.shape[0]
    take_per_chain = max(1, min(n_samples, tarp_n_draws // max(n_good_, 1)))
    key_chain = jax.random.PRNGKey(seed + 999)
    chain_sample_idx = jax.random.choice(key_chain, n_samples, (take_per_chain,), replace=(n_samples < take_per_chain))
    mean_chain_j = jnp.asarray(mean_all)[:, chain_sample_idx, :, :]    # (n_good, take_per_chain, n_test, 2)
    scale_chain_j = jnp.asarray(scale_all)[:, chain_sample_idx, :, :]
    corr_chain_j = jnp.asarray(corr_std)[:, chain_sample_idx, :]
    _, sk_chain = jax.random.split(key_chain)
    sk_chains = jax.random.split(sk_chain, n_good_)
    chain_theta_samples = np.asarray(jax.vmap(metrics.sample_posterior_predictive)(
        sk_chains, mean_chain_j, scale_chain_j, corr_chain_j
    ))  # (n_good, take_per_chain, n_test, 2)

    K_mu = mean_all.reshape(-1, mean_all.shape[-2], mean_all.shape[-1])      # (K, n_test, 2)
    K_s = scale_all.reshape(-1, scale_all.shape[-2], scale_all.shape[-1])    # (K, n_test, 2)
    K_corr = corr_std.reshape(-1, corr_std.shape[-1])                       # (K, n_test)

    mu = jnp.asarray(K_mu)
    sd = jnp.asarray(K_s)
    corr = jnp.asarray(K_corr)

    key = jax.random.PRNGKey(seed)
    K = mu.shape[0]
    take = min(tarp_n_draws, K)
    idx = jax.random.choice(key, K, (take,), replace=(K < tarp_n_draws))
    key, sk = jax.random.split(key)
    bnn_theta_samples = np.asarray(metrics.sample_posterior_predictive(sk, mu[idx], sd[idx], corr[idx]))
    bnn_f = metrics.compute_tarp_f_stats(bnn_theta_samples, theta_test_np, tarp_lo, tarp_hi, tarp_seeds)
    bnn_sbc_rank = metrics.sbc_marginal_rank(bnn_theta_samples, theta_test_np)

    bnn_raw = dict(
        mean=mean, aleatoric=aleatoric, epistemic=epistemic, corr=corr_mean,
        log_prob_true=bnn_log_prob_true, sbc_rank=bnn_sbc_rank,
        # (take, n_test, 2) joint posterior-predictive draws -- same draws used for TARP/SBC
        # above, exposed here too since they're a genuine per-test-point posterior sample
        # (e.g. for a corner plot of one example test point), not just an internal
        # intermediate.
        theta_samples=bnn_theta_samples,
        # Per-chain decomposition (see comment above where these are computed) -- lets a caller
        # tell between-chain disagreement (bad mixing) apart from each chain's own within-chain
        # spread, instead of only seeing the two folded together into the pooled 'epistemic'
        # term above.
        chain_mean=chain_mean, chain_aleatoric=chain_aleatoric,
        chain_within_spread=chain_within_spread, chain_corr=chain_corr,
        between_chain_std=between_chain_std, within_chain_std=within_chain_std,
        # (n_good_members, take_per_chain, n_test, 2) -- chain_theta_samples[c] is chain c's own
        # posterior-predictive draws, chain identity preserved (unlike theta_samples above, which
        # pools all chains together). For a per-chain overlay plot, e.g. corner.corner on
        # chain_theta_samples[c, :, idx, :] for each c, or a per-chain TARP curve via
        # metrics.compute_tarp_f_stats(chain_theta_samples[c], ...).
        chain_theta_samples=chain_theta_samples,
        # Training-set diagnostic (see _summarize_correlated_raw) -- is the model fitting the data
        # it was actually shown at all, as opposed to merely failing to generalize? train_log_prob_true
        # uses yb_np (the real simulated (sigma, rho) that generated each training example -- known
        # exactly, since this is simulated data), the training-set analogue of log_prob_true above.
        # First moment only (2026-08-26): no train_aleatoric/train_epistemic/
        # train_chain_aleatoric/train_chain_within_spread -- see fit_correlated's docstring above.
        train_mean=train_summary["mean"], train_corr=train_summary["corr"],
        train_chain_mean=train_summary["chain_mean"], train_chain_corr=train_summary["chain_corr"],
        train_log_prob_true=train_summary["log_prob_true"],
        # Pretrained (pre-MCLMC) point estimate on the training set, per kept member -- (n_good,
        # n_train, 2)/(n_good, n_train) -- localizes a "predicts one constant" collapse to
        # pretraining itself vs. the MCLMC/prior phase (2026-08-25 diagnostic, see
        # CorrelatedBdeRegressor.fit's pretrain_train_preds_).
        pretrain_train_mean=pretrain_train_mean, pretrain_train_corr=pretrain_train_corr,
        # 2026-09-04: per-kept-member pretraining loss curve (see _fit_correlated's docstring) --
        # a list of n_good {epoch, trainloss, valloss, stop_epoch} dicts, NOT stacked into an array
        # (members can stop at different epochs). Not written to any existing script's .npz output
        # unless a caller explicitly reads this key -- purely additive, existing callers that don't
        # reference it are unaffected.
        pretrain_loss_history=pretrain_loss_history,
        # 2026-09-05: raw flattened pretrained weight vector per kept member, (n_good, n_weights) --
        # see CorrelatedBdeRegressor.fit's pretrain_weights_ docstring. Purely additive.
        pretrain_weights=pretrain_weights,
        # 2026-09-08: MCLMC's own tuned per-member sampler hyperparameters -- see
        # CorrelatedBdeRegressor.fit's tuned_L_/tuned_step_size_/tuned_sqrt_diag_cov_ docstring.
        # tuned_L/tuned_step_size are (n_good,); tuned_sqrt_diag_cov is (n_good, n_weights).
        # Purely additive.
        tuned_L=tuned_L, tuned_step_size=tuned_step_size, tuned_sqrt_diag_cov=tuned_sqrt_diag_cov,
        # 2026-09-10: (n_good, n_weights) PER-WEIGHT posterior mean/std of physical
        # tau=exp(log_tau) across MCMC samples, per kept member, only when
        # prior_family=="hierarchical" -- None otherwise. NOT the full per-sample array (storage
        # size decision, see CorrelatedBdeRegressor.fit's docstring). See
        # _HierarchicalBivariateProbabilisticModel's docstring for the per-weight (ARD-style)
        # model this now is.
        hierarchical_tau_mean=hierarchical_tau_mean, hierarchical_tau_std=hierarchical_tau_std,
    )

    # Informational only (2026-08-25) -- no longer used to decide which/how many members
    # _fit_correlated keeps; see its docstring. _rhat_ess_or_nan (2026-09-17) guards the <2-chain
    # case that blackjax.rhat/ess itself can't handle -- see that function's own docstring for the
    # real crash (job 58370119) this fixes.
    rhat, ess, mixing_ok = _rhat_ess_or_nan(mu_std, rhat_threshold)

    diag = dict(
        rhat_mean=float(np.mean(rhat)), rhat_max=float(np.max(rhat)),
        rhat_sigma_max=float(np.max(rhat[..., 0])), rhat_rho_max=float(np.max(rhat[..., 1])),
        ess_mean=float(np.mean(ess)), ess_min=float(np.min(ess)),
        # 2026-09-08: full per-test-point/per-parameter (n_test, 2) rhat/ess arrays -- computed just
        # above unconditionally already, only their scalar mean/max/min summaries were ever kept.
        # Added to localize exactly which test point (and sigma vs. rho) is driving the worst-case
        # numbers, e.g. an ess_min that doesn't move even when average mixing improves substantially
        # (n_train=10,000 preconditioning finding) -- purely additive.
        rhat_full=rhat, ess_full=ess,
        n_members_used=n_good, seeds_used=seeds_used, mixing_ok=mixing_ok,
        temperature=float(temperature),
        **retry_extra,
        member_fail_rate_across_attempts=(
            retry_extra["members_failed_across_attempts"] / retry_extra["members_tried_across_attempts"]
        ),
    )

    # 2026-10-29 (final sweep): the per-draw test-set outputs (standardized; n_good x kept x n_test x 5)
    # and the destandardization constants, exposed as extra keys so a caller can save every draw
    # (chain-wise / width R-hat). Additive: nothing that existed reads these keys.
    bnn_raw["draws_raw_test"], bnn_raw["draws_ymu"], bnn_raw["draws_ystd"] = raw, ymu, ystd

    if score_pretrained:   # 2026-11-02 -- see the docstring
        pt, ptr = np.asarray(pretrain_test), np.asarray(pretrain_train)
        finite_pt = np.isfinite(pt).all(axis=(1, 2)) & np.isfinite(ptr).all(axis=(1, 2))
        if finite_pt[0]:
            g = np.flatnonzero(finite_pt)
            ens_raw, ens_f = _summarize_pretrained(pt[g][:, None], ptr[g][:, None], ptr[g], ystd, ymu, theta_test_np,
                                                   yb_np, seed, tarp_lo, tarp_hi, tarp_n_draws, tarp_seeds)
            single_raw, single_f = _summarize_pretrained(pt[:1][:, None], ptr[:1][:, None], ptr[:1], ystd, ymu,
                                                         theta_test_np, yb_np, seed, tarp_lo, tarp_hi, tarp_n_draws,
                                                         tarp_seeds)
            diag["pretrained"] = dict(ens_raw=ens_raw, ens_f=ens_f, single_raw=single_raw, single_f=single_f,
                                      member_finite=finite_pt, n_members_used=int(len(g)))
        else:
            diag["pretrained"] = None
    return bnn_raw, bnn_f, diag


def fit(Xb, yb, X_test, theta_test, seed,
        n_members, hidden_layers, epochs, patience, validation_split,
        n_samples, warmup_steps, lr, weight_decay, n_thinning,
        desired_energy_var_start, desired_energy_var_end,
        max_seed_tries, rhat_threshold,
        tarp_lo, tarp_hi, tarp_n_draws, tarp_seeds, quiet=True):
    """Fits independent BdeRegressors for sigma and rho, pools ensemble members + MCMC draws into a
    joint (independence-assumption) BNN posterior, and evaluates it. Returns
    (bnn_raw, bnn_f, tau_mean, tau_std, diag) matching the experiment's .npz schema.

    quiet=True (default) silences bde's internal progress bars — appropriate for batch/slurm runs;
    pass quiet=False for an interactive run where you want to see them (e.g. bde_test.py)."""
    if quiet:
        _silence_bde_progress()

    Xb_np = np.asarray(Xb).reshape(len(Xb), -1)
    yb_np = np.asarray(yb)
    Xte_np = np.asarray(X_test).reshape(len(X_test), -1)
    theta_test_np = np.asarray(theta_test)

    Xtr, ytr, Xmu, Xstd, ymu, ystd = _standardize(Xb_np, yb_np)
    Xte = (Xte_np - Xmu) / Xstd

    bde_kwargs = dict(
        hidden_layers=hidden_layers, n_members=n_members, loss=GaussianNLL(),
        epochs=epochs, validation_split=validation_split, lr=lr, weight_decay=weight_decay,
        warmup_steps=warmup_steps, n_samples=n_samples, n_thinning=n_thinning, patience=patience,
        desired_energy_var_start=desired_energy_var_start, desired_energy_var_end=desired_energy_var_end,
    )

    means, scales, diags = {}, {}, {}
    for j, name in enumerate(PARAM_NAMES):
        raw, used_seed, n_total, n_good, mixing_ok = _fit_one_param(
            Xtr, ytr[:, j], Xte, bde_kwargs, seed + j * 1000, max_seed_tries, rhat_threshold
        )
        mu_std, s_std = raw[..., 0], np.abs(raw[..., 1])

        means[name] = mu_std * ystd[j] + ymu[j]
        scales[name] = s_std * ystd[j]

        rhat = np.asarray(blackjax.rhat(jnp.asarray(mu_std), chain_axis=0, sample_axis=1))
        ess = np.asarray(blackjax.ess(jnp.asarray(mu_std), chain_axis=0, sample_axis=1))
        diags[name] = dict(
            rhat_mean=float(np.mean(rhat)), rhat_max=float(np.max(rhat)),
            ess_mean=float(np.mean(ess)), ess_min=float(np.min(ess)),
            n_members_used=n_good, n_members_total=n_total, used_seed=used_seed,
            failure_rate=1.0 - n_good / n_total, mixing_ok=mixing_ok,
        )

    # Pool into a joint (K, n_test, 2) posterior. Sigma and rho draws are paired by flat index
    # (member/sample slot i of the sigma regressor with slot i of the rho regressor) — an explicit
    # independence assumption (corr=0), not a fitted correlation; see module docstring.
    sigma_flat_mu = means["sigma"].reshape(-1, means["sigma"].shape[-1])
    rho_flat_mu = means["rho"].reshape(-1, means["rho"].shape[-1])
    sigma_flat_s = scales["sigma"].reshape(-1, scales["sigma"].shape[-1])
    rho_flat_s = scales["rho"].reshape(-1, scales["rho"].shape[-1])

    K = min(sigma_flat_mu.shape[0], rho_flat_mu.shape[0])
    mu = jnp.stack([jnp.asarray(sigma_flat_mu[:K]), jnp.asarray(rho_flat_mu[:K])], axis=-1)  # (K, n_test, 2)
    sd = jnp.stack([jnp.asarray(sigma_flat_s[:K]), jnp.asarray(rho_flat_s[:K])], axis=-1)    # (K, n_test, 2)
    corr = jnp.zeros(mu.shape[:2])                                                            # (K, n_test)

    theta_te = jnp.asarray(theta_test_np)
    log_comp = metrics.bivariate_normal_logpdf(theta_te[None], mu, sd, corr)  # (K, n_test)
    bnn_log_prob_true = np.asarray(jax.scipy.special.logsumexp(log_comp, axis=0) - jnp.log(mu.shape[0]))

    mean = np.asarray(mu.mean(0))
    aleatoric = np.sqrt(np.asarray((sd**2).mean(0)))
    epistemic = np.sqrt(np.asarray(mu.var(0)))

    key = jax.random.PRNGKey(seed)
    take = min(tarp_n_draws, K)
    idx = jax.random.choice(key, K, (take,), replace=(K < tarp_n_draws))
    key, sk = jax.random.split(key)
    bnn_theta_samples = np.asarray(metrics.sample_posterior_predictive(sk, mu[idx], sd[idx], corr[idx]))
    bnn_f = metrics.compute_tarp_f_stats(bnn_theta_samples, theta_test_np, tarp_lo, tarp_hi, tarp_seeds)
    bnn_sbc_rank = metrics.sbc_marginal_rank(bnn_theta_samples, theta_test_np)

    bnn_raw = dict(
        mean=mean, aleatoric=aleatoric, epistemic=epistemic,
        log_prob_true=bnn_log_prob_true, sbc_rank=bnn_sbc_rank,
    )

    # bde has no per-layer hierarchical tau prior — NaN placeholders, kept only for .npz schema
    # compatibility with the analysis notebook's existing tau plots.
    tau_mean = np.full(5, np.nan)
    tau_std = np.full(5, np.nan)

    diag = dict(
        rhat_mu_mean=float(np.mean([diags[n]["rhat_mean"] for n in PARAM_NAMES])),
        rhat_mu_max=float(np.max([diags[n]["rhat_max"] for n in PARAM_NAMES])),
        ess_mu_mean=float(np.mean([diags[n]["ess_mean"] for n in PARAM_NAMES])),
        ess_mu_min=float(np.min([diags[n]["ess_min"] for n in PARAM_NAMES])),
        rhat_tau_mean=np.nan, rhat_tau_max=np.nan, ess_tau_mean=np.nan, ess_tau_min=np.nan,
        n_chains_used=int(np.min([diags[n]["n_members_used"] for n in PARAM_NAMES])),
        sigma_failure_rate=diags["sigma"]["failure_rate"], rho_failure_rate=diags["rho"]["failure_rate"],
        sigma_mixing_ok=diags["sigma"]["mixing_ok"], rho_mixing_ok=diags["rho"]["mixing_ok"],
        sigma_seed_used=diags["sigma"]["used_seed"], rho_seed_used=diags["rho"]["used_seed"],
    )

    return bnn_raw, bnn_f, tau_mean, tau_std, diag
