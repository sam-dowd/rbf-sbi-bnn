"""Conditional MAF normalizing-flow neural posterior estimator (NPE), via nflows."""
import copy

import numpy as np
import torch
import torch.nn as torch_nn
from tqdm.auto import tqdm

from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform

from rbf_sbi.metrics import compute_tarp_f_stats, sbc_marginal_rank

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContextEncoder(torch_nn.Module):
    def __init__(self, input_dim=100, context_dim=32, hidden=128):
        super().__init__()

        self.net = torch_nn.Sequential(
            torch_nn.Linear(input_dim, hidden),
            torch_nn.ReLU(),
            torch_nn.Linear(hidden, hidden),
            torch_nn.ReLU(),
            torch_nn.Linear(hidden, context_dim),
        )

    def forward(self, x):
        return self.net(x)


class ConditionalBivariateGaussian(torch_nn.Module):
    """(2026-11-02) Drop-in replacement for the conditional flow in create_npe_flow(head="gaussian"): a full
    bivariate Gaussian over theta given the encoder's context, N(mu(c), diag(s) R(r) diag(s)), with
    context -> hidden (ReLU) -> 5 outputs (mu_1, mu_2, softplus scales + 1e-6, correlation 0.99*tanh -- the
    gaussian_predictor.py convention). Implements the two nflows Flow methods fit() calls, with the same
    shapes: log_prob(y, context) -> (n,), sample(num_samples, context) -> (n_context, num_samples, 2).
    Purpose: NPE with its encoder, optimiser, split and stopping rule unchanged but the flow head replaced by
    the BNN's head family, to test whether NPE's better sigma mean at n=50 comes from the flow head.
    ASSUMPTIONS: one hidden layer of hidden_features units (one MADE block of the flow has the same width);
    0.99 cap on |corr| for numerical safety, as in the Gaussian predictor."""

    def __init__(self, context_features=8, hidden_features=16):
        super().__init__()
        self.net = torch_nn.Sequential(
            torch_nn.Linear(context_features, hidden_features),
            torch_nn.ReLU(),
            torch_nn.Linear(hidden_features, 5),
        )

    def _params(self, context):
        o = self.net(context)
        return o[..., :2], torch_nn.functional.softplus(o[..., 2:4]) + 1e-6, 0.99 * torch.tanh(o[..., 4])

    def log_prob(self, inputs, context):
        mu, s, r = self._params(context)
        z = (inputs - mu) / s
        one_m = 1.0 - r ** 2
        return (-np.log(2 * np.pi) - torch.log(s).sum(-1) - 0.5 * torch.log(one_m)
                - (z[..., 0] ** 2 - 2 * r * z[..., 0] * z[..., 1] + z[..., 1] ** 2) / (2 * one_m))

    def sample(self, num_samples, context):
        mu, s, r = self._params(context)                                  # (n, 2), (n, 2), (n,)
        e = torch.randn(context.shape[0], num_samples, 2, device=context.device, dtype=mu.dtype)
        x1 = mu[:, None, 0] + s[:, None, 0] * e[..., 0]
        x2 = mu[:, None, 1] + s[:, None, 1] * (r[:, None] * e[..., 0] + torch.sqrt(1 - r[:, None] ** 2) * e[..., 1])
        return torch.stack([x1, x2], -1)


def create_npe_flow(
    theta_dim=2,
    input_context_dim=100,
    encoded_context_dim=8,
    hidden_features=16,
    num_layers=2,
    num_blocks=1,
    head="flow",
):
    """head (2026-11-02): "flow" (default, unchanged) or "gaussian" -- ConditionalBivariateGaussian in place of
    the MAF flow; the encoder is built first either way, so it gets the same initial weights."""
    encoder = ContextEncoder(
        input_dim=input_context_dim,
        context_dim=encoded_context_dim,
        hidden=16,
    )
    if head == "gaussian":
        if theta_dim != 2:
            raise ValueError("head='gaussian' is the two-target (sigma, rho) head")
        return torch_nn.ModuleDict({
            "encoder": encoder,
            "flow": ConditionalBivariateGaussian(encoded_context_dim, hidden_features),
        })
    if head != "flow":
        raise ValueError(f"head must be 'flow' or 'gaussian', got {head!r}")

    transforms = []

    for i in range(num_layers):
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=theta_dim,
                hidden_features=hidden_features,
                context_features=encoded_context_dim,
                num_blocks=num_blocks,
                activation=torch_nn.ReLU(),
                dropout_probability=0.0,
                use_batch_norm=False,
            )
        )

        transforms.append(ReversePermutation(features=theta_dim))

    transform = CompositeTransform(transforms)
    base_dist = StandardNormal(shape=[theta_dim])
    flow = Flow(transform=transform, distribution=base_dist)

    return torch_nn.ModuleDict({
        "encoder": encoder,
        "flow": flow,
    })


def fit(Xt_np, yt_np, Xv_np, yv_np, X_test_np, theta_test_np, seed, lr, wd,
        steps, eval_every, patience,
        tarp_n_draws, n_posterior_samples, tarp_seeds, tarp_lo, tarp_hi, head="flow", gaussian_hidden=16,
        info_out=None, context_dim=8):
    """Trains a conditional MAF NPE on (Xt_np, yt_np) with early stopping on (Xv_np, yv_np), then
    evaluates against the shared X_test_np/theta_test_np. Returns (npe_raw, npe_f) matching the
    npe_* keys in the experiment's .npz schema. head (2026-11-02): "flow" (default, unchanged) or
    "gaussian" -- see create_npe_flow / ConditionalBivariateGaussian. gaussian_hidden: hidden width of that
    head (16 = one MADE block's width, 229 parameters; 128 gives 1,797, about the flow's 1,896); unused for "flow".
    info_out (2026-11-04): optional dict; if given, receives steps_run (optimiser steps taken), best_step (step whose
    weights are kept), stopped_early (patience hit before `steps`) and val_history ((step, val loss) per evaluation).
    Recording only -- training is identical with or without it.
    context_dim (2026-11-04): the encoder's output width (encoded_context_dim of create_npe_flow; default 8, unchanged).
    16 gives the encoder 100->16->16->16, the BNN's width (models final_npe_ctx16 / final_npe_gauss_ctx16)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    def T(z):
        return torch.as_tensor(np.asarray(z).reshape(len(z), -1).copy(),
                                dtype=torch.float32, device=DEVICE)

    Xt, Xv = T(Xt_np), T(Xv_np)
    yt = torch.as_tensor(np.asarray(yt_np).copy(), dtype=torch.float32, device=DEVICE)
    yv = torch.as_tensor(np.asarray(yv_np).copy(), dtype=torch.float32, device=DEVICE)

    net = create_npe_flow(
        theta_dim=2, input_context_dim=Xt.shape[1],
        encoded_context_dim=int(context_dim), hidden_features=(int(gaussian_hidden) if head == "gaussian" else 16),
        num_layers=2, num_blocks=1, head=head,
    ).to(DEVICE)

    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    def loss(x, y):
        return -net["flow"].log_prob(y, context=net["encoder"](x)).mean()

    best, best_val, bad = None, np.inf, 0
    best_step, steps_run, stopped_early, val_hist = 0, 0, False, []
    bar = tqdm(range(steps), desc="NPE", leave=False)

    for step in bar:
        net.train(); opt.zero_grad(set_to_none=True)
        L = loss(Xt, yt)
        if not torch.isfinite(L): raise RuntimeError("Nonfinite NPE loss")
        L.backward(); opt.step()
        steps_run = step + 1

        if step == 0 or (step + 1) % eval_every == 0:
            net.eval()
            with torch.no_grad(): v = loss(Xv, yv).item()
            val_hist.append((step + 1, v))

            if v < best_val - 1e-4:
                best, best_val, bad = copy.deepcopy(net.state_dict()), v, 0
                best_step = step + 1
            else:
                bad += 1

            bar.set_postfix(val=f"{v:.3f}", patience=f"{bad}/{patience}")
            if bad >= patience:
                stopped_early = True
                break

    if info_out is not None:
        info_out.update(steps_run=steps_run, best_step=best_step, stopped_early=stopped_early,
                        val_history=np.array(val_hist, dtype=np.float64))

    if best is None: raise RuntimeError("NPE failed")
    net.load_state_dict(best); net.eval()

    n_draw = int(n_posterior_samples)
    Xte_t = T(np.asarray(X_test_np).reshape(len(X_test_np), -1))
    with torch.no_grad():
        c = net["encoder"](Xte_t)
        s = net["flow"].sample(n_draw, context=c).cpu().numpy()  # (n_test, n_draw, 2)

        theta_true_t = torch.as_tensor(np.asarray(theta_test_np), dtype=torch.float32, device=DEVICE)
        npe_log_prob_true = net["flow"].log_prob(theta_true_t, context=c).cpu().numpy()  # (n_test,)

    theta_samples = np.swapaxes(s[:, :tarp_n_draws, :], 0, 1)  # (K, n_test, 2)
    npe_f = compute_tarp_f_stats(theta_samples, np.asarray(theta_test_np), tarp_lo, tarp_hi, tarp_seeds)
    npe_sbc_rank = sbc_marginal_rank(theta_samples, np.asarray(theta_test_np))

    npe_raw = dict(
        mean=s.mean(1), std=s.std(1),
        log_prob_true=npe_log_prob_true, sbc_rank=npe_sbc_rank,
        # 2026-10-29 (final sweep, NPE ensembles): the (tarp_n_draws, n_test, 2) draws used for
        # TARP/SBC above. Additive key; nothing that existed reads it.
        theta_samples=theta_samples,
    )

    return npe_raw, npe_f


# ---------------------------------------------------------------------------------------------
# One-target path (2026-10-04, GP surface dataset). Added
# alongside fit(); nothing above this line was changed, so the two-target RBF-kernel path is
# untouched. NOTE: with theta_dim=1 the affine autoregressive layers compose to a single
# shift-and-scale of the standard-normal base, so this "flow" is exactly a conditional Gaussian
# N(mu(x), sigma(x)^2) -- a heteroskedastic Gaussian with a different network from
# gaussian_predictor.py, not a more flexible density.
# ---------------------------------------------------------------------------------------------
def fit_1d(Xt_np, zt_np, Xv_np, zv_np, seed, lr, wd, steps, eval_every, patience):
    """Trains the conditional flow for ONE target on (Xt_np, zt_np (n,)) with early stopping on
    (Xv_np, zv_np). Returns (net, best_val, info) -- info = dict(steps_run, best_step,
    val_history (n_evals, 2) of (step, validation loss)). Same optimiser/stopping rule as fit()."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    def T(a):
        return torch.as_tensor(np.asarray(a).reshape(len(a), -1).copy(), dtype=torch.float32, device=DEVICE)

    Xt, Xv, yt, yv = T(Xt_np), T(Xv_np), T(zt_np), T(zv_np)
    net = create_npe_flow(theta_dim=1, input_context_dim=Xt.shape[1], encoded_context_dim=8,
                          hidden_features=16, num_layers=2, num_blocks=1).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    def loss(x, y):
        return -net["flow"].log_prob(y, context=net["encoder"](x)).mean()

    best, best_val, bad, best_step, history = None, np.inf, 0, 0, []
    for step in range(steps):
        net.train(); opt.zero_grad(set_to_none=True)
        L = loss(Xt, yt)
        if not torch.isfinite(L): raise RuntimeError("Nonfinite NPE loss")
        L.backward(); opt.step()
        if step == 0 or (step + 1) % eval_every == 0:
            net.eval()
            with torch.no_grad(): v = loss(Xv, yv).item()
            history.append((step + 1, v))
            if v < best_val - 1e-4:
                best, best_val, bad, best_step = copy.deepcopy(net.state_dict()), v, 0, step + 1
            else:
                bad += 1
            if bad >= patience: break

    if best is None: raise RuntimeError("NPE failed")
    net.load_state_dict(best); net.eval()
    return net, best_val, dict(steps_run=step + 1, best_step=best_step, val_history=np.asarray(history))


def predict_1d(net, X_np, n_draws, z_np=None):
    """Predictive mean/std at inputs X_np from n_draws flow samples per input, and (if z_np is
    given) the flow's exact log density at the observed targets. Returns dict(mean, std (n,),
    log_prob_true (n,) or None)."""
    X = torch.as_tensor(np.asarray(X_np).reshape(len(X_np), -1).copy(), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        c = net["encoder"](X)
        s = net["flow"].sample(int(n_draws), context=c).cpu().numpy()[..., 0]   # (n, n_draws)
        lp = None
        if z_np is not None:
            z = torch.as_tensor(np.asarray(z_np).reshape(-1, 1).copy(), dtype=torch.float32, device=DEVICE)
            lp = net["flow"].log_prob(z, context=c).cpu().numpy().astype(np.float64)
    return dict(mean=s.mean(1).astype(np.float64), std=s.std(1).astype(np.float64), log_prob_true=lp)
