"""Chain-wise convergence diagnostics (Sommer et al. 2024, App. B; Vehtari et al. 2021) on saved per-draw outputs.

Merged, with function bodies unchanged, from the research repository's chainwise_diagnostics.py (the statistics) and
chainwise_diagnostics_rbf.py (the float64 bivariate log density). ASSUMPTIONS (not from the paper): Eq. 9's window and
tolerance (100 kept draws, 0.01 nats); bulk rank-normalised R-hat without the folded/tail maximum; kappa = 4.
"""
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

KAPPA, WINDOW, EPS = 4, 100, 0.01


def rank_normal_rhat(chains):
    """chains (K, S, P) -> rank-normalised split-free R-hat over the K given sub-chains, per P."""
    K, S, P = chains.shape
    flat = chains.transpose(2, 0, 1).reshape(P, K * S)
    r = rankdata(flat, axis=1)
    z = ndtri((r - 0.375) / (K * S + 0.25)).reshape(P, K, S).transpose(1, 2, 0)   # (K, S, P)
    means = z.mean(1)
    W = z.var(1, ddof=1).mean(0)
    B = S * means.var(0, ddof=1)
    return np.sqrt(((S - 1) / S * W + B / S) / W)


def split(x, k):
    """x (M, S, P) -> (M*k, S//k, P): each chain cut into k consecutive sub-chains."""
    M, S, P = x.shape
    s = S // k
    return x[:, : s * k].reshape(M, k, s, P).reshape(M * k, s, P)


def lppd_path(mu, s, z):
    """mu, s (S, n) one member; z (n,) -> LPPD_l for l = 1..S (Eq. 8)."""
    lp = -0.5 * np.log(2 * np.pi) - np.log(s) - 0.5 * ((z[None] - mu) / s) ** 2      # (S, n)
    cum = np.logaddexp.accumulate(lp, axis=0) - np.log(np.arange(1, lp.shape[0] + 1))[:, None]
    return cum.mean(1)


def eq9_stop(path):
    for l in range(WINDOW, len(path)):
        if abs(path[l - WINDOW:l].mean() - path[l]) < EPS:
            return l + 1
    return -1


def q(a):
    return dict(med=float(np.median(a)), p95=float(np.percentile(a, 95)),
                f101=float(np.mean(a <= 1.01)), f11=float(np.mean(a <= 1.1)))


def logpdf2(th, mu, s, c):
    """float64 log N2(th | mu, diag(s) R(c) diag(s)); th (n, 2), mu/s (..., n, 2), c (..., n). Same formula and
    1 - c^2 >= 1e-6 clip as metrics.bivariate_normal_logpdf; recomputed test NLL matches the saved files to 2e-7."""
    z = (th - mu) / s
    one_m = np.clip(1.0 - c ** 2, 1e-6, None)   # the clip of metrics.bivariate_normal_logpdf (|corr| can round to 1)
    return (-np.log(2 * np.pi) - np.log(s).sum(-1) - 0.5 * np.log(one_m)
            - (z[..., 0] ** 2 - 2 * c * z[..., 0] * z[..., 1] + z[..., 1] ** 2) / (2 * one_m))
