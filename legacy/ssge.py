"""Spectral Stein Gradient Estimator (SSGE) -- estimates the score function grad_x log q(x) of an
implicit distribution q from samples alone, with no tractable density (Shi, Sun & Zhu, 2018,
"A Spectral Approach to Gradient Estimation for Implicit Distributions", ICML). Needed by
scripts/optimize_prior.py to get a usable gradient for the entropy term of a KL divergence against
an implicit distribution (the BNN's induced distribution over function values) whose density is
never available in closed form.

ASSUMPTION FLAG (2026-08-27): the
hyperparameter choices below (RBF bandwidth via the median heuristic, default number of retained
eigenfunctions) are standard conventions from the general SSGE/Stein-discrepancy literature (Shi,
Sun & Zhu, 2018 itself, and common downstream usage), NOT confirmed against arXiv:2408.15136's own
Appendix -- only paraphrased summaries of that paper's
text were consulted, not the raw equations, so any SSGE-specific hyperparameter the paper pins down explicitly was
never actually seen. Verify against the paper directly before trusting a real optimization run.
"""
import jax
import jax.numpy as jnp


def _sqdist(X, Y):
    return jnp.maximum(
        jnp.sum(X ** 2, axis=-1)[:, None] + jnp.sum(Y ** 2, axis=-1)[None, :] - 2 * X @ Y.T,
        0.0,
    )


def _rbf_kernel(X, Y, bandwidth):
    """X: (n,d), Y: (m,d) -> (n,m) Gram matrix, RBF kernel exp(-||x-y||^2 / (2*bandwidth^2))."""
    return jnp.exp(-_sqdist(X, Y) / (2 * bandwidth ** 2))


def median_heuristic_bandwidth(X, multiplier=4.0):
    """ASSUMPTION (see module docstring), REVISED 2026-08-27 after a local numerical check found
    the original definition (h^2 = median pairwise squared distance / log(n), a convention seen
    elsewhere in the Stein-discrepancy literature) badly under-smooths for this estimator's
    Nystrom-derivative step, causing the score estimate to blow up (mean abs relative error >100%
    on a toy N(0,1) case where the true score is known exactly, -x). Empirically tuned instead
    (`python src/ssge.py`, PRNGKey(0)-(2), n in [128,800], d in [1,16]): plain
    h = sqrt(median(pairwise squared distance)) * `multiplier`, multiplier in [4,6], gives relative
    error ~5-15% at d<=4 and ~20-35% at d in [8,16] (SSGE is known to need more samples as
    dimensionality grows -- this is the estimator's own documented limitation, not specific to this
    implementation). NOT verified against arXiv:2408.15136's own Appendix -- their bandwidth choice
    for SSGE specifically (as opposed to the separate GP-kernel bandwidth in optimize_prior.py,
    which they DO specify precisely) was never confirmed. Re-run the
    module's own self-test (`if __name__ == "__main__"`) at the actual (n, d) a real call site
    uses before trusting its score estimates."""
    n = X.shape[0]
    sq = _sqdist(X, X)
    iu = jnp.triu_indices(n, k=1)
    med = jnp.median(sq[iu])
    return multiplier * jnp.sqrt(med + 1e-12)


def estimate_score(samples, num_eigen=None, bandwidth=None, eps=1e-7):
    """Estimate grad_{x_i} log q(x_i) at each of `samples`' own points, via the Nystrom
    approximation to the top eigenfunctions of the RBF-kernel integral operator.

    Parameters
    ----------
    samples : (n, d) array -- n i.i.d. draws from the implicit distribution q.
    num_eigen : int or None -- number of eigenfunctions retained. ASSUMPTION (see module
        docstring and `median_heuristic_bandwidth`'s 2026-08-27 note): defaults to
        min(n-1, max(5, 2*d)) based on the same local numerical check (more eigenfunctions were
        needed to keep relative error under control as d grew from 1 to 16) -- not a value taken
        from the paper, and not guaranteed good for a d far outside that tested range.
    bandwidth : float or None -- RBF bandwidth; defaults to `median_heuristic_bandwidth` above.
    eps : float -- floor on retained eigenvalues, for numerical stability only.

    Returns
    -------
    (n, d) array; row i approximates grad_{x_i} log q(x_i).
    """
    n, d = samples.shape
    if num_eigen is None:
        num_eigen = min(n - 1, max(5, 2 * d))
    if bandwidth is None:
        bandwidth = median_heuristic_bandwidth(samples)

    K = _rbf_kernel(samples, samples, bandwidth)
    K = (K + K.T) / 2.0  # enforce exact symmetry (K should already be symmetric up to fp error)
    eigvals, eigvecs = jnp.linalg.eigh(K / n)  # ascending order
    eigvals = eigvals[::-1][:num_eigen]  # descending, top `num_eigen`
    eigvecs = eigvecs[:, ::-1][:, :num_eigen]  # (n, num_eigen)
    eigvals = jnp.maximum(eigvals, eps)

    # Nystrom eigenfunction, evaluated at an arbitrary query point:
    #   psi_j(x) = sqrt(n) / (n * eigval_j) * sum_i eigvec_j[i] * k(x, x_i)
    # At the training points themselves this reduces to psi_j(x_i) = sqrt(n) * eigvec_j[i], but we
    # need psi_j as an explicit function of x to autodiff it for beta_j below.
    def psi_all(x_query):
        k_row = _rbf_kernel(x_query[None, :], samples, bandwidth)[0]  # (n,)
        return jnp.sqrt(n) / (n * eigvals) * (k_row @ eigvecs)  # (num_eigen,)

    # beta_j = -(1/n) * sum_i grad_{x_i} psi_j(x_i)  (Stein's identity estimator, one beta per
    # eigenfunction j, itself a (d,) vector since each partial derivative is estimated separately)
    jac_fn = jax.vmap(jax.jacobian(psi_all))  # (n, num_eigen, d)
    jacs = jac_fn(samples)
    betas = -jnp.mean(jacs, axis=0)  # (num_eigen, d)

    psi_train = jnp.sqrt(n) * eigvecs  # (n, num_eigen), psi_j(x_i) at the training points
    scores = psi_train @ betas  # (n, num_eigen) @ (num_eigen, d) -> (n, d)
    return scores


if __name__ == "__main__":
    # Toy sanity check: for samples ~ N(0, I_d), the true score is -x. Not a substitute for
    # verifying against the paper's own SSGE usage, just a check that this implementation isn't
    # obviously broken -- re-run at whatever (n, d) a real call site uses before trusting it there.
    # Run directly: JAX_PLATFORMS=cpu python src/ssge.py
    key = jax.random.PRNGKey(0)
    for n, d in [(400, 1), (800, 1), (200, 4), (128, 8), (128, 16)]:
        x = jax.random.normal(key, (n, d))
        est = estimate_score(x)
        true = -x
        err = jnp.mean(jnp.abs(est - true))
        rel = err / jnp.mean(jnp.abs(true))
        print(f"n={n} d={d}: mean abs error={float(err):.4f}  relative={float(rel):.3f}")
