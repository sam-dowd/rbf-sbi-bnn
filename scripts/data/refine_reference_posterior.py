"""Re-evaluate the RBF-kernel exact posterior of the 1,000 test datasets on a per-dataset zoomed grid, and
compare with the cached 240 x 240 global grid (data/reference_posterior/test_n100_ntest1000_seed0.npz).

Why: on the global grid 2 test datasets put 1.5% / 4.7% of their mass in the lowest rho cell (rho >= 0.02)
and 40 have a rho posterior narrower than 2 cells (paper/paper.tex, "Grid limitations").

Method: same likelihood and prior as src/data.py (data._loglik, data._logprior; untouched). For each test
dataset, a 300 x 300 grid over [m - 10 sd, m + 10 sd] per parameter, sd = max(coarse posterior sd, 2 coarse
cells), lower bounds floored at 1e-3 (below the global grid's 0.05 / 0.02). float64. log p(theta* | S) is
evaluated exactly at the true theta (unnormalised log posterior minus log of the grid normaliser), not
bilinearly interpolated. ASSUMPTION: +-10 coarse sd contains the posterior; checked by the mass in each
window's edge rows/columns (printed).

Analysis only (no model fit, no experiment-log entry). Output: results/reference_posterior_fine_test.npz
(mean, sd, corr, log_prob_true, window bounds, edge mass), a comparison table, and the BNN rho
width-to-warranted ratio recomputed against both references.

Usage: python scripts/data/refine_reference_posterior.py
"""
import sys
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))   # this repository's src/
from rbf_sbi.paths import ROOT, DATA, RESULTS, SCRATCH  # noqa: E402,F401
from rbf_sbi import data  # noqa: E402

COARSE = DATA / "reference_posterior" / "test_n100_ntest1000_seed0.npz"
OUT = RESULTS / "reference_posterior_fine_test.npz"
G, K_SD, FLOOR, BATCH = 300, 10.0, 1e-3, 10


def coarse_moments(ref):
    sg, rg = np.asarray(data.SIGMA_GRID, dtype=np.float64), np.asarray(data.RHO_GRID, dtype=np.float64)
    P = ref["post"].astype(np.float64); P /= P.sum((1, 2), keepdims=True)
    ps, pr = P.sum(2), P.sum(1)
    m = np.stack([(ps * sg).sum(1), (pr * rg).sum(1)], 1)
    sd = np.stack([np.sqrt((ps * (sg - m[:, :1]) ** 2).sum(1)), np.sqrt((pr * (rg - m[:, 1:]) ** 2).sum(1))], 1)
    cov = (P * (sg[None, :, None] - m[:, 0, None, None]) * (rg[None, None, :] - m[:, 1, None, None])).sum((1, 2))
    lp = np.asarray(data.compute_reference_log_prob_true(ref["post"], ref["theta"]))
    return m, sd, cov / (sd[:, 0] * sd[:, 1]), lp, (sg[1] - sg[0], rg[1] - rg[0])


@jax.jit
def logpost_batch(S, sgrid, rgrid):   # S (B,D,D); grids (B,G)
    f = lambda S1, s1, r1: jax.vmap(lambda s: jax.vmap(lambda r: data._loglik(s, r, S1) + data._logprior(s, r))(r1))(s1)
    return jax.vmap(f)(S, sgrid, rgrid)


@jax.jit
def logpost_points(S, theta):        # exact unnormalised log posterior at one theta per dataset
    return jax.vmap(lambda S1, t: data._loglik(t[0], t[1], S1) + data._logprior(t[0], t[1]))(S, theta)


def main():
    ref = np.load(COARSE)
    _, _, X_test, theta = data.load_dataset()
    assert np.allclose(ref["X"], X_test)
    S = jnp.asarray(X_test.reshape(-1, data.D, data.D), dtype=jnp.float64)
    m_c, sd_c, corr_c, lp_c, cell = coarse_moments(ref)
    sd_w = np.maximum(sd_c, 2 * np.asarray(cell))
    lo = np.maximum(m_c - K_SD * sd_w, FLOOR); hi = m_c + K_SD * sd_w
    n = len(theta)
    m_f, sd_f, corr_f, lp_f, edge = np.zeros((n, 2)), np.zeros((n, 2)), np.zeros(n), np.zeros(n), np.zeros(n)
    lp_true = np.asarray(logpost_points(S, jnp.asarray(theta, dtype=jnp.float64)))
    for b in range(0, n, BATCH):
        sl = slice(b, min(b + BATCH, n))
        sgr = np.linspace(lo[sl, 0], hi[sl, 0], G).T; rgr = np.linspace(lo[sl, 1], hi[sl, 1], G).T   # (B, G)
        L = np.asarray(logpost_batch(S[sl], jnp.asarray(sgr), jnp.asarray(rgr)))                    # (B, G, G)
        mx = L.max((1, 2), keepdims=True); W = np.exp(L - mx)
        Z = W.sum((1, 2)); P = W / Z[:, None, None]
        ds, dr = sgr[:, 1] - sgr[:, 0], rgr[:, 1] - rgr[:, 0]
        ps, pr = P.sum(2), P.sum(1)
        ms = (ps * sgr).sum(1); mr = (pr * rgr).sum(1)
        vs = (ps * (sgr - ms[:, None]) ** 2).sum(1); vr = (pr * (rgr - mr[:, None]) ** 2).sum(1)
        cv = (P * (sgr[:, :, None] - ms[:, None, None]) * (rgr[:, None, :] - mr[:, None, None])).sum((1, 2))
        m_f[sl] = np.stack([ms, mr], 1); sd_f[sl] = np.sqrt(np.stack([vs, vr], 1)); corr_f[sl] = cv / np.sqrt(vs * vr)
        # log density at truth: unnormalised log post minus log(sum W * ds * dr) - max
        lp_f[sl] = lp_true[sl] - (np.log(Z * ds * dr) + mx[:, 0, 0])
        # edge mass: outer rows/columns of the window, except where the window sits on the 1e-3 floor (a real bound)
        e = np.stack([ps[:, -1], pr[:, -1], np.where(lo[sl, 0] > FLOOR, ps[:, 0], 0), np.where(lo[sl, 1] > FLOOR, pr[:, 0], 0)], 1)
        edge[sl] = e.max(1)
    np.savez_compressed(OUT, mean=m_f, sd=sd_f, corr=corr_f, log_prob_true=lp_f, window_lo=lo, window_hi=hi,
                        edge_mass=edge, grid_points=G, k_sd=K_SD, floor=FLOOR, theta=theta)

    narrow = sd_c[:, 1] < 2 * cell[1]
    print(f"window check: max edge mass {edge.max():.1e}; windows on the 1e-3 floor: sigma {np.sum(lo[:,0] <= FLOOR)}, rho {np.sum(lo[:,1] <= FLOOR)}")
    print(f"{'':34s} {'coarse 240x240':>16s} {'fine (zoomed)':>16s}")
    for lab, c, f in [("RMS posterior sd sigma", np.sqrt((sd_c[:,0]**2).mean()), np.sqrt((sd_f[:,0]**2).mean())),
                      ("RMS posterior sd rho", np.sqrt((sd_c[:,1]**2).mean()), np.sqrt((sd_f[:,1]**2).mean())),
                      ("median posterior corr", np.median(corr_c), np.median(corr_f)),
                      ("reference test NLL (mean -log p)", -lp_c.mean(), -lp_f.mean()),
                      ("z-spread sigma", np.std((theta[:,0]-m_c[:,0])/sd_c[:,0]), np.std((theta[:,0]-m_f[:,0])/sd_f[:,0])),
                      ("z-spread rho", np.std((theta[:,1]-m_c[:,1])/sd_c[:,1]), np.std((theta[:,1]-m_f[:,1])/sd_f[:,1])),
                      ("frac |z|<1 rho", np.mean(np.abs((theta[:,1]-m_c[:,1])/sd_c[:,1])<1), np.mean(np.abs((theta[:,1]-m_f[:,1])/sd_f[:,1])<1))]:
        print(f"{lab:34s} {c:16.4f} {f:16.4f}")
    r = sd_f / sd_c
    print(f"fine/coarse sd ratio, all: sigma median {np.median(r[:,0]):.4f} [min {r[:,0].min():.3f}, max {r[:,0].max():.3f}]; "
          f"rho median {np.median(r[:,1]):.4f} [min {r[:,1].min():.3f}, max {r[:,1].max():.3f}]")
    print(f"the {narrow.sum()} narrow-rho datasets: rho sd coarse median {np.median(sd_c[narrow,1]):.4f} -> fine {np.median(sd_f[narrow,1]):.4f}; "
          f"|log p diff| median {np.median(np.abs(lp_f-lp_c)[narrow]):.3f}, max {np.abs(lp_f-lp_c)[narrow].max():.3f}")
    print(f"other datasets: |log p diff| median {np.median(np.abs(lp_f-lp_c)[~narrow]):.4f}, max {np.abs(lp_f-lp_c)[~narrow].max():.3f}")
    for i in np.argsort(-np.abs(lp_f - lp_c))[:5]:
        print(f"  largest change: test {i} theta {theta[i].round(3)} rho sd {sd_c[i,1]:.4f}->{sd_f[i,1]:.4f}  log p {lp_c[i]:.2f}->{lp_f[i]:.2f}")

    # BNN width relative to warranted width, rho, against both references (stock N(0,1) sweep, 5 repeats)
    import os
    print("\nBNN rho width / warranted width, mean over test points and repeats (coarse | fine | coarse w/o narrow-40):")
    for nn in (50, 100, 500, 1000):
        vals = []
        for rr in range(5):
            f = next(p for p in (RESULTS / f"raw/sweep_bnn_vanilla_hpnew_n{nn}_r{rr}.npz", RESULTS / f"raw/sweep_bnn_vanilla_n{nn}_r{rr}.npz") if p.exists())
            d = np.load(f); w = np.sqrt(d["aleatoric"][:, 1]**2 + d["epistemic"][:, 1]**2); mu = d["mean"][:, 1]
            wc = w / np.sqrt(sd_c[:, 1]**2 + (mu - m_c[:, 1])**2); wf = w / np.sqrt(sd_f[:, 1]**2 + (mu - m_f[:, 1])**2)
            vals.append([wc.mean(), wf.mean(), wc[~narrow].mean(), (w / sd_c[:, 1]).mean(), (w / sd_f[:, 1]).mean()])
        v = np.mean(vals, 0)
        print(f"  n={nn:5d}: warranted ratio {v[0]:.3f} | {v[1]:.3f} | {v[2]:.3f}    (width/exact sd: coarse {v[3]:.2f}, fine {v[4]:.2f})")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
