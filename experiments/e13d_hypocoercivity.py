"""E13d: the hypocoercive estimates that close step 1 of Proposition F.1, checked on the r-chain.

In time units 1/g and with alpha = a_k / g, the two-point function of the dephased XX chain at centre-of-mass momentum k
obeys dc/dt = M c, M = -alpha (S - S*) - (1 - P), on the relative distance r (S shifts r -> r + 1, P projects onto
r = 0, Q = 1 - P). The modified energy H = |c|^2 / 2 + (1/5) Re <A c, c>, A = P T* / (1 + 2 alpha^2), T = alpha (S - S*),
gives the estimates below with lambda = alpha^2 / (6 (1 + 2 alpha^2)), uniformly in alpha (docs/theory/phase4-theory.md):
  (1) ||e^{tM}||              <= sqrt(3/2) e^{-lambda t}            (all initial data)
  (2) ||Q e^{tM} e_0||        <= 2 |alpha| e^{-lambda t}            (coherences of a density mode)
  (3) ||P e^{tM} Q||          <= 2 |alpha| e^{-lambda t}            (density fed by coherences; duality)
  (4) ||Q e^{tM} Q||          <= e^{-t} + 4 alpha^2 e^{-lambda t}
  (5) ||Q d_alpha e^{tM} e_0|| <= (3 + 16 alpha^2 t) e^{-lambda t}   (the k-derivative, sector by sector)
  (6) int |G_k(t)| dk / 2 pi  <= sqrt(3/2) [min(1, (3/2) sqrt(pi / (D t))) + e^{-t/18}],  D = 2 (J/g)^2,
      with G_k(t) = <e_0, e^{tM} e_0> at alpha = (2J/g) sin(k/2).
Each is a proven inequality for the chain truncated to |r| <= R as well, so the check needs no truncation error
estimate. Exact matrix exponentials (scaling and squaring), the alpha derivative by central differences of them
(step 1e-5 alpha); alpha from 0.01 to 10, t up to 15/lambda; (6) for J/g = 0.25 (E13) and 1.
Fixed before the run: all six hold at every alpha and t to a relative tolerance of 1e-9.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import sys

import numpy as np
from scipy.linalg import expm
from scipy.sparse import diags

from common import CAT, INK, MUTED, RESULTS, save_figure, save_json, style

ALPHAS = np.logspace(-2, 1, 31)
N_T, TOL = 40, 1e-9
J_OVER_G = (0.25, 1.0)


def rate(alpha):
    return alpha**2 / (6.0 * (1.0 + 2.0 * alpha**2))


def chain(alpha, R):
    """M and the shift difference S - S* on |r| <= R (index R is r = 0)."""
    n = 2 * R + 1
    K = diags([np.ones(n - 1), -np.ones(n - 1)], [-1, 1]).toarray()  # (S - S*) c at r is c_{r-1} - c_{r+1}
    M = -alpha * K - np.diag(np.where(np.arange(n) == R, 0.0, 1.0))
    return M, K


def operator_checks(alpha):
    """max over t of (measured / bound) for (1)-(5)."""
    R = int(60 + 10 * alpha)
    M, K = chain(alpha, R)
    n, lam, h = M.shape[0], rate(alpha), 1e-5 * alpha
    q = np.arange(n) != R
    worst = np.zeros(5)
    for t in np.logspace(-2, np.log10(15.0 / lam), N_T):
        E = expm(t * M)
        dE0 = (expm(t * (M - h * K))[:, R] - expm(t * (M + h * K))[:, R]) / (2 * h)  # M(alpha +- h) = M -+ h K
        decay = np.exp(-lam * t)
        meas = [np.linalg.norm(E, 2), np.linalg.norm(E[q, R]), np.linalg.norm(E[R, q]),
                np.linalg.norm(E[np.ix_(q, q)], 2), np.linalg.norm(dE0[q])]
        bound = [np.sqrt(1.5) * decay, 2 * alpha * decay, 2 * alpha * decay, np.exp(-t) + 4 * alpha**2 * decay,
                 (3 + 16 * alpha**2 * t) * decay]
        worst = np.maximum(worst, np.array(meas) / np.array(bound))
    return worst


def true_rate(alpha):
    """Slowest decay rate of the infinite chain: the slow pole for 2 alpha < 1, else the pole pair at Re p = -1."""
    return 1.0 - np.sqrt(1.0 - 4 * alpha**2) if 2 * alpha < 1 else 1.0


def integrated_check(jg, times):
    """int |G_k(t)| dk / 2 pi by Gauss-Legendre in s with k = pi s^2, against the bound (6)."""
    s, w = np.polynomial.legendre.leggauss(200)
    s, w = 0.5 * (s + 1), 0.5 * w
    k = np.pi * s**2
    alphas = 2 * jg * np.sin(k / 2)
    G = np.empty((len(k), len(times)))
    for m, a in enumerate(alphas):
        R = int(60 + 10 * a)
        M, _ = chain(a, R)
        G[m] = [expm(t * M)[R, R] for t in times]
    integral = (2 * np.pi * s * w) @ np.abs(G) / np.pi  # (1/pi) int_0^pi dk = int_{-pi}^{pi} dk / 2 pi
    D = 2 * jg**2
    bound = np.sqrt(1.5) * (np.minimum(1.0, 1.5 * np.sqrt(np.pi / (D * times))) + np.exp(-times / 18))
    return integral, bound


def run():
    worst = np.array([operator_checks(a) for a in ALPHAS])
    integ = {}
    for jg in J_OVER_G:
        times = np.logspace(-1, np.log10(4e3 / (2 * jg**2)), 50)
        integ[jg] = (times, *integrated_check(jg, times))
    return worst, integ


def analyse(worst, integ):
    names = ["(1) ||e^{tM}||", "(2) ||Q e^{tM} e_0||", "(3) ||P e^{tM} Q||", "(4) ||Q e^{tM} Q||",
             "(5) ||Q d_alpha e^{tM} e_0||"]
    out = {f"{n}: max measured / bound over alpha and t": float(worst[:, i].max()) for i, n in enumerate(names)}
    for jg, (times, actual, bound) in integ.items():
        out[f"(6) J/g = {jg:g}: max measured / bound over t"] = float((actual / bound).max())
    ratio = np.array([rate(a) / true_rate(a) for a in ALPHAS])
    out["lambda / true slowest rate: min, max over alpha"] = [float(ratio.min()), float(ratio.max())]
    out["acceptance"] = {"all six estimates hold to 1e-9": bool(
        worst.max() <= 1 + TOL and all((a / b).max() <= 1 + TOL for _, a, b in integ.values()))}
    return out, names


def plot(worst, integ, names):
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7))
    ax = axes[0]
    looks = [(CAT[0], "o", "-"), (CAT[1], "o", "-"), (CAT[1], "x", "--"), (CAT[2], "o", "-"), (INK, "s", ":")]
    for i, (n, (c, mk, ls)) in enumerate(zip(names, looks)):  # (2) and (3) coincide: e^{tM^T} is the chain at -alpha
        ax.semilogx(ALPHAS, worst[:, i], marker=mk, ls=ls, ms=3.5, lw=1, color=c, label=n)
    ax.axhline(1.0, color=INK, lw=0.8, ls=":")
    ax.set_xlabel(r"$\alpha = a_k / g$")
    ax.set_ylabel("max$_t$ measured / bound")
    ax.set_title("(a) operator estimates (1)-(5)", fontsize=9)
    ax.legend(fontsize=6.5)
    ax = axes[1]
    ax.loglog(ALPHAS, [rate(a) for a in ALPHAS], color=CAT[0], lw=1.6, label=r"$\lambda$ (hypocoercive)")
    ax.loglog(ALPHAS, [true_rate(a) for a in ALPHAS], color=INK, lw=1.2, ls="--", label="slowest rate, infinite chain")
    ax.set_xlabel(r"$\alpha = a_k / g$")
    ax.set_ylabel("decay rate (units of g)")
    ax.set_title(r"(b) uniform rate $\lambda \geq \min(\alpha^2, 1)/18$", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[2]
    for i, (jg, (times, actual, bound)) in enumerate(integ.items()):
        Dt = 2 * jg**2 * times
        ax.loglog(Dt, actual, color=CAT[i], lw=1.6, label=f"$\\int|G_k|dk/2\\pi$, J/g = {jg:g}")
        ax.loglog(Dt, bound, color=CAT[i], lw=1.0, ls="--", label=f"bound (6), J/g = {jg:g}")
    ax.loglog(Dt, 0.5 * Dt**-0.5, color=MUTED, lw=0.8, ls=":", label=r"$(Dt)^{-1/2}$")
    ax.set_xlabel("D t")
    ax.set_title("(c) integrated estimate (6)", fontsize=9)
    ax.legend(fontsize=6.5)
    fig.suptitle("E13d: hypocoercive estimates for the dephased XX r-chain (exact matrix exponentials)")
    fig.tight_layout()
    return save_figure(fig, "e13d_hypocoercivity")


if __name__ == "__main__":
    if "--replot" not in sys.argv:
        worst, integ = run()
        np.savez_compressed(RESULTS / "e13d_hypocoercivity.npz", worst=worst,
                            **{f"{jg:g}|{k}": v for jg, r in integ.items() for k, v in zip(("times", "actual", "bound"), r)})
    z = np.load(RESULTS / "e13d_hypocoercivity.npz")
    integ = {jg: tuple(z[f"{jg:g}|{k}"] for k in ("times", "actual", "bound")) for jg in J_OVER_G}
    out, names = analyse(z["worst"], integ)
    save_json(RESULTS / "e13d_hypocoercivity.json", out)
    print(plot(z["worst"], integ, names))
    for key, v in out.items():
        print(f"{key}: {v}")
