"""E13e: the weighted hypocoercive estimates that close the l1 sum over r in m(u) (Proposition F.1), checked.

Notation of E13d: time in units of 1/g, alpha = a_k / g, dc/dt = M c with M = -alpha K - Q, K = S - S*. The weight
W = diag(e^{theta |r|}) with s = sinh(theta / 2) = 1 / (40 (1 + alpha_m)), alpha_m = 2J/g, and
lambda_theta = alpha^2 / (13 (1 + 2 alpha^2)) (docs/theory/phase4-theory.md, Lemma F.3) give, for |alpha| <= alpha_m:
  (1)  ||W^1/2 e^{tM} W^-1/2||          <= sqrt(3/2) e^{-lambda_theta t}       (also with theta -> -theta)
  (2)  ||W^1/2 Q e^{tM} e_0||           <= 2 |alpha| e^{-lambda_theta t}
  (3)  ||P W^1/2 e^{tM} W^-1/2 Q||      <= 2 |alpha| e^{-lambda_theta t}
  (4)  ||Q W^1/2 e^{tM} W^-1/2 Q||      <= e^{-0.95 t} + 3.2 alpha^2 e^{-lambda_theta t}
  (5)  ||W^1/2 Q d_alpha e^{tM} e_0||   <= (3 + 14 alpha^2 t) e^{-lambda_theta t}
  (5') |P d_alpha e^{tM} e_0|           <= 8.6 |alpha| t e^{-lambda_theta t}
  (6)  sum_r |(e^{tM} e_0)_r|           <= coth(theta/2)^1/2 sqrt(3/2) e^{-lambda_theta t}   (l1 over r)
All are proven for the chain truncated to |r| <= R as well. Exact matrix exponentials, the alpha derivative by
central differences (step 1e-5 alpha), alpha from 0.01 to alpha_m, t up to 15 / lambda_theta, J/g = 0.25 and 1.
(7) m(u) = sum_{x != y} |M_xy(u)| of E13 (N = 400, J = g = 1, eps = 0.01, O = n_C) against three curves:
    the Carlson step evaluated on the exact r-chain, m_C(u) = sum_{r != 0} [sqrt 2 X_r + sqrt(2 pi) (X_r Y_r)^1/2] with
    X_r, Y_r the L2(dk) norms of M_r(k, u) and d_k M_r(k, u); and the closed bound m_B(u) built from (1), (5), (5').
Fixed before the run: (1)-(6) hold at every alpha and t to a relative tolerance of 1e-9, and m <= m_C <= m_B on the
E13 grid; the log-slope of m_B on u in [20, 800] is reported (expected -1/2).
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import sys

import numpy as np
from scipy.linalg import expm

from common import CAT, INK, MUTED, ORDINAL, RESULTS, save_figure, save_json, style
from e13d_hypocoercivity import chain

J_OVER_G, N_ALPHA, N_T, TOL = (0.25, 1.0), 25, 40, 1e-9
EPS = 0.01  # E13: J = g = 1


def s_of(alpha_m):
    return 1.0 / (40.0 * (1.0 + alpha_m))


def lam(alpha):
    return alpha**2 / (13.0 * (1.0 + 2.0 * alpha**2))


def weights(R, theta):
    r = np.abs(np.arange(-R, R + 1))
    return np.exp(0.5 * theta * r), np.exp(-theta * r)


def operator_checks(alpha, alpha_m):
    """max over t of measured / bound for (1)-(6) at one alpha; (1) for both signs of theta."""
    R = int(60 + 10 * alpha)
    M, K = chain(alpha, R)
    theta = 2.0 * np.arcsinh(s_of(alpha_m))
    w, _ = weights(R, theta)
    n, la, h = M.shape[0], lam(alpha), 1e-5 * alpha
    q = np.arange(n) != R
    coth = 1.0 / np.tanh(theta / 2)
    worst = np.zeros(8)
    for t in np.logspace(-2, np.log10(15.0 / la), N_T):
        E = expm(t * M)
        Ew = (w[:, None] * E) / w[None, :]
        Emw = (E / w[:, None]) * w[None, :]                      # theta -> -theta
        dE0 = (expm(t * (M - h * K))[:, R] - expm(t * (M + h * K))[:, R]) / (2 * h)
        dec = np.exp(-la * t)
        meas = [np.linalg.norm(Ew, 2), np.linalg.norm(Emw, 2), np.linalg.norm(w[q] * E[q, R]),
                np.linalg.norm(Ew[R, q]), np.linalg.norm(Ew[np.ix_(q, q)], 2), np.linalg.norm(w[q] * dE0[q]),
                abs(dE0[R]), np.abs(E[:, R]).sum()]
        bound = [np.sqrt(1.5) * dec, np.sqrt(1.5) * dec, 2 * alpha * dec, 2 * alpha * dec,
                 np.exp(-0.95 * t) + 3.2 * alpha**2 * dec, (3 + 14 * alpha**2 * t) * dec, 8.6 * alpha * t * dec,
                 np.sqrt(coth * 1.5) * dec]
        worst = np.maximum(worst, np.array(meas) / np.array(bound))
    return worst


def m_curves(U):
    """m_C(u) from the exact r-chain (Heisenberg picture, a_k -> -a_k) and the closed bound m_B(u); J = g = 1."""
    s, wq = np.polynomial.legendre.leggauss(160)
    s, wq = 0.5 * (s + 1), 0.5 * wq
    k = np.pi * s**2
    dk = 2 * np.pi * s * wq / np.pi                           # (1/pi) int_0^pi dk = int_{-pi}^{pi} dk / 2 pi
    a, da = 2 * np.sin(k / 2), np.cos(k / 2)                  # a_k and d a_k / dk (J = 1)
    alpha_m = 2.0
    theta = 2.0 * np.arcsinh(s_of(alpha_m))
    Rmax = int(60 + 10 * alpha_m)
    X2 = np.zeros((len(U), 2 * Rmax + 1))
    Y2 = np.zeros_like(X2)
    for m, (al, dal) in enumerate(zip(a, da)):
        R = int(60 + 10 * al)
        M, K = chain(-al, R)                                  # the Heisenberg chain
        h = 1e-5 * al
        sl = slice(Rmax - R, Rmax + R + 1)
        for j, u in enumerate(U):
            w0 = expm(u * M)[:, R]
            dw = (expm(u * (M + h * K))[:, R] - expm(u * (M - h * K))[:, R]) / (2 * h)  # d/d(alpha) of chain(-alpha)
            F = EPS * al * (K @ w0)
            dF = EPS * (dal * (K @ w0) + al * (K @ (dal * dw)))
            X2[j, sl] += dk[m] * np.abs(F) ** 2
            Y2[j, sl] += dk[m] * np.abs(dF) ** 2
    # k in [0, pi] only: F_r(-k) and F_{-r}(k) have the same modulus (reflection), so symmetrise in r
    X2, Y2 = 0.5 * (X2 + X2[:, ::-1]), 0.5 * (Y2 + Y2[:, ::-1])
    q = np.arange(2 * Rmax + 1) != Rmax
    X, Y = np.sqrt(X2[:, q]), np.sqrt(Y2[:, q])
    mC = (np.sqrt(2) * X + np.sqrt(2 * np.pi) * np.sqrt(X * Y)).sum(axis=1)
    return mC, m_bound(U)


def m_bound(U, n_k=4000):
    """Closed bound m_B(u): the pointwise estimates (1), (5), (5') integrated over k; J = g = 1, alpha_m = 2."""
    s, wq = np.polynomial.legendre.leggauss(n_k)
    s, wq = 0.5 * (s + 1), 0.5 * wq
    k = np.pi * s**2
    dk = 2 * s * wq
    a, da = 2 * np.sin(k / 2), np.cos(k / 2)
    theta = 2.0 * np.arcsinh(s_of(2.0))
    S = 2.0 / np.expm1(theta)
    la = lam(a)[None, :]
    tau = np.asarray(U, float)[:, None]
    Xb = EPS * np.sqrt((dk * (a**2 * 4 * np.exp(theta) * 1.5 * np.exp(-2 * la * tau))).sum(axis=1))
    yb = 2 * EPS * np.exp(theta / 2) * np.abs(da) * (np.sqrt(1.5) + a * (3 + 8.6 * a * tau + 14 * a**2 * tau)) \
        * np.exp(-la * tau)
    Yb = np.sqrt((dk * yb**2).sum(axis=1))
    return np.sqrt(2 * S) * Xb + np.sqrt(2 * np.pi * S) * np.sqrt(Xb * Yb)


def run():
    worst = {jg: np.array([operator_checks(al, 2 * jg) for al in np.logspace(-2, np.log10(2 * jg), N_ALPHA)])
             for jg in J_OVER_G}
    z = np.load(RESULTS / "e13_two_sided_bound.npz")
    U = np.concatenate([[0.0], np.geomspace(0.01, 1500.0, 140)])
    m = z["obs"][:, 1]
    mC, mB = m_curves(U)
    return worst, U, m, mC, mB


def slope(x, y, lo, hi):
    sel = (x >= lo) & (x <= hi) & (y > 0)
    return float(np.polyfit(np.log(x[sel]), np.log(y[sel]), 1)[0])


NAMES = ["(1) weighted ||e^{tM}||", "(1-) the same, theta -> -theta", "(2) weighted ||Q e^{tM} e_0||",
         "(3) weighted ||P e^{tM} Q||", "(4) weighted ||Q e^{tM} Q||", "(5) weighted ||Q d_alpha e^{tM} e_0||",
         "(5') |P d_alpha e^{tM} e_0|", "(6) l1 over r of e^{tM} e_0"]


def analyse(worst, U, m, mC, mB):
    out = {}
    for jg, w in worst.items():
        out[f"J/g = {jg:g} (alpha_m = {2 * jg:g}, theta = {2 * np.arcsinh(s_of(2 * jg)):.4f})"] = {
            f"{n}: max measured / bound": float(w[:, i].max()) for i, n in enumerate(NAMES)}
    pos = U > 0
    out["(7) m(u) of E13 against the Carlson step m_C and the closed bound m_B"] = {
        "max m / m_C": float((m[pos] / mC[pos]).max()), "max m_C / m_B": float((mC[pos] / mB[pos]).max()),
        "m_B / m at u = 1, 100, 1500": [float(np.interp(u, U, mB / np.where(m > 0, m, np.nan))) for u in (1, 100, 1500)],
        "log-slopes on u in [20, 800]: m, m_C, m_B": [slope(U, m, 20, 800), slope(U, mC, 20, 800),
                                                     slope(U, mB, 20, 800)]}
    far = np.geomspace(1e3, 1e8, 41)
    mBf = m_bound(far)
    out["(7) m(u) of E13 against the Carlson step m_C and the closed bound m_B"].update({
        "m_B log-slope on u in [1e5, 1e6] and [1e7, 1e8] (added after the run: the bound's hump)": [
            slope(far, mBf, 1e5, 1e6), slope(far, mBf, 1e7, 1e8)],
        "u of the largest m_B * u^(1/2) on [1e3, 1e8]": float(far[np.argmax(mBf * np.sqrt(far))])})
    ok_ops = all(w.max() <= 1 + TOL for w in worst.values())
    ok_m = bool(np.all(m[pos] <= mC[pos] * (1 + TOL)) and np.all(mC[pos] <= mB[pos] * (1 + TOL)))
    out["acceptance"] = {"(1)-(6) hold to 1e-9": bool(ok_ops), "m <= m_C <= m_B on the E13 grid": ok_m}
    return out


def plot(worst, U, m, mC, mB):
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
    ax = axes[0]
    colors = [CAT[0], ORDINAL[0], CAT[1], ORDINAL[3], CAT[2], INK, MUTED, ORDINAL[2]]
    for ls, (jg, w) in zip(("-", "--"), worst.items()):
        al = np.logspace(-2, np.log10(2 * jg), N_ALPHA)
        for i, c in enumerate(colors):
            ax.semilogx(al, w[:, i], ls=ls, lw=1.1, color=c, label=NAMES[i].split(" ")[0] if ls == "-" else None)
    ax.plot([], [], color=MUTED, ls="-", label="J/g = 0.25")
    ax.plot([], [], color=MUTED, ls="--", label="J/g = 1")
    ax.axhline(1.0, color=INK, lw=0.8, ls=":")
    ax.set_xlabel(r"$\alpha = a_k / g$")
    ax.set_ylabel("max$_t$ measured / bound")
    ax.set_title("(a) weighted estimates (1)-(6); (1) and (1-) coincide", fontsize=9)
    ax.set_ylim(-0.36, 1.06)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.legend(fontsize=6.5, ncol=5, loc="lower center")
    ax = axes[1]
    pos = U > 0
    ax.loglog(U[pos], m[pos], color=INK, lw=1.6, label="m(u), E13 (N = 400)")
    ax.loglog(U[pos], mC[pos], color=CAT[0], lw=1.4, ls="--", label="Carlson step on the exact r-chain")
    ax.loglog(U[pos], mB[pos], color=CAT[1], lw=1.4, label="closed bound (proof constants)")
    ax.loglog(U[pos], mB[pos][-1] * (U[pos] / U[pos][-1]) ** -0.5, color=MUTED, lw=0.8, ls=":", label=r"$u^{-1/2}$")
    ax.set_xlabel("u (units of 1/g)")
    ax.set_title("(b) the l1 sum over r and x", fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle("E13e: weighted hypocoercivity closes the l1 sum of Proposition F.1")
    fig.tight_layout()
    return save_figure(fig, "e13e_weighted_l1")


if __name__ == "__main__":
    if "--replot" not in sys.argv:
        worst, U, m, mC, mB = run()
        np.savez_compressed(RESULTS / "e13e_weighted_l1.npz", U=U, m=m, mC=mC, mB=mB,
                            **{f"worst_{jg:g}": w for jg, w in worst.items()})
    z = np.load(RESULTS / "e13e_weighted_l1.npz")
    worst = {jg: z[f"worst_{jg:g}"] for jg in J_OVER_G}
    out = analyse(worst, z["U"], z["m"], z["mC"], z["mB"])
    save_json(RESULTS / "e13e_weighted_l1.json", out)
    print(plot(worst, z["U"], z["m"], z["mC"], z["mB"]))
    for key, v in out.items():
        print(f"{key}: {v}")
