"""E13: a time-uniform local bound under charge conservation, from forgetting in both pictures (Theorem F).

Model as E10: XX chain, N = 400, hopping -1, local number dephasing g = 1 (diffusion constant D = 2), uniform hopping
miscalibration dH = eps sum_k (c_k^dag c_k+1 + h.c.), eps = 0.01; observable n_C at the centre (monopole).
The Heisenberg-only bound of Theorems A and E, int_0^t ||[dH, O_u]|| du, grows as sqrt(t). Theorem F pairs the
reference Heisenberg factors of i[dH, O_u] with factors of the perturbed state (qerr.hydro):
    |Delta(t)| <= F(t) = int_0^t [a(u) gamma(t - u) + m(u) kappa(t - u)] du,
a(u): antisymmetric nearest-neighbour part of the coefficient matrix of O_u (times eps), m(u): off-diagonal part of
the commutator; gamma(s): largest density gradient, kappa(s): largest coherence of the perturbed state.
Initial product states with bounded density: density waves (crest at C, amplitude 0.4, lambda = 20..160), domain walls
(filled for x < C - d, d = 10, 40), the Neel state and random occupations. Checks: F >= |error at C| at all times;
F saturates while the Theorem-A bound grows as sqrt(t); the factors decay as u^(-1/2) and s^(-1/2).
All operators and states are quadratic (second moments): exact single-particle computations.
"""

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.integrate import cumulative_trapezoid

from common import CAT, INK, MUTED, ORDINAL, RESULTS, save_figure, save_json, style
from qerr.hydro import commutator_coefficients, evolve_two_point, observable_factors, state_factors

N, G_DEPH, EPS = 400, 1.0, 0.01
C = N // 2
T_MAX = 1500.0
U = np.concatenate([[0.0], np.geomspace(0.01, T_MAX, 140)])
STATES = ["wave 20", "wave 40", "wave 80", "wave 160", "wall 10", "wall 40", "Neel", "random"]
TOL = {"rtol": 1e-9, "atol": 1e-12}


def hopping(amp):
    h = np.zeros((N, N), complex)
    for i in range(N - 1):
        h[i, i + 1] = h[i + 1, i] = amp
    return h


def occupations(name):
    x = np.arange(N)
    kind, _, arg = name.partition(" ")
    if kind == "wave":
        return 0.5 + 0.4 * np.cos(2 * np.pi * (x - C) / float(arg))
    if kind == "wall":
        return (x < C - int(arg)).astype(float)
    if kind == "Neel":
        return ((x + 1) % 2).astype(float)
    return (np.random.default_rng(7).random(N) < 0.5).astype(float)


def observable_job(_=None):
    """a(u), m(u) and the Theorem-A integrand f(u) = ||i[dH, O_u]|| for O = n_C."""
    w0 = np.zeros((N, N), complex)
    w0[C, C] = 1.0
    W = evolve_two_point(-np.ones(N - 1), G_DEPH * np.ones(N), w0, U, **TOL)
    dh = hopping(EPS)
    out = []
    for w in W:
        M = commutator_coefficients(dh, w)
        a, m = observable_factors(w, dh)
        f = 0.5 * np.abs(np.linalg.eigvalsh(0.5 * (M + M.conj().T))).sum()  # ||sum M c^dag c||, Tr M = 0
        grad = float(np.abs(np.diff(np.real(np.diag(w)))).sum())
        out.append((a, m, f, grad))
    return np.array(out)


def state_job(args):
    """Two-point evolution: n(s) at C and the state factors at every grid time."""
    name, eps = args
    Cs = evolve_two_point((-1.0 + eps) * np.ones(N - 1), G_DEPH * np.ones(N),
                          np.diag(occupations(name)).astype(complex), U, **TOL)
    rows = []
    for Cm in Cs:
        gamma, kappa = state_factors(Cm)
        rows.append((np.real(Cm[C, C]), gamma, kappa))
    return np.array(rows)


def convolve(fa, fb, t, n=600):
    """int_0^t fa(u) fb(t - u) du, nodes clustered at both ends (Chebyshev), functions interpolated on U."""
    th = np.linspace(0.0, np.pi, n)
    u = 0.5 * t * (1.0 - np.cos(th))
    du = 0.5 * t * np.sin(th)
    vals = np.interp(u, U, fa) * np.interp(t - u, U, fb) * du
    return float(np.trapezoid(vals, th))


def slope(x, y, lo, hi):
    ok = (x >= lo) & (x <= hi) & (y > 0)
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


def run():
    with ProcessPoolExecutor(max_workers=4) as pool:
        obs = pool.submit(observable_job)
        states = dict(zip([(s, e) for s in STATES for e in (0.0, EPS)],
                          pool.map(state_job, [(s, e) for s in STATES for e in (0.0, EPS)])))
        return obs.result(), states


def analyse(obs, states):
    a, m, f, grad = obs.T
    FA = cumulative_trapezoid(f, U, initial=0.0)
    out = {"observable factors: exponent of a(u), m(u), f(u), ||grad c_u||_1 for u in [20, 800]":
           [slope(U, v, 20.0, 800.0) for v in (a, m, f, grad)]}
    curves, flags = {}, {"F >= |error at C| at all times, every state": True}
    for s in STATES:
        n0, n1 = states[(s, 0.0)][:, 0], states[(s, EPS)][:, 0]
        gamma, kappa = states[(s, EPS)][:, 1], states[(s, EPS)][:, 2]
        err = np.abs(n1 - n0)
        F = np.array([0.0] + [convolve(a, gamma, t) + convolve(m, kappa, t) for t in U[1:]])
        curves[s] = {"err": err, "F": F, "gamma": gamma, "kappa": kappa}
        ok = bool(np.all(F >= err - 1e-10))
        flags["F >= |error at C| at all times, every state"] &= ok
        kmid, kend = int(np.argmin(np.abs(U - 0.1 * T_MAX))), len(U) - 1
        out[s] = {"max_t |error at C|": float(err.max()), "at t": float(U[np.argmax(err)]),
                  "F(T/10), F(T)": [float(F[kmid]), float(F[kend])],
                  "Theorem-A bound (T/10), (T)": [float(FA[kmid]), float(FA[kend])],
                  "sup_t F / sup_t |error|": float(F.max() / max(err.max(), 1e-300)),
                  "Theorem A(T) / sup_t |error|": float(FA[kend] / max(err.max(), 1e-300)),
                  "Theorem A at the time of the peak error / peak error": float(
                      FA[np.argmax(err)] / max(err.max(), 1e-300)),
                  "state-factor exponents gamma, kappa for s in [20, 800]": [slope(U, gamma, 20.0, 800.0),
                                                                             slope(U, kappa, 20.0, 800.0)]}
    ratios = [out[s]["F(T/10), F(T)"][1] / out[s]["F(T/10), F(T)"][0] if out[s]["F(T/10), F(T)"][0] > 0
              else float("nan") for s in STATES]
    flags[f"F saturates: F(T)/F(T/10) < 1.3 for every state (T = {T_MAX:g})"] = bool(np.nanmax(ratios) < 1.3)
    kmid = int(np.argmin(np.abs(U - 0.1 * T_MAX)))
    flags["Theorem-A bound grows: A(T)/A(T/10) > 2.5"] = bool(FA[-1] / FA[kmid] > 2.5)
    flags["observable factors decay as u^(-1/2) (a, m exponents in [-0.6, -0.4])"] = bool(
        all(-0.6 <= x <= -0.4 for x in out["observable factors: exponent of a(u), m(u), f(u), ||grad c_u||_1 for u in [20, 800]"][:2]))
    waves = [out[f"wave {l}"]["sup_t F / sup_t |error|"] for l in (20, 40, 80, 160)]
    out["wave lambda -> sup F / sup |error|"] = waves
    flags["sup F / sup |error| for the waves varies by < 2x across lambda (lambda-independent tightness)"] = bool(
        max(waves) / min(waves) < 2.0)
    flags["sup F within 10x of the peak true error for every state"] = bool(
        max(out[s]["sup_t F / sup_t |error|"] for s in STATES) < 10.0)
    out["F(T)/F(T/10) per state"] = dict(zip(STATES, ratios))
    out["acceptance"] = flags
    return out, curves, FA


def plot(obs, curves, FA, out):
    plt = style()
    a, m, f, grad = obs.T
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4))
    pos = U > 0
    for ax, names, title in ((axes[0, 0], ["wall 10", "wall 40"], "(a) domain walls"),
                             (axes[0, 1], ["wave 20", "wave 80", "wave 160"], "(b) density waves")):
        ax.loglog(U[pos], FA[pos], color=MUTED, lw=1.2, label="Theorem A/E (Heisenberg only)")
        for name, c in zip(names, (CAT[0], CAT[1], CAT[2])):
            ax.loglog(U[pos], curves[name]["F"][pos], color=c, lw=1.8, label=f"Theorem F, {name}")
            ax.loglog(U[pos], np.where(curves[name]["err"][pos] > 0, curves[name]["err"][pos], np.nan), color=c,
                      lw=1.0, ls="--", label=f"true |error|, {name}")
        ax.set_xlabel("t [1/J]")
        ax.set_ylabel("error of $n_C$")
        ax.set_title(title)
        ax.set_ylim(1e-6, 2.0)
        ax.legend(fontsize=6.5, loc="lower right")
    ax = axes[1, 0]
    ax.loglog(U[pos], a[pos], color=CAT[0], label="a(u): observable density pairing")
    ax.loglog(U[pos], m[pos], color=CAT[1], label="m(u): observable coherence pairing")
    for name, ls in (("wall 10", "-"), ("random", ":")):
        ax.loglog(U[pos], EPS * curves[name]["gamma"][pos], color=CAT[2], ls=ls, label=f"$\\varepsilon\\gamma$(s), {name}")
        ax.loglog(U[pos], EPS * curves[name]["kappa"][pos], color=ORDINAL[3], ls=ls, label=f"$\\varepsilon\\kappa$(s), {name}")
    uu = np.array([20.0, 800.0])
    ax.loglog(uu, 2e-3 * (uu / 20.0) ** -0.5, color=INK, lw=0.8, ls="--")
    ax.text(900, 2e-3 * 40 ** -0.5, "$u^{-1/2}$", fontsize=7)
    ax.set_xlabel("u or s [1/J]")
    ax.set_title("(c) both pictures forget diffusively")
    ax.legend(fontsize=6, ncol=2, loc="lower left")
    ax = axes[1, 1]
    names = STATES
    yA = [out[s]["Theorem A(T) / sup_t |error|"] for s in names]
    yAp = [out[s]["Theorem A at the time of the peak error / peak error"] for s in names]
    yF = [out[s]["sup_t F / sup_t |error|"] for s in names]
    yy = np.arange(len(names))
    ax.plot(yA, yy, "o", color=MUTED, label=f"Theorem A/E at t = {T_MAX:g}")
    ax.plot(yAp, yy, "o", mfc="white", color=MUTED, label="Theorem A/E at the peak time")
    ax.plot(yF, yy, "D", color=CAT[0], label="Theorem F, supremum over all t")
    ax.axvline(1.0, color=INK, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yticks(yy, names)
    ax.invert_yaxis()
    ax.set_xlabel("bound / sup$_t$ |true error at C|")
    ax.set_title(f"(d) tightness over {T_MAX:g}/J")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    axes[0, 0].text(0.02, 0.96, "Theorem F uses global state factors, so it is the same for both walls",
                    transform=axes[0, 0].transAxes, fontsize=7, color=MUTED, va="top")
    fig.suptitle("E13: forgetting in both pictures gives a time-uniform bound for conserved densities")
    fig.tight_layout()
    return save_figure(fig, "e13_two_sided_bound")


if __name__ == "__main__":
    if "--replot" not in sys.argv:
        obs, states = run()
        np.savez_compressed(RESULTS / "e13_two_sided_bound.npz", obs=obs,
                            **{f"state_{s}_{int(e > 0)}": v for (s, e), v in states.items()})
    z = np.load(RESULTS / "e13_two_sided_bound.npz")
    obs = z["obs"]
    states = {(s, e): z[f"state_{s}_{int(e > 0)}"] for s in STATES for e in (0.0, EPS)}
    out, curves, FA = analyse(obs, states)
    save_json(RESULTS / "e13_two_sided_bound.json", out)
    print(plot(obs, curves, FA, out))
    for k, v in out.items():
        print(f"{k}: {v}")
