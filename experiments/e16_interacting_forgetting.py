"""E16: two-sided forgetting (Theorem F) beyond quasi-free dynamics, on a dephased XXZ chain (exact, N = 10).

Reference L: H = 1/2 sum_b (X_b X_{b+1} + Y_b Y_{b+1} + Delta Z_b Z_{b+1}) on an open chain (hopping amplitude 1, as in
E13) with number dephasing g = 4 (the coherence of a hop decays at rate g; D = 2/g = 0.5 at Delta = 0). Implemented L':
the hopping miscalibrated by eps = 0.01, dH = eps/2 sum_b (X X + Y Y), which conserves Q. Observable n_C, C = 4.
Delta = 1 is the interacting case (the isotropic point; the dephasing breaks SU(2) and integrability), Delta = 0 the
quasi-free control at the same size. Initial product states: the domain wall |1111100000>, Neel, the lowest open-chain
density mode n_x = 1/2 + 0.4 cos(pi (x + 1/2) / N), and random occupations.
Everything is exact (1024 x 1024 density matrices) on a uniform grid, dt = 0.125, t <= 40:
  the true error Delta(t);
  the Theorem A bound A(t) = int_0^t ||i[dH, O_u]|| du, which controls the integrand by the worst state;
  the integrand G(u, s) = <i[dH, O_u]>_{rho'_s} of Lemma F.0 and F_abs(t) = int_0^t |G(u, t - u)| du, the smallest
  bound that discards only the sign of the integrand (no factorisation, no hypotheses);
  the factors behind (H_O), ||grad c_u||_1 of the Z profile of O_u and the Theorem A integrand, and behind (H_rho), the
  largest density difference, spin current and bond coherence of rho'_s, with log-log slopes on [1, 10].
The diagonal integrals use Simpson's rule on t = 0, 0.25, 0.5, ... (an even number of steps; error O(dt^4)).
At Delta = 0 the nearest-neighbour bond coherence vanishes identically for diagonal initial states (sublattice
symmetry: Re C_xy lives at even and Im C_xy at odd distances), so there the current carries the off-diagonal part.
Finite size: the slowest mode decays at D (pi/N)^2 ~ 0.05 (Delta = 0), so every quantity saturates after t ~ 20 and the
diffusive window is about one decade.
Fixed before the run: Lemma F.0 reproduced to 1% of the peak error; the slopes of the (H_O) factors at Delta = 1 within
0.15 of those at Delta = 0; F_abs grows less than A between t = 2 and t = 10 for every state and both Delta.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import gc
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.integrate import simpson, solve_ivp
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from common import CAT, INK, MUTED, RESULTS, save_figure, save_json, style
from qerr import spins
from qerr.lindblad import OpenModel, adjoint_rhs, lindblad_rhs
from qerr.pauli import expectation_tables, expectations, psparse

N, G_DEPH, EPS, C = 10, 4.0, 0.01, 4
DT, T_MAX = 0.125, 40.0
TIMES = np.arange(0.0, T_MAX + DT / 2, DT)
DELTAS = (1.0, 0.0)
STATES = ("wall", "Neel", "lowest mode", "random")
FIT = (1.0, 10.0)
BONDS = range(N - 1)
STRINGS = ([((x, "Z"),) for x in range(N)] + [((b, a), (b + 1, c)) for a, c in ("XY", "YX", "XX", "YY") for b in BONDS])


def hamiltonian(hop, delta):
    terms = [(0.5 * hop, ((b, a), (b + 1, a))) for b in BONDS for a in "XY"]
    terms += [(0.5 * delta, ((b, "Z"), (b + 1, "Z"))) for b in BONDS] if delta else []
    return sum(c * psparse(p, N) for c, p in terms).tocsr()


def occupations(name):
    if name == "wall":
        return (np.arange(N) < N // 2).astype(float)
    if name == "Neel":
        return (np.arange(N) % 2 == 0).astype(float)
    if name == "lowest mode":
        return 0.5 + 0.4 * np.cos(np.pi * (np.arange(N) + 0.5) / N)
    return np.random.default_rng(7).uniform(0.0, 1.0, N)


def product_state(n):
    """Diagonal product state with <n_x> = n[x], n_x = (1 + Z_x)/2."""
    z = spins.zvals(N)
    return np.diag(np.prod((1.0 + z * (2.0 * n - 1.0)) / 2.0, axis=1)).astype(complex)


def spectral_norm(M, v0=None):
    try:
        w, v = eigsh(M, k=1, which="LM", tol=1e-9, ncv=24, v0=v0)
        return float(abs(w[0])), v[:, 0]
    except ArpackNoConvergence:
        return float(np.abs(np.linalg.eigvalsh(M)).max()), v0


def trajectory(model, X0, heisenberg=False, chunk=16):
    """(index, X(t)) at every grid time; the whole trajectory would not fit in memory."""
    rhs = adjoint_rhs if heisenberg else lindblad_rhs
    D = X0.shape[0]
    y = np.asarray(X0, complex).ravel()
    yield 0, y.reshape(D, D)
    for start in range(0, len(TIMES) - 1, chunk):
        ts = TIMES[start:start + chunk + 1]
        sol = solve_ivp(lambda t, v: rhs(model, v.reshape(D, D)).ravel(), (ts[0], ts[-1]), y, t_eval=ts[1:],
                        method="DOP853", rtol=1e-10, atol=1e-12)
        for m in range(len(ts) - 1):
            yield start + m + 1, sol.y[:, m].reshape(D, D)
        y = sol.y[:, -1].copy()
        del sol
        gc.collect()


def delta_job(delta):
    gamma = G_DEPH / 4.0 * np.eye(N)  # Gamma_ii (Z X Z - X): a hop's coherence decays at 4 Gamma_ii = g
    ref = OpenModel(H=hamiltonian(1.0, delta), Gamma=gamma, emission=np.zeros(N))
    imp = OpenModel(H=hamiltonian(1.0 + EPS, delta), Gamma=gamma, emission=np.zeros(N))
    dH = hamiltonian(EPS, 0.0)
    z = spins.zvals(N).astype(float)
    D, T = 2**N, len(TIMES)
    o_diag = (1.0 + z[:, C]) / 2.0
    rho0 = {s: product_state(occupations(s)) for s in STATES}
    K = np.empty((T, D * D), np.complex64)  # i[dH, O_u], flattened
    integrand_a, grad_c = np.zeros(T), np.zeros(T)
    reference = {s: np.zeros(T) for s in STATES}
    v0 = None
    for m, Ou in trajectory(ref, np.diag(o_diag).astype(complex), heisenberg=True):
        Ku = 1j * (dH @ Ou - Ou @ dH)
        Ku = 0.5 * (Ku + Ku.conj().T)
        integrand_a[m], v0 = spectral_norm(Ku, v0)
        K[m] = Ku.ravel()
        diag = np.real(np.diag(Ou))
        grad_c[m] = np.abs(np.diff(diag @ z / D)).sum()
        for s in STATES:
            reference[s][m] = diag @ np.real(np.diag(rho0[s]))  # <O>_L(t) = Tr[O_t rho_0]
    idx, ph = expectation_tables(STRINGS, N)
    out = {"integrand_a": integrand_a, "grad_c": grad_c}
    for s in STATES:
        G = np.zeros((T, T))  # G[u index, s index]
        implemented, factors = np.zeros(T), np.zeros((T, 3))
        for k, rho in trajectory(imp, rho0[s]):
            G[:, k] = np.real(K @ rho.conj().ravel().astype(np.complex64))
            implemented[k] = np.real(o_diag @ np.diag(rho))
            e = expectations(rho, idx, ph)
            dens, xy, yx, xx, yy = e[:N], *np.split(e[N:], 4)
            factors[k] = (0.5 * np.abs(np.diff(dens)).max(), 0.5 * np.abs(xy - yx).max(), 0.5 * np.abs(xx + yy).max())
        out[s] = {"G": G, "error": implemented - reference[s], "factors": factors}
    return out


def along_diagonal(G, j, absolute):
    """int_0^{t_j} G(u, t_j - u) du (Simpson) from the grid values G[i, j - i]; j even, so Simpson is exact to O(dt^4)."""
    if j == 0:
        return 0.0
    vals = G[np.arange(j + 1), j - np.arange(j + 1)]
    return float(simpson(np.abs(vals) if absolute else vals, dx=DT))


EVEN = np.arange(0, len(TIMES), 2)  # the diagonal integrals are evaluated on t = 0, 0.25, 0.5, ...


def slope(x, y, lo, hi):
    m = (x >= lo) & (x <= hi) & (y > 1e-14)
    return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0]) if m.sum() > 3 else None


def analyse(res):
    out, curves = {}, {}
    at = lambda t: int(round(t / (2 * DT)))  # index into the even grid
    for delta, r in res.items():
        key = f"Delta={delta:g}"
        A = np.array([simpson(r["integrand_a"][:j + 1], dx=DT) if j else 0.0 for j in EVEN])
        row = {"(H_O) slope of the Theorem A integrand on [1, 10]": slope(TIMES, r["integrand_a"], *FIT),
               "(H_O) slope of ||grad c_u||_1 on [1, 10]": slope(TIMES, r["grad_c"], *FIT),
               "A(10) / A(2)": float(A[at(10)] / A[at(2)]), "A(40) / A(10)": float(A[at(40)] / A[at(10)])}
        for s in STATES:
            G, err, fac = r[s]["G"], r[s]["error"][EVEN], r[s]["factors"]
            F = np.array([along_diagonal(G, j, True) for j in EVEN])
            ident = np.array([along_diagonal(G, j, False) for j in EVEN])
            peak = np.abs(err).max()
            curves[(key, s)] = (err, F, A, fac)
            row[s] = {
                "Lemma F.0: max |int G - error| / peak error": float(np.abs(ident - err).max() / peak),
                "F_abs >= |error| - 1% of peak at all t": bool(np.all(F >= np.abs(err) - 0.01 * peak)),
                "sup F_abs / peak error": float(F.max() / peak),
                "sup A / peak error (t <= 40)": float(A.max() / peak),
                "A / |error| at the peak time": float(A[np.argmax(np.abs(err))] / peak),
                "F_abs(10) / F_abs(2)": float(F[at(10)] / F[at(2)]), "F_abs(40) / F_abs(10)": float(F[at(40)] / F[at(10)]),
                "(H_rho) slopes on [1, 10]: gradient, current, coherence": [slope(TIMES, fac[:, i], *FIT) for i in range(3)],
            }
        out[key] = row
    a1, a0 = out["Delta=1"], out["Delta=0"]
    hk = ["(H_O) slope of the Theorem A integrand on [1, 10]", "(H_O) slope of ||grad c_u||_1 on [1, 10]"]
    out["acceptance"] = {
        "Lemma F.0 reproduced to 1% of the peak error": all(
            out[d][s]["Lemma F.0: max |int G - error| / peak error"] < 0.01 for d in ("Delta=1", "Delta=0") for s in STATES),
        "(H_O) slopes at Delta = 1 within 0.15 of Delta = 0": all(abs(a1[k] - a0[k]) < 0.15 for k in hk),
        "F_abs grows less than A between t = 2 and 10, every state and Delta": all(
            out[d][s]["F_abs(10) / F_abs(2)"] < out[d]["A(10) / A(2)"] for d in ("Delta=1", "Delta=0") for s in STATES),
    }
    return out, curves


def plot(res, curves, out):
    plt = style()
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.8))
    pos = TIMES > 0
    te = TIMES[EVEN][1:]
    styles = {"Delta=1": (CAT[0], "interacting, $\\Delta$ = 1"), "Delta=0": (CAT[1], "quasi-free, $\\Delta$ = 0")}
    ax = axes[0]
    for key, (c, lab) in styles.items():
        err, F, A, _ = (x[1:] if x.ndim == 1 else x for x in curves[(key, "wall")])
        ax.loglog(te, A, color=c, lw=1, ls=":")
        ax.loglog(te, F, color=c, lw=1.8, label=f"$F_{{abs}}$, {lab}")
        ax.loglog(te, np.where(np.abs(err) > 0, np.abs(err), np.nan), color=c, lw=1, ls="--")
    ax.set_ylim(1e-5, 1.0)
    ax.set_xlabel("t [1/J]")
    ax.set_ylabel("error of $n_C$ (domain wall)")
    ax.set_title("(a) dotted: Theorem A; solid: $F_{abs}$; dashed (hidden): true", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="upper left")
    ax = axes[1]
    for key, (c, lab) in styles.items():
        r = res[float(key.split("=")[1])]
        ax.loglog(TIMES[pos], r["integrand_a"][pos], color=c, lw=1.8, label=f"$\\|i[\\delta H, O_u]\\|$, {lab}")
        ax.loglog(TIMES[pos], r["grad_c"][pos], color=c, lw=1, ls="--", label=f"$\\|\\nabla c_u\\|_1$, {lab}")
    u = np.array(FIT)
    ax.loglog(u, 0.02 * u**-0.5, color=INK, lw=0.8, ls=":", label="$u^{-1/2}$")
    ax.axvspan(*FIT, color=MUTED, alpha=0.08, lw=0)
    ax.set_xlabel("u [1/J]")
    ax.set_title("(b) observable factors (H$_O$)", fontsize=9)
    ax.legend(fontsize=6, loc="lower left")
    ax = axes[2]
    for key, (c, lab) in styles.items():
        fac = curves[(key, "wall")][3]
        ax.loglog(TIMES[pos], fac[pos, 0], color=c, lw=1.8, label=f"largest gradient, {lab}")
        ax.loglog(TIMES[pos], fac[pos, 1], color=c, lw=1, ls="--", label=f"largest current, {lab}")
    ax.loglog(u, 0.3 * u**-0.5, color=INK, lw=0.8, ls=":", label="$s^{-1/2}$")
    ax.axvspan(*FIT, color=MUTED, alpha=0.08, lw=0)
    ax.set_xlabel("s [1/J]")
    ax.set_title("(c) state factors (H$_\\rho$), domain wall", fontsize=9)
    ax.legend(fontsize=6, loc="lower left")
    ax = axes[3]
    yy = np.arange(len(STATES))
    for key, (c, lab) in styles.items():
        off = -0.12 if key == "Delta=1" else 0.12
        ax.plot([out[key][s]["sup F_abs / peak error"] for s in STATES], yy + off, "D", color=c, label=f"$F_{{abs}}$, {lab}")
        ax.plot([out[key][s]["sup A / peak error (t <= 40)"] for s in STATES], yy + off, "D", mfc="white", mec=c,
                label=f"Theorem A, {lab}")
    ax.axvline(1.0, color=INK, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yticks(yy, STATES)
    ax.invert_yaxis()
    ax.set_xlabel("sup$_t$ bound / peak true error")
    ax.set_title("(d) tightness", fontsize=9)
    ax.legend(fontsize=6)
    fig.suptitle("E16: two-sided forgetting in a dephased XXZ chain (N = 10, exact)")
    fig.tight_layout()
    return save_figure(fig, "e16_interacting_forgetting")


if __name__ == "__main__":
    if "--replot" not in sys.argv:
        with ProcessPoolExecutor(max_workers=2) as pool:
            res = dict(zip(DELTAS, pool.map(delta_job, DELTAS)))
        np.savez_compressed(RESULTS / "e16_interacting_forgetting.npz", **{
            f"{d:g}|{k}": v for d, r in res.items() for k, v in r.items() if not isinstance(v, dict)}, **{
            f"{d:g}|{s}|{k}": v for d, r in res.items() for s in STATES for k, v in r[s].items()})
    zz = np.load(RESULTS / "e16_interacting_forgetting.npz")
    res = {d: {"integrand_a": zz[f"{d:g}|integrand_a"], "grad_c": zz[f"{d:g}|grad_c"],
               **{s: {k: zz[f"{d:g}|{s}|{k}"] for k in ("G", "error", "factors")} for s in STATES}} for d in DELTAS}
    out, curves = analyse(res)
    save_json(RESULTS / "e16_interacting_forgetting.json", out)
    print(plot(res, curves, out))
    for key, v in out.items():
        print(f"{key}: {v}")
