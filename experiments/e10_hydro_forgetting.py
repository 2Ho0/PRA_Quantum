"""E10: Heisenberg forgetting under charge conservation follows the multipole order of the observable.

Free fermions (the XX chain), N = 400, nearest-neighbour hopping -1, local number dephasing g = 1, no collective
dephasing: the particle number is strongly conserved and densities diffuse. Imperfection: a uniform hopping
miscalibration dH = eps sum_k (c_k^dag c_{k+1} + h.c.), eps = 0.01 (charge 0, conserves N).
Theorem E predicts the forgetting integrand f_O(u) = ||[dH, e^{u L^dag}(O)]|| ~ u^{-(m+1)/2} for an observable whose
density moments vanish below order m: monopole n_c (m = 0) -> u^{-1/2}, so the Theorem-A bound grows as sqrt(t);
dipole n_c - n_{c+1} (m = 1) -> u^{-1} (log t); quadrupole (m = 2) -> u^{-3/2} (bounded). The bond current relaxes into a
density dipole (u^{-1}). True errors: density waves n_k = 1/2 + 0.4 cos(2 pi (k - c) / lambda), observed at the crest;
their peak error is set by the relative change of the diffusion constant and does not grow with lambda, while the
Theorem-A bound at the peak time does.
All operators are quadratic: commutators, norms and evolutions are exact single-particle computations.
"""

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.integrate import cumulative_trapezoid

from common import CAT, MUTED, ORDINAL, RESULTS, save_figure, save_json, style
from qerr.fermions import evolve_covariance, occupation, product_covariance
from qerr.quadratic import bdg_operator, bdg_to_majorana, heisenberg_evolve, majorana_to_bdg, number_operator, quad_norm

N, G_DEPH, EPS = 400, 1.0, 0.01
C = N // 2
U = np.concatenate([[0.0], np.geomspace(0.02, 1500.0, 70)])
LAMBDAS = [20, 40, 80, 160]
EDGE, EDGE_TOL, U_FIT_MIN = 20, 1e-3, 10.0


def hopping(N, amp):
    h = np.zeros((N, N), complex)
    for i in range(N - 1):
        h[i, i + 1] = h[i + 1, i] = amp
    return h


def observables():
    zero = np.zeros((N, N))
    n = lambda i: number_operator(N, i)
    jc = np.zeros((N, N), complex)
    jc[C, C + 1], jc[C + 1, C] = 1j, -1j
    return {"monopole n_c": n(C), "dipole n_c - n_c+1": n(C) - n(C + 1),
            "quadrupole n_c-1 - 2n_c + n_c+1": n(C - 1) - 2 * n(C) + n(C + 1),
            "bond current j_c": bdg_operator(jc, zero)}


def integrand_job(name):
    M0 = observables()[name]
    A_O = bdg_to_majorana(M0).real
    A_D = bdg_to_majorana(bdg_operator(hopping(N, EPS), np.zeros((N, N)))).real
    A_u = heisenberg_evolve(hopping(N, -1.0), np.zeros((N, N)), G_DEPH * np.ones(N), 0.0, A_O, U, rtol=1e-8,
                            atol=1e-11)
    f = np.array([quad_norm(A_D @ A.real - A.real @ A_D) for A in A_u])
    edge = []
    for A in A_u:
        dens = np.abs(np.diag(majorana_to_bdg(A)[N:, N:]))
        edge.append(dens[:EDGE].sum() + dens[-EDGE:].sum())
    return f, np.array(edge)


def wave_job(args):
    lam, eps = args
    occ = 0.5 + 0.4 * np.cos(2 * np.pi * (np.arange(N) - C) / lam)
    t_peak = lam**2 / (4 * np.pi**2 * 2.0 / G_DEPH)  # 1/(D k^2) with D = 2 J^2 / g
    times = np.linspace(0.0, 4.0 * t_peak, 161)
    Gs = evolve_covariance(hopping(N, -1.0 + eps), np.zeros((N, N)), G_DEPH * np.ones(N), 0.0,
                           product_covariance(occ), times, rtol=1e-9, atol=1e-12)
    return times, np.array([occupation(G)[C] for G in Gs])


def run():
    names = list(observables())
    waves = [(lam, e) for lam in LAMBDAS for e in (0.0, EPS)]
    with ProcessPoolExecutor(max_workers=len(names) + len(waves)) as pool:
        fi = pool.map(integrand_job, names)
        wv = pool.map(wave_job, waves)
        integrands = dict(zip(names, fi))
        traj = dict(zip(waves, wv))
    return integrands, traj


def slope_fit(f, edge):
    ok = (U >= U_FIT_MIN) & (edge < EDGE_TOL) & (f > 0)
    p = np.polyfit(np.log(U[ok]), np.log(f[ok]), 1)
    return float(p[0]), float(U[ok].min()), float(U[ok].max())


def bound(f):
    """Theorem A: F(t) = int_0^t f(u) du on the integrand grid (trapezoid, exact at the grid points)."""
    return cumulative_trapezoid(f, U, initial=0.0)


def analyse(integrands, traj):
    acc = {}
    slopes = {}
    for name, (f, edge) in integrands.items():
        slopes[name] = slope_fit(f, edge)
        acc[f"{name}: fitted exponent of f(u), fit window [u_min, u_max]"] = slopes[name]
    acc["kernel edge weight < 1e-3 in fit window"] = True  # enforced by the fit mask
    acc["monopole exponent in [-0.6, -0.4]"] = bool(-0.6 <= slopes["monopole n_c"][0] <= -0.4)
    acc["dipole exponent in [-1.15, -0.85]"] = bool(-1.15 <= slopes["dipole n_c - n_c+1"][0] <= -0.85)
    acc["quadrupole exponent <= -1.3"] = bool(slopes["quadrupole n_c-1 - 2n_c + n_c+1"][0] <= -1.3)
    F = {name: bound(f) for name, (f, _) in integrands.items()}
    for name in F:
        acc[f"{name}: Theorem-A bound at u = 10, 100, 1000"] = [float(np.interp(x, U, F[name])) for x in (10, 100, 1000)]
    peaks, ok = {}, True
    Fm = F["monopole n_c"]
    for lam in LAMBDAS:
        t, n0 = traj[(lam, 0.0)]
        _, n1 = traj[(lam, EPS)]
        err = np.abs(n1 - n0)
        k = int(np.argmax(err))
        peaks[lam] = (float(err[k]), float(t[k]), float(np.interp(t[k], U, Fm)))
        ok &= bool(np.all(np.interp(t, U, Fm) >= err - 1e-12))
    acc["density waves: lambda -> (peak |error|, peak time, Theorem-A bound at peak time)"] = {
        str(k): v for k, v in peaks.items()}
    acc["Theorem-A bound >= true error at all times (all lambda)"] = ok
    pk = [v[0] for v in peaks.values()]
    acc["peak true error spread across lambda (max/min) within 2"] = [float(max(pk) / min(pk)),
                                                                     bool(max(pk) / min(pk) <= 2.0)]
    acc["predicted peak ~ 0.4 * (2 eps) / e (diffusion-constant change)"] = float(0.4 * 2 * EPS / np.e)
    return acc, F, peaks


def plot(integrands, traj, F, peaks):
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    ax = axes[0]
    cols = [CAT[0], CAT[1], CAT[2], MUTED]
    for (name, (f, edge)), col in zip(integrands.items(), cols):
        m = U > 0
        ax.loglog(U[m], np.where(edge[m] < EDGE_TOL, f[m], np.nan), color=col, label=name)
        ax.loglog(U[m], np.where(edge[m] >= EDGE_TOL, f[m], np.nan), color=col, lw=0.6, alpha=0.4)
    for p, y0, lab in ((-0.5, 2e-3, r"$u^{-1/2}$"), (-1.0, 4e-4, r"$u^{-1}$"), (-1.5, 6e-5, r"$u^{-3/2}$")):
        uu = np.array([20.0, 400.0])
        ax.loglog(uu, y0 * (uu / 20.0) ** p, color=MUTED, lw=0.8, ls="--")
        ax.text(uu[-1] * 1.1, y0 * (uu[-1] / 20.0) ** p, lab, fontsize=7, color=MUTED, va="center")
    ax.set_xlabel("time  u  [1/J]")
    ax.set_ylabel(r"$\Vert[\delta H,\, e^{u\mathcal{L}^\dagger}(O)]\Vert$")
    ax.set_title("forgetting integrand by multipole order (N = 400)", fontsize=9)
    ax.legend(fontsize=6.8, loc="lower left")
    ax = axes[1]
    ramp = ORDINAL
    for lam, col in zip(LAMBDAS, ramp):
        t, n0 = traj[(lam, 0.0)]
        _, n1 = traj[(lam, EPS)]
        ax.loglog(t[1:], np.abs(n1 - n0)[1:], color=col, label=f"true error, λ = {lam}")
    m = U > 0
    ax.loglog(U[m], F["monopole n_c"][m], color=CAT[1], ls="--", label="Theorem-A bound (∝ √t)")
    ax.set_xlabel("time  t  [1/J]")
    ax.set_title("monopole: bound grows as √t, true error stays bounded", fontsize=9)
    ax.set_ylim(1e-5, None)
    ax.legend(fontsize=6.8, loc="lower right")
    fig.suptitle("E10  hydrodynamic forgetting under charge conservation (free fermions, dephasing g = 1)",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    return save_figure(fig, "e10_hydro_forgetting")


def save_cache(integrands, traj, path):
    flat = {f"f|{name}": f for name, (f, _) in integrands.items()}
    flat |= {f"edge|{name}": e for name, (_, e) in integrands.items()}
    for (lam, eps), (t, n) in traj.items():
        flat[f"t|{lam}|{eps}"], flat[f"n|{lam}|{eps}"] = t, n
    np.savez_compressed(path, **flat)


def load_cache(path):
    integrands, traj = {}, {}
    with np.load(path) as z:
        for name in observables():
            integrands[name] = (z[f"f|{name}"], z[f"edge|{name}"])
        for lam in LAMBDAS:
            for eps in (0.0, EPS):
                traj[(lam, eps)] = (z[f"t|{lam}|{eps}"], z[f"n|{lam}|{eps}"])
    return integrands, traj


if __name__ == "__main__":
    cache = RESULTS / "e10_hydro_forgetting.npz"
    if "--replot" not in sys.argv:  # --replot: analyse and plot the cached integrands and trajectories only
        save_cache(*run(), cache)
    integrands, traj = load_cache(cache)
    acc, F, peaks = analyse(integrands, traj)
    save_json(RESULTS / "e10_hydro_forgetting.json", {"acceptance": acc})
    print(plot(integrands, traj, F, peaks))
    for k, v in acc.items():
        print(f"{k}: {v}")
