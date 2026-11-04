"""E7: the commutator-form Zeno certificate (Thm B3') does not grow with system size; the global bound (B3) does.

Model of E5: nearest-neighbour hopping chain (the XX chain), Neel occupation, local dephasing 0.05, collective
dephasing gamma_c (jump sqrt(gamma_c) N), and the E5 pairing imperfection (purely charged, |q| = 2). The
observable is the centre occupation n_c (charge 0). In this fermionic convention the pairing sector decays at
rate 2 gamma_c, so
  certificate(t) = int_{a+b<=t} e^{-2 gamma_c b} || [dH, e^{b L0^dag}(i[dH, O_a])] || da db      (Thm B3')
  global(t)      = 4 ||dH||^2 osc(n_c) phi(2 gamma_c, t)                                       (B3, +-2 grouped)
L0 is the reference generator without collective dephasing, so one integrand table per N serves every gamma_c.
All operators are quadratic, so commutators and norms are exact (qerr.quadratic).
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import CAT, MUTED, ORDINAL, RESULTS, save_figure, save_json, style, triangle_integral
from e5_free_fermion import EPS, GAMMA_LOC, couplings, simulate
from qerr.bounds import _phi
from qerr.quadratic import bdg_operator, bdg_to_majorana, heisenberg_evolve, number_operator, quad_norm

NS = [20, 50, 100, 200]
GAMMA_CS = [0.0, 1.0, 4.0]
T_MAX, DT = 10.0, 0.25
OSC = 0.5  # osc(n_c)


def operators(N):
    h0, _ = couplings(N, "none", EPS)
    _, Delta = couplings(N, "pairing", EPS)
    A_O = bdg_to_majorana(number_operator(N, N // 2)).real
    A_D = bdg_to_majorana(bdg_operator(np.zeros((N, N)), Delta)).real
    return h0, A_O, A_D


def certificate_table(args):
    """f[i, j] = ||[dH, e^{b_j L0^dag}(i[dH, O_{a_i}])]|| on the grid triangle i + j <= n."""
    N, dt = args
    start = time.time()
    h0, A_O, A_D = operators(N)
    n = int(round(T_MAX / dt))
    grid = dt * np.arange(n + 1)
    g, zero = GAMMA_LOC * np.ones(N), np.zeros((N, N))
    O_a = heisenberg_evolve(h0, zero, g, 0.0, A_O, grid, rtol=1e-9, atol=1e-11)
    assert np.abs(O_a.imag).max() < 1e-8
    f = np.full((n + 1, n + 1), np.nan)
    for i in range(n + 1):
        Oa = O_a[i].real
        C = -(A_D @ Oa - Oa @ A_D)  # Majorana matrix of i[dH, O_a]
        Cb = heisenberg_evolve(h0, zero, g, 0.0, C, grid[: n + 1 - i], rtol=1e-9, atol=1e-11).real
        for j in range(n + 1 - i):
            f[i, j] = quad_norm(A_D @ Cb[j] - Cb[j] @ A_D)
    return f, quad_norm(A_D), time.time() - start


def certificate(f, gc, dt):
    b = dt * np.arange(f.shape[1])
    return triangle_integral(f * np.exp(-2.0 * gc * b)[None, :], dt)


def run():
    jobs = [(N, DT) for N in reversed(NS)] + [(50, DT / 2)]
    sims = [(N, gc, kind, EPS) for N in NS for gc in GAMMA_CS for kind in ("none", "pairing")]
    with ProcessPoolExecutor(max_workers=min(len(jobs) + len(sims), os.cpu_count() or 1)) as pool:
        tables = dict(zip(jobs, pool.map(certificate_table, jobs)))
        traj = dict(zip(sims, pool.map(simulate, sims)))
    return tables, traj


def analyse(tables, traj):
    t = DT * np.arange(int(round(T_MAX / DT)) + 1)
    out = {}
    for N in NS:
        f, norm_dH, secs = tables[(N, DT)]
        for gc in GAMMA_CS:
            obs = np.abs(traj[(N, gc, "pairing", EPS)][0] - traj[(N, gc, "none", EPS)][0])
            out[(N, gc)] = {"cert": certificate(f, gc, DT), "obs": obs,
                            "global": 4 * norm_dH**2 * OSC * np.array([_phi(2 * gc, s) for s in t]),
                            "norm_dH": norm_dH, "seconds": secs}
    acc = {}
    acc["certificate >= observed at every grid time (all N, gamma_c)"] = bool(all(
        np.all(r["cert"][1:] >= r["obs"][1:]) for r in out.values()))
    end = {(N, gc): out[(N, gc)]["cert"][-1] for N in NS for gc in GAMMA_CS}
    spread = max(abs(end[(200, gc)] - end[(100, gc)]) / end[(100, gc)] for gc in GAMMA_CS)
    acc["certificate(t=10) rel. spread N=100 vs 200 (max over gamma_c)"] = float(spread)
    acc["certificate N-independent within 5%"] = bool(spread < 0.05)
    ratio = out[(200, 1.0)]["global"][-1] / out[(50, 1.0)]["global"][-1]
    acc["global B3 ratio N=200 / N=50 (expect ~16)"] = float(ratio)
    acc["global ratio within 16 +- 25%"] = bool(12.0 <= ratio <= 20.0)
    c100 = [end[(100, gc)] for gc in GAMMA_CS]
    acc["certificate(t=10), N=100, gamma_c = 0, 1, 4"] = [float(c) for c in c100]
    acc["certificate decreases with gamma_c"] = bool(np.all(np.diff(c100) < 0))
    fine, _, _ = tables[(50, DT / 2)]
    conv = abs(certificate(fine, 1.0, DT / 2)[-1] - end[(50, 1.0)]) / end[(50, 1.0)]
    acc["grid convergence |cert(dt/2) - cert(dt)| / cert, N=50, gamma_c=1, t=10"] = float(conv)
    acc["certificate grid convergence < 2%"] = bool(conv < 0.02)
    for gc in GAMMA_CS:
        r = out[(100, gc)]
        acc[f"N=100, gamma_c={gc:g}: certificate / observed at t=10"] = float(r["cert"][-1] / r["obs"][-1])
        acc[f"N=100, gamma_c={gc:g}: global / certificate at t=10"] = float(r["global"][-1] / r["cert"][-1])
    acc["integrand table seconds per N"] = {N: round(tables[(N, DT)][2], 1) for N in NS}
    return t, out, acc


def plot(t, out):
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    gc = 1.0
    for key, col, mk, lab in (("global", CAT[1], "s", "global bound (Thm B3)"),
                              ("cert", CAT[0], "o", "commutator-form certificate (Thm B3′)"),
                              ("obs", CAT[2], "^", "observed |Δn_c|")):
        ax.loglog(NS, [out[(N, gc)][key][-1] for N in NS], marker=mk, color=col, ms=5, label=lab)
    ax.set_xticks(NS)
    ax.set_xticklabels([str(N) for N in NS])
    ax.minorticks_off()
    ax.set_xlabel("system size N")
    ax.set_title(f"error and bounds at t = {T_MAX:g}, γ_c = {gc:g}", fontsize=9)
    ax.legend(fontsize=7, loc="center right")
    ax.text(0.04, 0.93, "global bound ∝ N²; certificate and error flat", transform=ax.transAxes,
            color=MUTED, fontsize=7.5, va="top")
    ax = axes[1]
    ramp = [ORDINAL[0], ORDINAL[2], ORDINAL[3]]
    for g, col in zip(GAMMA_CS, ramp):
        r = out[(100, g)]
        ax.semilogy(t[1:], r["cert"][1:], color=col, label=f"γ_c = {g:g}")
        ax.semilogy(t[1:], np.where(r["obs"][1:] > 0, r["obs"][1:], np.nan), color=col, ls="--")
    ax.set_xlabel("time  t  [1/J]")
    ax.set_title("N = 100: certificate (solid) above observed error (dashed)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("E7  local Zeno bound: free fermions with number dephasing, pairing imperfection", fontsize=10, y=1.03)
    fig.tight_layout()
    return save_figure(fig, "e7_local_zeno")


if __name__ == "__main__":
    t, out, acc = analyse(*run())
    save_json(RESULTS / "e7_local_zeno.json", {"acceptance": acc})
    print(plot(t, out))
    for k, v in acc.items():
        print(f"{k}: {v}")
