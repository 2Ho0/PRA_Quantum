"""E5: Zeno protection and selection rules at large N (free fermions with number dephasing).

Nearest-neighbour hopping chain (= XX spin chain), half-filled Neel occupation, local dephasing
0.05 plus collective dephasing gamma_c (jump sqrt(gamma_c) N). Imperfections are local random
terms with the same pattern around the chain centre for every N:
  * pairing  Delta_{i,i+1}  (charge +-2, like XX-YY anisotropy)  -> Zeno-suppressed (Thm B3)
  * on-site  mu_i n_i       (charge 0, like z-field errors)      -> blind to gamma_c
Observables at the centre: occupation n_c (charge 0) and pairing <c_c c_{c+1}> (charge -2).
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import CAT, MUTED, ORDINAL, RESULTS, save_json, style
from qerr.fermions import evolve_covariance, occupation, pairing, product_covariance

NS = [20, 50, 100, 200]
GAMMA_CS = [0.0, 1.0, 4.0]
GAMMA_LOC, EPS = 0.05, 0.05
T = np.linspace(0.0, 10.0, 41)
PATTERN = np.random.default_rng(42).normal(size=(2, 1000))   # centred random pattern, row 0 pairing, row 1 on-site


def couplings(N, kind, eps):
    c = N // 2
    h = np.zeros((N, N), complex)
    for i in range(N - 1):
        h[i, i + 1] = h[i + 1, i] = -1.0
    Delta = np.zeros((N, N), complex)
    if kind == "pairing":
        for i in range(N - 1):
            Delta[i, i + 1] = eps * PATTERN[0, 500 + i - c]
            Delta[i + 1, i] = -Delta[i, i + 1]
    elif kind == "on-site":
        h = h + np.diag(eps * PATTERN[1, 500 + np.arange(N) - c])
    return h, Delta


def simulate(args):
    N, gc, kind, eps = args
    h, Delta = couplings(N, kind, eps)
    occ = np.arange(N) % 2
    Gs = evolve_covariance(h, Delta, GAMMA_LOC * np.ones(N), gc, product_covariance(occ), T, rtol=1e-9, atol=1e-11)
    c = N // 2
    return np.array([occupation(G)[c] for G in Gs]), np.array([pairing(G, c, c + 1) for G in Gs])


def run():
    jobs = [(N, gc, kind, EPS) for N in NS for gc in GAMMA_CS for kind in ("none", "pairing", "on-site")]
    jobs += [(50, gc, "pairing", 2 * EPS) for gc in GAMMA_CS]
    with ProcessPoolExecutor(max_workers=min(20, os.cpu_count() or 1)) as pool:
        return dict(zip(jobs, pool.map(simulate, jobs)))


def analyse(raw):
    res, acc = {}, {}
    for N in NS:
        for gc in GAMMA_CS:
            n0, _ = raw[(N, gc, "none", EPS)]
            for kind in ("pairing", "on-site"):
                n1, p1 = raw[(N, gc, kind, EPS)]
                res[(N, gc, kind)] = {"d_occ": np.abs(n1 - n0), "pair": np.abs(p1)}
    d10 = {gc: [float(res[(N, gc, "pairing")]["d_occ"][-1]) for N in NS] for gc in GAMMA_CS}
    big = [i for i, N in enumerate(NS) if N >= 50]
    acc["N-independence: |d n_c|(t=10), pairing, N>=50, max rel spread"] = float(
        max(np.ptp([d10[gc][i] for i in big]) / np.mean([d10[gc][i] for i in big]) for gc in GAMMA_CS))
    acc["N-independence within 5%"] = acc["N-independence: |d n_c|(t=10), pairing, N>=50, max rel spread"] < 0.05
    acc["|d n_c|(t=10), pairing, N=100, gamma_c = 0, 1, 4"] = [d10[gc][NS.index(100)] for gc in GAMMA_CS]
    acc["Zeno: decreases monotonically with gamma_c"] = bool(np.all(np.diff(acc["|d n_c|(t=10), pairing, N=100, gamma_c = 0, 1, 4"]) < 0))
    onsite = [res[(100, gc, "on-site")]["d_occ"] for gc in GAMMA_CS]
    acc["on-site (charge-0) error independent of gamma_c (max diff)"] = float(max(np.abs(o - onsite[0]).max() for o in onsite))
    for t_check in (2.0, 10.0):
        it = np.searchsorted(T, t_check)
        ratio = []
        for gc in GAMMA_CS:
            n0, _ = raw[(50, gc, "none", EPS)]
            big = np.abs(raw[(50, gc, "pairing", 2 * EPS)][0] - n0)[it]
            small = np.abs(raw[(50, gc, "pairing", EPS)][0] - n0)[it]
            ratio.append(float(big / small))
        acc[f"eps-doubling ratio of |d n_c|(t={t_check:g}), pairing, N=50, gamma_c=0,1,4 (leading order: 4)"] = ratio
    return res, acc


def plot(res):
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    pos = lambda y: np.where(y > 0, y, np.nan)
    ramp = [ORDINAL[0], ORDINAL[2], ORDINAL[3]]
    ax = axes[0]
    for gc, col in zip(GAMMA_CS, ramp):
        ax.semilogy(T, pos(res[(100, gc, "pairing")]["d_occ"]), color=col, label=f"pairing (|q|=2), γ_c = {gc}")
    ax.semilogy(T, pos(res[(100, 0.0, "on-site")]["d_occ"]), color=CAT[1], ls="--", label="on-site (q=0), any γ_c")
    ax.set_title("error in centre occupation, N = 100", fontsize=9)
    ax.set_xlabel("time  t  [1/J]")
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[1]
    for gc, col in zip(GAMMA_CS, ramp):
        ax.loglog(NS, [res[(N, gc, "pairing")]["d_occ"][-1] for N in NS], "o-", color=col, ms=5, label=f"γ_c = {gc}")
    ax.set_title("pairing imperfection: |Δn_c|(t=10) vs N", fontsize=9)
    ax.set_xlabel("system size N  (N=20: boundary reflections)")
    ax.set_xticks(NS)
    ax.set_xticklabels([str(N) for N in NS])
    ax.minorticks_off()
    ax.legend(fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.65))
    ax.text(0.04, 0.68, "flat in N: error is local,\nsuppressed by γ_c", transform=ax.transAxes, va="center",
            color=MUTED, fontsize=7.5)
    ax = axes[2]
    for gc, col in zip(GAMMA_CS, ramp):
        ax.semilogy(T, pos(res[(100, gc, "pairing")]["pair"]), color=col, label=f"γ_c = {gc}")
    ax.set_title("charged observable |⟨c_c c_c+1⟩| (ideal: 0), N = 100", fontsize=9)
    ax.set_xlabel("time  t  [1/J]")
    ax.set_ylim(top=0.2)
    ax.legend(fontsize=7, loc="upper right", ncol=3)
    fig.suptitle(f"E5  free fermions with number dephasing, local rate {GAMMA_LOC}, imperfection size {EPS}",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    path = RESULTS / "e5_free_fermion.png"
    fig.savefig(path, bbox_inches="tight")
    return path


if __name__ == "__main__":
    res, acc = analyse(run())
    save_json(RESULTS / "e5_free_fermion.json", {"acceptance": acc})
    print(plot(res))
    for k, v in acc.items():
        print(f"{k}: {v}")
