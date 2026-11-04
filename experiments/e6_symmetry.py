"""E6b: symmetry diagnostics of the BEQS 51-ion Neel data and the error budget of the learned 10-ion model.

(a, b) Model-free: the 51-ion Neel quench (z basis, 1000 shots per time, 0-10 ms). Under U(1)-covariant dynamics
with time-independent detection errors the distribution of the total magnetization M is stationary, while
charge-0 observables such as the staggered magnetization evolve. Drifts of <M> and Var M measure charge-changing
processes (spontaneous decay of the metastable level, U(1)-breaking couplings).
(c) Model-based: a Neel quench on 10 ions in the generator learned by E6a. Reference = its U(1)-symmetric part with
the structured dephasing matrix gamma_c 11^T + diag(g); imperfection = the learned U(1)-breaking couplings
(charge +-1 fields, charge +-2 anisotropies). Compared: the actual error of the staggered magnetization, the
commutator-form certificate (Thm B3'), Thm A, the global B3 bound and the BEQS bias bound 2t||k||_1||O||.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp

from common import CAT, MUTED, ORDINAL, RESULTS, ROOT, save_figure, save_json, style, triangle_integral
from e6_ion_learning import N, PAIRS, RAD_S, ansatz, symmetrized
from qerr.bounds import _phi
from qerr.iondata import load_neel51
from qerr.learning import gamma_matrix
from qerr.lindblad import OpenModel, evolve
from qerr.pauli import psparse
from qerr.spins import basis_state

NEEL51 = ROOT / "data" / "raw" / "NeelDataZ51" / "DataHL51IonmeasType14Base3.mat"
TAU_D52_MS = 1168.0     # 40Ca+ D5/2 lifetime, Kreuter et al., PRA 71, 032504 (2005)
T_GRID = np.linspace(0.0, 10.0, 21)       # ms
AB_GRID = np.linspace(0.0, 10.0, 11)      # ms, grid of the B3' certificate
N_BOOT = 1000


# ---------- (a, b) 51-ion Neel data ----------
def neel_moments():
    times, s = load_neel51(NEEL51)
    n = s.shape[2]
    M = s.sum(axis=2) / 2.0                                      # (T, shots)
    stag = -(s * (-1.0) ** np.arange(n)).mean(axis=2)          # +1 for the ideal Neel state
    twoM = np.rint(2 * M[0]).astype(int)
    sector = np.bincount(twoM - twoM.min()).argmax() + twoM.min()   # most frequent 2M at t = 0
    rng = np.random.default_rng(7)
    idx = rng.integers(0, M.shape[1], size=(N_BOOT, M.shape[1]))
    stats = {}
    for name, f in (("M", lambda x: x.mean(axis=-1)), ("varM", lambda x: x.var(axis=-1, ddof=1)),
                    ("stag", lambda x: x.mean(axis=-1)), ("p_sector", lambda x: (np.rint(2 * x) == sector).mean(axis=-1))):
        data = stag if name == "stag" else M
        est = f(data)
        boot = np.array([f(data[:, i]) for i in idx])          # (B, T)
        stats[name] = (est, boot.std(axis=0, ddof=1))
    return times, stats, n


def slope(t, y, se):
    w = 1.0 / se**2
    W = np.sum(w)
    tb = np.sum(w * t) / W
    b = np.sum(w * (t - tb) * (y - np.sum(w * y) / W)) / np.sum(w * (t - tb) ** 2)
    return float(b), float(np.sqrt(1.0 / np.sum(w * (t - tb) ** 2)))


# ---------- (c) error budget in the learned 10-ion model ----------
def pieces():
    ans = ansatz()
    theta = np.load(RESULTS / "e6_ion_learning.npz")["theta"]
    th_sym = symmetrized(theta, ans)
    mats = [psparse(P, N) for P in ans.h_terms]
    H = lambda th: sum(c * m for c, m in zip(th, mats)).tocsr()
    k = {p: m for m, p in enumerate(ans.h_terms)}
    single = [k[((i, a),)] for i in range(N) for a in "XY"]
    th_ch = theta - th_sym
    th_ch1 = np.zeros_like(theta)
    th_ch1[single] = th_ch[single]
    G = gamma_matrix(theta, ans)
    gc = float(G[np.triu_indices(N, 1)].mean())
    g = np.clip(np.diag(G) - gc, 0.0, None)
    em = np.clip(theta[-N:], 0.0, None)
    return dict(H_sym=H(th_sym), H_ch=H(th_ch), H_ch1=H(th_ch1), H_ch2=H(th_ch - th_ch1), gc=gc, g=g, em=em,
                k1=float(np.abs(th_ch[: len(ans.h_terms)]).sum()))


def models(p):
    ref = OpenModel(H=p["H_sym"], Gamma=p["gc"] * np.ones((N, N)) + np.diag(p["g"]), emission=p["em"])
    full = OpenModel(H=(p["H_sym"] + p["H_ch"]).tocsr(), Gamma=ref.Gamma, emission=p["em"])
    bare = OpenModel(H=p["H_sym"], Gamma=np.diag(p["g"]), emission=p["em"])   # L_0: no collective part
    return ref, full, bare


def stag_op():
    return sp.diags(-sum((-1.0) ** i * psparse(((i, "Z"),), N).diagonal() for i in range(N)) / N).tocsr()


def opnorm(X):
    X = X.toarray() if sp.issparse(X) else X
    return float(np.abs(np.linalg.eigvalsh(0.5 * (X + X.conj().T))).max()
                 + np.abs(np.linalg.eigvalsh(0.5j * (X.conj().T - X))).max()) if not np.allclose(X, X.conj().T) \
        else float(np.abs(np.linalg.eigvalsh(X)).max())


def trajectory(model, X, grid, heisenberg, f):
    """[f(X(t)) for t in grid], evolving interval by interval so that only one operator is held in memory."""
    out = [f(X)]
    for a, b in zip(grid[:-1], grid[1:]):
        X = evolve(model, X, [0.0, b - a], heisenberg=heisenberg, rtol=1e-7, atol=1e-9)[-1]
        out.append(f(X))
    return out, X


def job(args):
    kind, payload = args
    p = pieces()
    ref, full, bare = models(p)
    O = stag_op().toarray()
    Hc = p["H_ch"].toarray()
    comm_norm = lambda X: opnorm(Hc @ X - X @ Hc)
    if kind == "actual":
        rho0 = basis_state(N, "10" * (N // 2))     # density matrix of the Neel state, qubit 1 down
        diag = O.diagonal()
        expect = lambda r: float(np.real(np.sum(diag * np.diag(r))))
        return [trajectory(m, rho0, T_GRID, False, expect)[0] for m in (ref, full)]
    if kind == "thmA":
        return trajectory(ref, O, T_GRID, True, comm_norm)[0]
    if kind == "cert":                                       # one a-row of the B3' integrand for both charges
        ia = payload
        _, Oa = trajectory(ref, O, AB_GRID[: ia + 1], True, lambda X: None)
        rows = []
        for key in ("H_ch1", "H_ch2"):
            Hp = p[key].toarray()
            rows.append(trajectory(bare, 1j * (Hp @ Oa - Oa @ Hp), AB_GRID[: len(AB_GRID) - ia], True, comm_norm)[0])
        return rows


def error_budget():
    jobs = [("actual", None), ("thmA", None)] + [("cert", ia) for ia in range(len(AB_GRID))]
    for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[var] = "3"  # inherited by the spawned workers: 8 x 3 threads, ~3 GB per worker
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1, 8)) as pool:
        res = list(pool.map(job, jobs))
    p = pieces()
    actual = np.abs(np.array(res[0][1]) - np.array(res[0][0]))
    integrand = np.array(res[1])
    thmA = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(T_GRID))])
    n = len(AB_GRID)
    cert = np.zeros(n)
    dt = AB_GRID[1] - AB_GRID[0]
    for q, key in ((1, 0), (2, 1)):
        f = np.full((n, n), np.nan)
        for ia in range(n):
            f[ia, : n - ia] = res[2 + ia][key]
        cert += triangle_integral(f * np.exp(-2 * p["gc"] * q**2 * AB_GRID)[None, :], dt)
    nH, n1, n2 = (opnorm(p[k]) for k in ("H_ch", "H_ch1", "H_ch2"))
    glob = np.array([4 * nH * (n1 * _phi(2 * p["gc"], t) + n2 * _phi(8 * p["gc"], t)) for t in T_GRID])
    beqs = 2 * T_GRID * p["k1"]
    return dict(actual=actual, thmA=thmA, cert=cert, glob=glob, beqs=beqs, gc=p["gc"], norms=(nH, n1, n2))


def first_order_budget():
    """Total bootstrap variance of the Hamiltonian parameters in the symmetric vs breaking combinations."""
    ans = ansatz()
    d = np.load(RESULTS / "e6_ion_learning.npz")
    boot = d["boot"][:, : len(ans.h_terms)]
    k = {p: m for m, p in enumerate(ans.h_terms)}
    var = lambda v: float(np.var(v, ddof=1))
    sym = brk = 0.0
    for i, j in PAIRS:
        xx, yy, xy, yx = (boot[:, k[((i, a), (j, b))]] for a, b in ("XX", "YY", "XY", "YX"))
        sym += var((xx + yy) / np.sqrt(2)) + var((xy - yx) / np.sqrt(2))
        brk += var((xx - yy) / np.sqrt(2)) + var((xy + yx) / np.sqrt(2))
    for i in range(N):
        sym += var(boot[:, k[((i, "Z"),)]])
        brk += var(boot[:, k[((i, "X"),)]]) + var(boot[:, k[((i, "Y"),)]])
    return sym, brk


def plot(times, st, eb):
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5))
    ax = axes[0]
    for key, col, lab in (("stag", CAT[0], "staggered magnetization (charge-0 observable)"),
                          ("p_sector", CAT[1], "P(initial magnetization sector) / P at t = 0")):
        y, e = st[key]
        y0 = y[0] if key == "p_sector" else 1.0
        ax.errorbar(times, y / y0, yerr=e / y0, fmt="o-", ms=3.5, lw=1.2, color=col, label=lab)
    ax.set_ylim(-0.5, 1.2)
    ax.set_xlabel("time  [ms]")
    ax.set_title("51-ion Néel quench (BEQS data)", fontsize=9)
    ax.legend(fontsize=6.8, loc="upper right")
    ax = axes[1]
    for key, col, lab in (("M", CAT[0], "⟨M⟩(t) − ⟨M⟩(0)"), ("varM", CAT[2], "Var M(t) − Var M(0)")):
        y, e = st[key]
        ax.errorbar(times, y - y[0], yerr=e, fmt="o", ms=3.5, lw=1.0, color=col, label=lab)
    decay = 25.5 * times / TAU_D52_MS
    ax.plot(times, -decay, color=MUTED, ls="--", lw=1.0, label="D₅/₂ decay alone (τ = 1.17 s): drift of ⟨M⟩")
    ax.plot(times, decay, color=MUTED, ls=":", lw=1.2, label="D₅/₂ decay alone: Poisson growth of Var M")
    ax.set_xlabel("time  [ms]")
    ax.set_title("moments of the total magnetization M", fontsize=9)
    ax.legend(fontsize=6.8, loc="upper left")
    ax = axes[2]
    for key, col, lab, ls in (("beqs", CAT[1], "BEQS bias bound 2t‖k‖₁‖O‖", "-"),
                              ("glob", ORDINAL[1], "global Zeno bound (Thm B3)", "-"),
                              ("thmA", ORDINAL[2], "forgetting bound (Thm A)", "-"),
                              ("cert", ORDINAL[3], "commutator-form certificate (Thm B3′)", "-"),
                              ("actual", CAT[2], "actual error |Δ m_s|", "--")):
        t = AB_GRID if key == "cert" else T_GRID
        y = eb[key]
        ax.semilogy(t[1:], np.where(y[1:] > 0, y[1:], np.nan), color=col, ls=ls, label=lab)
    ax.axhline(2.0, color=MUTED, lw=0.8, ls=":", label="trivial bound 2 osc(O)")
    ax.set_xlabel("time  [ms]")
    ax.set_title("learned 10-ion model, Néel quench: effect of U(1)-breaking terms", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("E6b  U(1) structure of the BEQS experiments", fontsize=10, y=1.03)
    fig.tight_layout()
    return save_figure(fig, "e6_symmetry")


CACHE = RESULTS / "e6_symmetry.npz"


def save_cache(times, st, eb):
    arrays = {f"st_{k}_{i}": np.asarray(v[i]) for k, v in st.items() for i in (0, 1)}
    arrays.update({f"eb_{k}": np.asarray(v) for k, v in eb.items()})
    np.savez_compressed(CACHE, times=times, **arrays)


def load_cache():
    z = np.load(CACHE)
    st = {k: (z[f"st_{k}_0"], z[f"st_{k}_1"]) for k in ("M", "varM", "stag", "p_sector")}
    eb = {k[3:]: z[k] for k in z.files if k.startswith("eb_")}
    eb["gc"], eb["norms"] = float(eb["gc"]), tuple(eb["norms"])
    return z["times"], st, eb


if __name__ == "__main__":
    import sys
    if "--replot" in sys.argv and CACHE.exists():   # figure only, from the cached arrays of the last full run
        print(plot(*load_cache()))
        sys.exit(0)
    times, st, n51 = neel_moments()
    eb = error_budget()
    save_cache(times, st, eb)
    sym, brk = first_order_budget()
    acc = {"51 ions: staggered magnetization at 0, 5, 10 ms": [float(st["stag"][0][i]) for i in (0, 5, 10)],
           "51 ions: P(initial sector) at 0 and 10 ms": [float(st["p_sector"][0][0]), float(st["p_sector"][0][-1])]}
    for key in ("M", "varM"):
        b, se = slope(times, st[key][0], st[key][1])
        acc[f"51 ions: slope of {key} per ms (+- 1 se)"] = [b, se]
    acc["expected |d<M>/dt| from D5/2 decay alone, 25.5 / tau (per ms)"] = 25.5 / TAU_D52_MS
    acc["10 ions: collective rate gamma_c used (rad/s)"] = eb["gc"] * RAD_S
    acc["10 ions: ||dH_ch||, ||dH_+-1||, ||dH_+-2|| (rad/s)"] = [x * RAD_S for x in eb["norms"]]
    for key in ("actual", "cert", "thmA", "glob", "beqs"):
        t = AB_GRID if key == "cert" else T_GRID
        acc[f"10 ions, t = 10 ms: {key}"] = float(eb[key][-1])
        acc[f"10 ions, t = 2 ms: {key}"] = float(eb[key][np.searchsorted(t, 2.0)])
    acc["certificate >= actual at common grid times"] = bool(np.all(eb["cert"][1:] >= eb["actual"][::2][1:]))
    acc["first-order variance share of symmetric combinations (Thm B1')"] = sym / (sym + brk)
    save_json(RESULTS / "e6_symmetry.json", {"acceptance": acc})
    print(plot(times, st, eb))
    for k, v in acc.items():
        print(f"{k}: {v}")
