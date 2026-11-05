"""E12: collective noise with memory (Theorem G of docs/theory/phase4-theory.md).

Reference (N = 6): the E8 model without collective dephasing (long-range XY, weak random z fields, local sigma^z
dephasing 0.05), Neel state. The collective noise xi(t) S^z is one of four processes with the same single-charge
coherence time T2 (chi_1(T2) = 1/e):
  white      Lindblad collective dephasing gamma_c = 1/(2 T2)
  static     quasi-static Gaussian field, sigma = sqrt(2)/T2 (uniform grid, revival time beyond t = 10)
  OU         Ehrenfest chain (24 telegraphs), correlation time T2, Gaussian-like
  telegraph  two-state noise switching at rate 1/T2 (non-Gaussian)
The noise average is exact (Kubo's stochastic Liouville equation, qerr.noise). Imperfections of strength kappa:
  Q1 = kappa sum_j X_j (charge +-1),  Q2 = kappa sum_j (X_j X_j+1 - Y_j Y_j+1)/2 (+-2),  C0 = kappa sum_j Z_j Z_j+1 (0).
Checks:
  (a) item 2: the C0 error of the staggered magnetization is the same for every process (and for no noise);
  (b) item 3: the leakage P(M != 0) equals the second-order kernel formula sum_Q int int chi_Q(s-u) K_Q(t,s,u) for
      every process, where the kernel K_Q of the noise-free reference is computed once;
  (c) protection times: the leakage ratio quasi-static / white tends to (sqrt(pi)/2)|Q| when the noise is strong;
  (d) kappa doubling multiplies the leakage by 4 for every process.
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

from common import CAT, INK, MUTED, ORDINAL, RESULTS, base_model, neel, save_figure, save_json, style
from qerr import spins
from qerr.charge import charge_projector
from qerr.lindblad import OpenModel, evolve
from qerr.noise import averaged_evolve, ehrenfest, quasi_static, telegraph

N, GAMMA_LOC, KAPPA = 6, 0.05, 1e-3
T2S = [0.5, 1.0, 2.0, 4.0]
KINDS = ["white", "static", "OU", "telegraph"]
T = np.linspace(0.0, 10.0, 41)
H_K = 0.025                       # kernel grid step
TARGETS = [2.5, 5.0, 10.0]
TOL = {"rtol": 1e-10, "atol": 1e-13}
M_DIAG = spins.total_sz(N)
STAG = (spins.zvals(N) * (-1.0) ** np.arange(N)).sum(axis=1) / N


def perturbation(kind):
    if kind == "Q1":
        return sum(spins.pauli_op(N, j, "x") for j in range(N))
    if kind == "Q2":
        return sum(0.5 * (spins.pauli_op(N, j, "x") @ spins.pauli_op(N, j + 1, "x")
                          - spins.pauli_op(N, j, "y") @ spins.pauli_op(N, j + 1, "y")) for j in range(N - 1))
    return sum(spins.pauli_op(N, j, "z") @ spins.pauli_op(N, j + 1, "z") for j in range(N - 1))


@lru_cache(maxsize=None)
def noise_process(kind, T2):
    """The process of the given kind with chi_1(T2) = 1/e (None for white noise, which is a Lindblad term)."""
    if kind == "white":
        return None
    if kind == "static":
        sigma = np.sqrt(2.0) / T2
        return quasi_static(sigma, n_nodes=int(max(41, np.ceil(60 * sigma * T[-1] / 10.0))), width=6.0)
    make = (lambda x: ehrenfest(x / T2, T2, 24)) if kind == "OU" else (lambda x: telegraph(x / T2, 1.0 / T2))
    x = brentq(lambda x: make(x).chi(1, T2).real - np.exp(-1.0), 1e-3, 4.0 if kind == "OU" else 2.0)
    return make(x)


def chi(kind, T2, q, b, analytic_static=False):
    """chi_q(b) of the simulated process; analytic_static replaces the static grid (which revives) by the Gaussian."""
    if kind == "white":
        return np.exp(-q**2 * b / T2)  # 2 gamma_c q^2 b with gamma_c = 1/(2 T2)
    if kind == "static" and analytic_static:
        return np.exp(-(q * b / T2) ** 2)  # sigma^2 = 2 / T2^2
    return noise_process(kind, T2).chi(q, b).real


def reference():
    return base_model(N, 0.0, GAMMA_LOC, 0.0)


def run_case(args):
    """(kind, T2, pert, kappa) -> (leakage(t), staggered magnetization(t)) of the noise-averaged evolution."""
    kind, T2, pert, kappa = args
    ref = reference()
    H = ref.H + kappa * perturbation(pert) if pert else ref.H
    rho0 = neel(N)
    if kind == "none" or kind == "white":
        extra = np.ones((N, N)) / (2.0 * T2) if kind == "white" else 0.0
        rhos = evolve(OpenModel(H=H, Gamma=ref.Gamma + extra, emission=ref.emission), rho0, T, **TOL)
    else:
        rhos = averaged_evolve(OpenModel(H=H, Gamma=ref.Gamma, emission=ref.emission), rho0, T,
                               noise_process(kind, T2), **TOL)
    diag = np.real(np.einsum("tii->ti", rhos))
    return diag[:, ~np.isclose(M_DIAG, 0.0)].sum(axis=1), diag @ STAG


def kernel_row(args):
    """K_Q(t_target, s, u_i) for all s = u_i + b_j <= t_target, for each target."""
    pert, q, i = args
    ref = reference()
    h = perturbation(pert)
    grid = np.arange(0.0, T[-1] + 1e-9, H_K)
    rho_u = evolve(ref, neel(N), [0.0, grid[i]], **TOL)[-1] if i else neel(N)
    hq, hmq = charge_projector(N, h, q), charge_projector(N, h, -q)
    X = -1j * (hq @ rho_u - rho_u @ hq)
    Ys = evolve(ref, X, grid[: len(grid) - i], **TOL)
    O = np.diag((~np.isclose(M_DIAG, 0.0)).astype(complex))
    Oa = evolve(ref, O, grid, heisenberg=True, **TOL)
    G = 1j * (hmq @ Oa - Oa @ hmq)          # V_{-q}^dag(O_a)
    out = {}
    for tt in TARGETS:
        K = int(round(tt / H_K))
        if i > K:
            continue
        j = np.arange(K - i + 1)
        out[tt] = np.real(np.einsum("jab,jba->j", G[K - i - j], Ys[j]))
    return out


def triangle_weights(K):
    """Trapezoid weights on {(i, j): i + j <= K} for int_{u + b <= t} f du db (h = 1)."""
    i, j = np.meshgrid(np.arange(K + 1), np.arange(K + 1), indexing="ij")
    w = np.where(i + j <= K, 1.0, 0.0) * np.where((i == 0) | (j == 0) | (i + j == K), 0.5, 1.0)
    w[0, 0] = w[K, 0] = w[0, K] = 1.0 / 6.0
    return w


def second_order(kernel, kind, T2, pert):
    """sum_q int int chi_q(b) K_q(t, u + b, u) du db at each target time."""
    qs = (1, -1) if pert == "Q1" else (2, -2)
    out = {}
    for tt in TARGETS:
        K = int(round(tt / H_K))
        total = 0.0
        w = triangle_weights(K)
        b = np.arange(K + 1) * H_K
        for q in qs:
            F = np.zeros((K + 1, K + 1))
            for i in range(K + 1):
                F[i, : K - i + 1] = kernel[(pert, q, i)][tt]
            total += np.sum(w * F * chi(kind, T2, q, b)[None, :]) * H_K**2
        out[tt] = float(total)
    return out


def run():
    cases = [("none", 1.0, "C0", KAPPA)] + [(k, T2, p, kap) for k in KINDS for T2 in T2S
                                           for p, kap in (("Q1", KAPPA), ("Q1", 2 * KAPPA), ("Q2", KAPPA),
                                                          ("Q2", 2 * KAPPA), ("C0", KAPPA))]
    cases += [(k, T2, None, 0.0) for k in KINDS for T2 in T2S] + [("none", 1.0, None, 0.0)]
    rows = [(p, q, i) for p, qs in (("Q1", (1, -1)), ("Q2", (2, -2))) for q in qs
            for i in range(int(round(TARGETS[-1] / H_K)) + 1)]
    with ProcessPoolExecutor(max_workers=16) as pool:
        res = dict(zip(cases, pool.map(run_case, cases)))
        kern = dict(zip(rows, pool.map(kernel_row, rows, chunksize=8)))
    return res, kern


def analyse(res, kern):
    it = {tt: int(np.argmin(np.abs(T - tt))) for tt in TARGETS}
    out, flags = {}, {}
    # (a) covariant imperfection: the same error for every process
    base_none = res[("none", 1.0, None, 0.0)][1]
    d_none = res[("none", 1.0, "C0", KAPPA)][1] - base_none
    dev = max(np.abs((res[(k, T2, "C0", KAPPA)][1] - res[(k, T2, None, 0.0)][1]) - d_none).max()
              for k in KINDS for T2 in T2S)
    out["C0: max |error(process) - error(no noise)|"] = float(dev)
    out["C0: max |error|"] = float(np.abs(d_none).max())
    flags["covariant error independent of the noise process (< 1e-10)"] = bool(dev < 1e-10)
    # (b) second-order kernel formula, (d) kappa doubling
    rel, ratios, table = [], [], {}
    for k in KINDS:
        for T2 in T2S:
            for p in ("Q1", "Q2"):
                pred = second_order(kern, k, T2, p)
                for tt in TARGETS:
                    exact = res[(k, T2, p, KAPPA)][0][it[tt]] / KAPPA**2
                    rel.append(abs(exact - pred[tt]) / abs(exact))
                    ratios.append(res[(k, T2, p, 2 * KAPPA)][0][it[tt]] / res[(k, T2, p, KAPPA)][0][it[tt]])
                    table[f"{k} T2={T2} {p} t={tt}"] = {"exact/kappa^2": float(exact), "kernel": pred[tt]}
    out["kernel formula: max relative deviation"] = float(max(rel))
    out["kappa doubling ratios (min, max)"] = [float(min(ratios)), float(max(ratios))]
    flags["second-order kernel formula holds for every process (< 2%)"] = bool(max(rel) < 0.02)
    flags["kappa doubling ratio 4 (3.95-4.05)"] = bool(3.95 < min(ratios) and max(ratios) < 4.05)
    # (c) protection: leakage ratio to white noise at t = 10
    prot = {}
    for k in KINDS[1:]:
        for p, q in (("Q1", 1), ("Q2", 2)):
            prot[f"{k}/{p}"] = [float(res[(k, T2, p, KAPPA)][0][it[10.0]] / res[("white", T2, p, KAPPA)][0][it[10.0]])
                                for T2 in T2S]
    asym = {}
    bb = np.linspace(0.0, 60.0, 24001)
    for k in KINDS[1:]:
        for q in (1, 2):
            asym[f"{k} q={q}"] = [float(np.trapezoid(chi(k, T2, q, bb * T2, analytic_static=True), bb * T2)
                                        / (T2 / q**2)) for T2 in T2S]
    out["leakage ratio to white noise at t = 10, per T2"] = prot
    out["protection-time ratio tau_Q/tau_Q(white) (t -> inf)"] = asym
    out["static/white predicted (sqrt(pi)/2)|Q|"] = [float(np.sqrt(np.pi) / 2), float(np.sqrt(np.pi))]
    r1, r2 = prot["static/Q1"][0], prot["static/Q2"][0]
    flags["quasi-static noise protects |Q| = 2 worse than white noise (ratio > 1.3 at T2 = 0.5)"] = bool(r2 > 1.3)
    flags["static/white ratio grows with |Q| (R2/R1 > 1.5 at T2 = 0.5)"] = bool(r2 / r1 > 1.5)
    out["acceptance"] = flags
    return out, table


def plot(res, kern, out):
    plt = style()
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4))
    colors = {"white": ORDINAL[3], "static": CAT[1], "OU": CAT[2], "telegraph": ORDINAL[1]}
    for ax, p, title in ((axes[0, 0], "Q2", "(a) pair creation (|Q| = 2), $T_2$ = 1"),
                         (axes[0, 1], "Q1", "(b) transverse field (|Q| = 1), $T_2$ = 1")):
        for k in KINDS:
            ax.plot(T, res[(k, 1.0, p, KAPPA)][0] / KAPPA**2, color=colors[k], label=k)
            pred = second_order(kern, k, 1.0, p)
            ax.plot(TARGETS, [pred[tt] for tt in TARGETS], "o", mfc="white", mec=colors[k], ms=6)
        ax.set_xlabel("t")
        ax.set_ylabel("leakage $P(M\\neq0)/\\kappa^2$")
        ax.set_title(title)
        ax.legend(fontsize=7, title="noise (same $T_2$)", title_fontsize=7)
    axes[0, 0].text(0.02, 0.96, "lines: exact average; circles: second-order kernel formula",
                    transform=axes[0, 0].transAxes, fontsize=7, color=MUTED, va="top")
    ax = axes[1, 0]
    strength = 1.0 / np.array(T2S)
    for k, mk in (("static", "o"), ("OU", "s"), ("telegraph", "^")):
        for p, ls in (("Q1", "--"), ("Q2", "-")):
            ax.plot(strength, out["leakage ratio to white noise at t = 10, per T2"][f"{k}/{p}"], ls, marker=mk,
                    color=colors[k], label=f"{k}, {p}")
    for q, val in ((1, np.sqrt(np.pi) / 2), (2, np.sqrt(np.pi))):
        ax.axhline(val, color=INK, lw=0.8, ls=":")
        ax.text(strength.min(), val + 0.015, f"static limit $(\\sqrt{{\\pi}}/2)|Q|$, |Q| = {q}", fontsize=7,
                va="bottom", color=INK)
    ax.set_xscale("log")
    ax.set_xticks(sorted(strength), [f"{s:g}" for s in sorted(strength)])
    ax.minorticks_off()
    ax.set_xlabel("noise strength $1/T_2$")
    ax.set_ylabel("leakage ratio to white noise, t = 10")
    ax.set_title("(c) leakage relative to white noise of the same $T_2$ (solid |Q| = 2, dashed |Q| = 1)")
    ax.legend(fontsize=6.5, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    ax = axes[1, 1]
    base = res[("none", 1.0, None, 0.0)][1]
    ax.plot(T, (res[("none", 1.0, "C0", KAPPA)][1] - base) / KAPPA, color=INK, lw=3, alpha=0.3, label="no noise")
    for k in KINDS:
        d = res[(k, 0.5, "C0", KAPPA)][1] - res[(k, 0.5, None, 0.0)][1]
        ax.plot(T, d / KAPPA, color=colors[k], lw=1.2, label=f"{k}, $T_2$ = 0.5")
    ax.set_xlabel("t")
    ax.set_ylabel("error of $m_s$ / $\\kappa$")
    ax.set_title(f"(d) covariant error: identical for every process (max dev {out['C0: max |error(process) - error(no noise)|']:.0e})")
    ax.legend(fontsize=7)
    fig.suptitle("E12: Zeno protection survives noise memory, but its charge scaling changes")
    fig.tight_layout()
    return save_figure(fig, "e12_noise_memory")


if __name__ == "__main__":
    if "--replot" not in sys.argv:
        res, kern = run()
        np.savez_compressed(RESULTS / "e12_noise_memory.npz",
                            res_keys=np.array([repr(k) for k in res]),
                            res_vals=np.array([np.stack(v) for v in res.values()]),
                            kern_keys=np.array([repr((k, tt)) for k, v in kern.items() for tt in v]),
                            kern_vals=np.array([v[tt] for v in kern.values() for tt in v], dtype=object))
    z = np.load(RESULTS / "e12_noise_memory.npz", allow_pickle=True)
    res = {eval(k): tuple(v) for k, v in zip(z["res_keys"], z["res_vals"])}
    kern = {}
    for key, val in zip(z["kern_keys"], z["kern_vals"]):
        k, tt = eval(key)
        kern.setdefault(k, {})[tt] = val
    out, table = analyse(res, kern)
    save_json(RESULTS / "e12_noise_memory.json", {"results": out, "kernel check": table})
    print(plot(res, kern, out))
    for k, v in out.items():
        print(f"{k}: {v}")
