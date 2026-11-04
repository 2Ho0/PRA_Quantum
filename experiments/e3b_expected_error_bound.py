"""E3b: Theorem A2 (expected error of a zero-mean ensemble) against the E3 ensemble and BEQS Eq. (12).

Leading term of Theorem A2 for diagonal covariance sigma^2:
    L2(t) = sigma^2 sum_m I_mm(t),  I_mm(t) = int_{a+b<=t} || [h_m, e^{b L^dag}( i[h_m, O_a] )] || da db,
evaluated on a grid for the E3 setup (N = 6, O = Z2 Z3, 99 terms, sigma = 0.03). Requires E3's
results/e3_mean_shift.npz for the observed ensemble-mean shift.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import CAT, INK, ORDINAL, RESULTS, base_model, ops, save_json, style
from qerr.bounds import beqs_expected_bound
from qerr.lindblad import evolve
from qerr.models import charge_basis_terms

N, SIGMA = 6, 0.03
GAMMA_CS = [0.1, 1.0]
GAMMA_LOC, GAMMA_EM = 0.03, 0.005          # as in E3
DT, T_MAX = 0.1, 4.0
GRID = np.arange(0.0, T_MAX + DT / 2, DT)


def observable():
    o2, o3 = ops(N, 2), ops(N, 3)
    return o2["z"] @ o3["z"]


def f_table(args):
    """f[k, j] = || [h, e^{b_j L^dag}(i[h, O_{a_k}])] || on the triangle k + j <= K."""
    gc, m = args
    model = base_model(N, gc, GAMMA_LOC, GAMMA_EM)
    h = charge_basis_terms(N, max_range=2)[m].op
    Oa = evolve(model, observable(), GRID, heisenberg=True)
    K = len(GRID) - 1
    f = np.zeros((K + 1, K + 1))
    for k in range(K + 1):
        A = 1j * (h @ Oa[k] - Oa[k] @ h)
        Ab = evolve(model, A, GRID[: K - k + 1], heisenberg=True) if K - k > 0 else A[None]
        f[k, : K - k + 1] = [np.linalg.norm(h @ X - X @ h, 2) for X in Ab]
    return f


def triangle_integral(f):
    """int_{a+b<=t_K'} f da db for every K' (trapezoid weights on the grid triangle)."""
    K = f.shape[0] - 1
    out = np.zeros(K + 1)
    for Kp in range(1, K + 1):
        k, j = np.meshgrid(np.arange(Kp + 1), np.arange(Kp + 1), indexing="ij")
        inside = k + j <= Kp
        w = np.where(inside, 1.0, 0.0)
        w = w * np.where((k == 0) | (j == 0) | (k + j == Kp), 0.5, 1.0)
        w[0, 0], w[Kp, 0], w[0, Kp] = 1 / 6, 1 / 6, 1 / 6     # triangle corners
        out[Kp] = np.sum(w * f[: Kp + 1, : Kp + 1]) * DT**2
    return out


def run():
    terms = charge_basis_terms(N, max_range=2)
    charges = np.array([t.abs_charge for t in terms])
    jobs = [(gc, m) for gc in GAMMA_CS for m in range(len(terms))]
    with ProcessPoolExecutor(max_workers=min(20, os.cpu_count() or 1)) as pool:
        tables = dict(zip(jobs, pool.map(f_table, jobs)))
    res = {}
    for gc in GAMMA_CS:
        I = np.array([triangle_integral(tables[(gc, m)]) for m in range(len(terms))])
        res[gc] = {"A2_sym": SIGMA**2 * I[charges == 0].sum(axis=0),
                   "A2_ch": SIGMA**2 * I[charges > 0].sum(axis=0)}
    data = np.load(RESULTS / "e3_mean_shift.npz")
    for gc in GAMMA_CS:
        res[gc]["observed"] = np.interp(GRID, data["T"], data[f"s{SIGMA}_g{gc}_all"][0])
    return res, len(terms)


def acceptance(res, M):
    beqs = beqs_expected_bound(GRID, SIGMA, M)
    acc = {}
    for gc in GAMMA_CS:
        a2 = res[gc]["A2_sym"] + res[gc]["A2_ch"]
        early = (GRID > 0) & (GRID <= 1.0)
        acc[f"gc={gc}: A2 leading >= observed bias for 0 < t <= 1"] = bool(np.all(a2[early] >= res[gc]["observed"][early]))
        acc[f"gc={gc}: A2 leading < BEQS Eq.(12) for all t > 0"] = bool(np.all(a2[1:] < beqs[1:]))
        for t in (1.0, 4.0):
            i = np.searchsorted(GRID, t - 1e-9)
            acc[f"gc={gc}: BEQS/A2 at t={t:g}"] = float(beqs[i] / a2[i])
            acc[f"gc={gc}: A2 charged share at t={t:g}"] = float(res[gc]["A2_ch"][i] / a2[i])
    acc["Zeno in A2: charged part at t=4 smaller for gc=1 than gc=0.1"] = bool(
        res[1.0]["A2_ch"][-1] < res[0.1]["A2_ch"][-1])
    return acc


def plot(res, M):
    plt = style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    pos = lambda y: np.where(y > 0, y, np.nan)
    ax.semilogy(GRID, pos(beqs_expected_bound(GRID, SIGMA, M)), color=INK, ls=":", lw=1.4,
                label=f"BEQS Eq.(12), M = {M}")
    for gc, col in zip(GAMMA_CS, [ORDINAL[1], ORDINAL[3]]):
        ax.semilogy(GRID, pos(res[gc]["A2_sym"] + res[gc]["A2_ch"]), color=col, ls="--",
                    label=f"Thm A2 leading term, γ_c = {gc}")
        ax.semilogy(GRID, pos(res[gc]["observed"]), color=col, label=f"observed |E⟨O′⟩ − ⟨O⟩|, γ_c = {gc}")
    ax.semilogy(GRID, pos(res[0.1]["A2_ch"]), color=CAT[2], lw=1.2, ls="-.", label="A2 charged part, γ_c = 0.1")
    ax.semilogy(GRID, pos(res[1.0]["A2_ch"]), color=CAT[2], lw=1.2, ls=(0, (1, 1)), label="A2 charged part, γ_c = 1.0")
    ax.set_xlabel("time  t  [1/J0]")
    ax.set_ylabel("expected error in ⟨Z2 Z3⟩")
    ax.set_ylim(1e-6, 10)
    ax.legend(fontsize=6.8, loc="lower right", ncol=1)
    ax.set_title(f"E3b  Theorem A2 vs BEQS Eq.(12), N={N}, σ={SIGMA}, 99 terms", fontsize=9.5)
    fig.tight_layout()
    path = RESULTS / "e3b_expected_error_bound.png"
    fig.savefig(path, bbox_inches="tight")
    return path


if __name__ == "__main__":
    res, M = run()
    acc = acceptance(res, M)
    save_json(RESULTS / "e3b_expected_error_bound.json", {"acceptance": acc})
    print(plot(res, M))
    for k, v in acc.items():
        print(f"{k}: {v}")
