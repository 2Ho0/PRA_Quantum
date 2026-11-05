"""E12b: the all-orders bound of Theorem G (item 4) for collective noise with memory, against the exact leakage of E12.

For noise that is a stationary Markov process (white noise, telegraph noise, the Ehrenfest/OU chain), the phase
accumulated after time u depends on the past only through the current noise state. Conditioning on it in the
D2 remainder of Theorem B3 gives the rigorous kernel
    K_Q(b) = max_k | 1^T exp(b (W - i Q diag(xi))) e_k |      (worst case over the current noise state),
which equals exp(-2 gamma_c Q^2 b) for white noise and 1 for quasi-static noise (no all-orders protection). The bound
for a purely charged dH and the leakage O = 1 - Pi_0 (osc = 1/2) is
    |Delta(t)| <= 4 ||dH|| osc(O) sum_{q != 0} ||dH_q|| int_0^t (t - b) K_q(b) db.
Exact leakage and parameters from results/e12_noise_memory.npz (N = 6, kappa = 1e-3).
"""

import numpy as np
from scipy.linalg import expm

from common import RESULTS, save_json
from e12_noise_memory import KAPPA, KINDS, N, T, T2S, noise_process, perturbation
from qerr.charge import charge_projector


def kernel(kind, T2, q, b):
    if kind == "white":
        return np.exp(-q * q * b / T2)
    if kind == "static":
        return np.ones_like(b)
    noise = noise_process(kind, T2)
    M = noise.W - 1j * q * np.diag(noise.xi)
    lam, V = np.linalg.eig(M)
    Vinv = np.linalg.inv(V)
    rows = (V.sum(axis=0)[None, :] * np.exp(np.outer(b, lam))) @ Vinv   # (len(b), K): 1^T e^{bM} e_k
    return np.abs(rows).max(axis=1)


def bound_curve(kind, T2, pert, t):
    h = perturbation(pert)
    qs = (1, -1) if pert == "Q1" else (2, -2)
    norm = KAPPA * np.linalg.norm(h, 2)
    out = np.zeros_like(t)
    b = np.linspace(0.0, t[-1], 4001)
    for q in qs:
        hq = KAPPA * np.linalg.norm(charge_projector(N, h, q), 2)
        K = kernel(kind, T2, q, b)
        cum0 = np.concatenate([[0.0], np.cumsum(0.5 * (K[1:] + K[:-1]) * np.diff(b))])            # int_0^s K
        cum1 = np.concatenate([[0.0], np.cumsum(0.5 * ((b * K)[1:] + (b * K)[:-1]) * np.diff(b))])  # int_0^s b K
        out += 4 * norm * 0.5 * hq * (np.interp(t, b, cum0) * t - np.interp(t, b, cum1))
    return out


if __name__ == "__main__":
    z = np.load(RESULTS / "e12_noise_memory.npz", allow_pickle=True)
    res = {eval(k): tuple(v) for k, v in zip(z["res_keys"], z["res_vals"])}
    out, ok = {}, True
    for kind in KINDS:
        for T2 in T2S:
            for pert in ("Q1", "Q2"):
                exact = res[(kind, T2, pert, KAPPA)][0]
                bnd = bound_curve(kind, T2, pert, T)
                holds = bool(np.all(bnd >= exact - 1e-14))
                ok &= holds
                out[f"{kind} T2={T2} {pert}"] = {"bound holds at all t": holds,
                                                 "bound / exact at t = 10": float(bnd[-1] / exact[-1])}
    tight = {k: [v["bound / exact at t = 10"] for key, v in out.items() if key.startswith(k)] for k in KINDS}
    summary = {"all-orders bound >= exact leakage for every process, T2 and charge": ok,
               "bound/exact at t = 10, range per process": {k: [float(min(v)), float(max(v))] for k, v in tight.items()}}
    save_json(RESULTS / "e12b_all_orders.json", {"summary": summary, "cases": out})
    for k, v in summary.items():
        print(f"{k}: {v}")
