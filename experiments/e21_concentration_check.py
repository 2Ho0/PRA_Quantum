"""E21: numerical check of the rigorous form of Theorem A3 (concentration under uniform forgetting).

System: N = 4 spins, H = nearest-neighbour XY chain (J = 1, open) with weak random z fields, local sigma^z dephasing
0.5 and emission 0.2 on every site, so L is primitive (unique fixed point). Ensemble: delta H(g) = sum_m g_m h_m over
all 21 one- and two-local Paulis h_m (unit norm), g ~ N(0, sigma^2 1). Observable Z_0, initial Neel state.
Checks (docs/theory/error-anatomy-theory.md, Theorem A3 and Lemma A3.1):
  (1) Lemma A3.1 in the Hilbert-Schmidt norm on the traceless subspace: with s(u) = ||e^{uL}|_0||_{2->2} <= K e^{-u/tau},
      every sample with ||delta H(g)|| <= 1/(4 K tau) has s'(u) <= K e^{-u/(2 tau)};
  (2) the concentration bound: with T_eff(t) = int_0^t min(1, sqrt(D) K e^{-u/(2 tau)}) du and
      delta' >= P(||delta H|| > 1/(4 K tau)) from the matrix Gaussian tail bound 2 D exp(-r^2 / (2 v^2)),
      P(|f - E f| > sigma osc(O) (sum ||h_m||^2)^1/2 T_eff(t) sqrt(2 ln(2/delta)) + osc(O) delta') <= delta + delta';
  (3) the gain over the bound without dissipation (T_eff = t).
Exact superoperators (256 x 256), 4000 samples, t = 1, 5, 20, 60.
Fixed before the run: (1) holds with ratio <= 1 + 1e-9 at every sampled u; (2) the empirical exceedance frequency is at
most delta + delta' + 3 binomial standard errors for delta = 0.1, 0.05, 0.01 at every t.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
from scipy.linalg import expm  # noqa: E402
from scipy.sparse.linalg import expm_multiply  # noqa: E402

from common import RESULTS, save_json  # noqa: E402
from qerr import spins  # noqa: E402
from qerr.lindblad import OpenModel, superop  # noqa: E402

N, D = 4, 16
T_LIST = (1.0, 5.0, 20.0, 60.0)
DELTAS = (0.1, 0.05, 0.01)
N_SAMPLES, SEED, N_LEMMA = 4000, 2047, 100


def paulis():
    """Normalised Pauli strings (vectorised, column stacking) except the identity: an orthonormal traceless basis."""
    single = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
              "Z": np.diag([1.0, -1.0])}
    import itertools
    out = []
    for letters in itertools.product("IXYZ", repeat=N):
        if set(letters) == {"I"}:
            continue
        P = np.array([[1.0]])
        for a in letters:
            P = np.kron(P, single[a])
        out.append(P.reshape(-1, order="F") / np.sqrt(D))
    return np.array(out).T  # (D^2, D^2 - 1)


def ensemble_terms():
    terms = [spins.pauli_op(N, i, a) for i in range(N) for a in "xyz"]
    terms += [spins.pauli_op(N, i, a) @ spins.pauli_op(N, i + 1, b) for i in range(N - 1) for a, b in ("xx", "yy", "zz")]
    return terms


def commutator_superop(h):
    I = np.eye(D)
    return -1j * (np.kron(I, h) - np.kron(h.T, I))


def run():
    rng = np.random.default_rng(SEED)
    H = sum(0.5 * (spins.pauli_op(N, i, "x") @ spins.pauli_op(N, i + 1, "x")
                   + spins.pauli_op(N, i, "y") @ spins.pauli_op(N, i + 1, "y")) for i in range(N - 1))
    H = H + sum(0.1 * rng.normal() * spins.pauli_op(N, i, "z") for i in range(N))
    L = superop(OpenModel(H=H, Gamma=0.5 * np.eye(N), emission=0.2 * np.ones(N)))
    B = paulis()
    us = np.linspace(0.0, 60.0, 241)
    s = np.array([np.linalg.norm(B.conj().T @ expm(u * L) @ B, 2) for u in us])
    gap = -np.max(np.linalg.eigvals(B.conj().T @ L @ B).real)
    tau = 1.02 / gap  # slightly slower than the slowest mode, so that K is finite
    K = float(np.max(s * np.exp(us / tau)))
    r = 1.0 / (4.0 * K * tau)
    terms = ensemble_terms()
    M = len(terms)
    v2_unit = float(np.linalg.norm(sum(h @ h for h in terms), 2))  # ||sum h_m^2|| for sigma = 1
    delta_p = 0.01
    sigma = r / np.sqrt(2.0 * v2_unit * np.log(2 * D / delta_p))  # tail bound 2 D exp(-r^2/(2 sigma^2 v2)) = delta'
    V = [commutator_superop(h) for h in terms]
    O = spins.pauli_op(N, 0, "z")
    rho0 = spins.basis_state(N, "0101").reshape(-1, order="F")
    o = O.conj().reshape(-1, order="F")
    g_all = sigma * rng.normal(size=(N_SAMPLES, M))
    norms = np.array([np.linalg.norm(sum(gm * h for gm, h in zip(g, terms)), 2) for g in g_all])
    f = np.zeros((N_SAMPLES, len(T_LIST)))
    lemma_worst = 0.0
    t_idx = [int(round(t)) for t in T_LIST]  # T_LIST holds integers; one Krylov sweep over t = 0, 1, ..., 60
    checked = 0
    for n, g in enumerate(g_all):
        Lg = L + sum(gm * Vm for gm, Vm in zip(g, V))
        traj = expm_multiply(Lg, rho0.astype(complex), start=0.0, stop=T_LIST[-1], num=int(T_LIST[-1]) + 1,
                             endpoint=True)
        f[n] = (traj[t_idx] @ o).real
        if norms[n] <= r and checked < N_LEMMA:  # the lemma, on the first N_LEMMA samples inside S
            checked += 1
            for u in us[::16]:
                sp_u = np.linalg.norm(B.conj().T @ expm(u * Lg) @ B, 2)
                lemma_worst = max(lemma_worst, sp_u / (K * np.exp(-u / (2 * tau))))
    return dict(f=f, norms=norms, K=K, tau=tau, r=r, sigma=sigma, delta_p=delta_p, M=M, gap=gap,
                lemma_worst=lemma_worst, s=s, us=us)


def analyse(r):
    osc, M = 2.0, r["M"]
    fac = r["sigma"] * osc * np.sqrt(M)  # sum ||h_m||^2 = M
    teff = []
    for t in T_LIST:
        u = np.linspace(0.0, t, 4001)
        teff.append(float(np.trapezoid(np.minimum(1.0, np.sqrt(D) * r["K"] * np.exp(-u / (2 * r["tau"]))), u)))
    f = r["f"]
    dev = np.abs(f - f.mean(axis=0))
    out = {"spectral gap of L on the traceless subspace": float(r["gap"]), "K, tau": [r["K"], r["tau"]],
           "radius 1/(4 K tau) and sigma": [r["r"], r["sigma"]],
           "empirical P(||delta H|| > radius) and the bound delta'": [float((r["norms"] > r["r"]).mean()), r["delta_p"]],
           "Lemma A3.1: max s'(u) / (K e^{-u/(2 tau)}) over samples inside S": r["lemma_worst"]}
    rows, ok = {}, r["lemma_worst"] <= 1 + 1e-9
    n = f.shape[0]
    for j, t in enumerate(T_LIST):
        row = {"T_eff": teff[j]}
        for d in DELTAS:
            bound = fac * teff[j] * np.sqrt(2 * np.log(2 / d)) + osc * r["delta_p"]
            exceed = float((dev[:, j] > bound).mean())
            allowed = d + r["delta_p"]
            ok &= exceed <= allowed + 3 * np.sqrt(allowed * (1 - allowed) / n)
            row[f"delta = {d:g}: bound, empirical (1-delta) quantile, exceedance, no-dissipation bound"] = [
                float(bound), float(np.quantile(dev[:, j], 1 - d)), exceed,
                float(fac * t * np.sqrt(2 * np.log(2 / d)))]
        rows[f"t = {t:g}"] = row
    out["concentration"] = rows
    out["acceptance: lemma and concentration hold"] = bool(ok)
    return out


if __name__ == "__main__":
    import sys  # noqa: F401
    res = run()
    np.savez_compressed(RESULTS / "e21_concentration_check.npz", **{k: v for k, v in res.items()})
    out = analyse(res)
    save_json(RESULTS / "e21_concentration_check.json", out)
    for key, v in out.items():
        print(f"{key}: {v}")
