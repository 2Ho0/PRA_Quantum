from functools import reduce

import numpy as np

from qerr.fermions import evolve_covariance, hopping, occupation, pairing, product_covariance
from qerr.lindblad import OpenModel, evolve, expect
from qerr.spins import pauli_op


def jw_annihilators(N):
    """c_i = (prod_{k<i} Z_k) sigma^+_i, with occupied = bit 1 (sigma^- = |1><0| creates)."""
    cs = []
    for i in range(N):
        string = reduce(np.matmul, [pauli_op(N, k, "z") for k in range(i)], np.eye(2**N, dtype=complex))
        cs.append(string @ pauli_op(N, i, "+"))
    return cs


def test_jw_anticommutation():
    c = jw_annihilators(3)
    for i in range(3):
        for j in range(3):
            assert np.allclose(c[i] @ c[j].conj().T + c[j].conj().T @ c[i], np.eye(8) * (i == j))
            assert np.allclose(c[i] @ c[j] + c[j] @ c[i], 0)


def test_fermion_covariance_matches_manybody():
    N = 4
    rng = np.random.default_rng(7)
    A = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    h = 0.5 * (A + A.conj().T)
    B = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    Delta = 0.3 * (B - B.T)
    gamma_loc, gamma_c = 0.1 + 0.2 * rng.random(N), 0.25
    occ = np.array([0, 1, 1, 0])

    c = jw_annihilators(N)
    cd = [x.conj().T for x in c]
    H = sum(h[i, j] * cd[i] @ c[j] for i in range(N) for j in range(N))
    pair = sum(Delta[i, j] * cd[i] @ cd[j] for i in range(N) for j in range(N))
    H = H + 0.5 * (pair + pair.conj().T)
    n_ops = [cd[i] @ c[i] for i in range(N)]
    jumps = tuple(np.sqrt(g) * n for g, n in zip(gamma_loc, n_ops)) + (np.sqrt(gamma_c) * sum(n_ops),)
    model = OpenModel(H=H, Gamma=np.zeros((N, N)), emission=np.zeros(N), jumps=jumps)
    rho0 = np.diag([np.prod([(1 - occ[i]) if ((b >> (N - 1 - i)) & 1) == 0 else occ[i] for i in range(N)])
                    for b in range(2**N)]).astype(complex)

    times = np.array([0.0, 0.5, 1.5])
    rhos = evolve(model, rho0, times)
    Gs = evolve_covariance(h, Delta, gamma_loc, gamma_c, product_covariance(occ), times)
    for rho, G in zip(rhos, Gs):
        for i in range(N):
            for j in range(N):
                assert abs(np.trace(cd[i] @ c[j] @ rho) - hopping(G, i, j)) < 1e-8
                assert abs(np.trace(c[i] @ c[j] @ rho) - pairing(G, i, j)) < 1e-8
        assert np.allclose(occupation(G), [np.trace(n @ rho).real for n in n_ops], atol=1e-8)
