"""Quadratic fermion operators as matrices, with exact commutators, norms and Heisenberg evolution.

Two representations of the same operator:
  BdG form       Q(M) = sum_ab M_ab Psi_a Psi_b^dag,  Psi = (c_1..c_N, c_1^dag..c_N^dag)
  Majorana form  q(A) = (i/4) sum_cd A_cd w_c w_d,     c_j = (w_j + i w_{N+j}) / 2,  A antisymmetric
They agree up to a multiple of the identity: Q(M) = q(A) + Tr(M)/2. In Majorana form
[q(A1), q(A2)] = i q([A1, A2]), and for real A (Hermitian q) the norm is ||q(A)|| = ||A||_1 / 4.
Under number dephasing (jumps sqrt(g_i) n_i and sqrt(g_c) N) quadratic operators stay quadratic:
e^{tL^dag} Q(M0) = Q(M(t)) with dM/dt = -i[H_BdG^T, M] + Lambda * M, the transpose of the
covariance equation in qerr.fermions.
"""

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

from qerr.fermions import bdg_matrix, damping_matrix


def _omega(N: int) -> np.ndarray:
    """Psi = Omega w."""
    I = np.eye(N)
    return 0.5 * np.block([[I, 1j * I], [I, -1j * I]])


def bdg_operator(h: np.ndarray, Delta: np.ndarray) -> np.ndarray:
    """M with Q(M) = sum h_ij c_i^dag c_j + 1/2 sum (Delta_ij c_i^dag c_j^dag + h.c.) exactly."""
    h, Delta = np.asarray(h, complex), np.asarray(Delta, complex)
    return np.block([[np.zeros_like(h), 0.5 * Delta.conj().T], [0.5 * Delta, h]])


def number_operator(N: int, i: int) -> np.ndarray:
    M = np.zeros((2 * N, 2 * N), complex)
    M[N + i, N + i] = 1.0
    return M


def bdg_to_majorana(M: np.ndarray) -> np.ndarray:
    W = _omega(M.shape[0] // 2)
    B = W.T @ M @ W.conj()
    return -2j * (B - B.T)


def majorana_to_bdg(A: np.ndarray) -> np.ndarray:
    """M with Q(M) = q(A) exactly (no identity part)."""
    W = _omega(A.shape[0] // 2)
    return 4.0 * W.conj() @ (0.25j * A) @ W.T


def quad_norm(A: np.ndarray) -> float:
    """||q(A)||: exact for real A (Hermitian) and imaginary A (anti-Hermitian), an upper bound otherwise."""
    A = np.asarray(A)
    return 0.25 * (np.linalg.norm(A.real, "nuc") + np.linalg.norm(A.imag, "nuc"))


def charge_part(A: np.ndarray, q: int) -> np.ndarray:
    """Component of q(A) that changes the particle number by q in {0, 2, -2}."""
    M = majorana_to_bdg(A)
    N = M.shape[0] // 2
    keep = np.zeros_like(M)
    blocks = {0: [(slice(0, N), slice(0, N)), (slice(N, None), slice(N, None))],
              2: [(slice(N, None), slice(0, N))],      # c^dag c^dag
              -2: [(slice(0, N), slice(N, None))]}     # c c
    for rows, cols in blocks[q]:
        keep[rows, cols] = M[rows, cols]
    return bdg_to_majorana(keep)


def heisenberg_evolve(h, Delta, gamma_loc, gamma_c, A0, times, rtol=1e-10, atol=1e-12) -> np.ndarray:
    """A(t) with e^{tL^dag} q(A0) = q(A(t)) (the dephasing generator is unital, so no identity part)."""
    times = np.asarray(times, float)
    HB = bdg_matrix(np.asarray(h, complex), np.asarray(Delta, complex))
    Ht, H = sp.csr_matrix(HB.T), sp.csr_matrix(HB)
    Lam = damping_matrix(gamma_loc, gamma_c)
    M0 = majorana_to_bdg(np.asarray(A0, complex))
    D = M0.shape[0]

    def rhs(t, y):
        M = y.reshape(D, D)
        return (-1j * (Ht @ M - (H @ M.T).T) + Lam * M).ravel()

    if times[-1] == 0.0:
        Ms = np.repeat(M0[None], len(times), axis=0)
    else:
        sol = solve_ivp(rhs, (0.0, times[-1]), M0.ravel(), t_eval=times, method="DOP853", rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        Ms = sol.y.T.reshape(len(times), D, D)
    return np.array([bdg_to_majorana(M) for M in Ms])
