"""Second moments of quadratic fermions with number dephasing (local n_i and collective N).

Psi = (c_1..c_N, c_1^dag..c_N^dag), G_ab = <Psi_a Psi_b^dag>. The Hamiltonian
H = sum h_ij c_i^dag c_j + 1/2 sum (Delta_ij c_i^dag c_j^dag + h.c.) gives dG/dt = -i[H_BdG, G].
Dephasing by the Hermitian quadratic jumps sqrt(g_i) n_i and sqrt(g_c) N damps every bilinear
monomial at a fixed rate, so the second moments stay closed (the state need not be Gaussian):
G(dot) = -i[H_BdG, G] + Lambda * G (elementwise).
"""

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp


def bdg_matrix(h: np.ndarray, Delta: np.ndarray) -> np.ndarray:
    return np.block([[h, Delta], [-Delta.conj(), -h.conj()]])


def damping_matrix(gamma_loc: np.ndarray, gamma_c: float) -> np.ndarray:
    """Normal blocks: -(g_i + g_j)/2 off the diagonal; anomalous blocks: additionally -2 g_c."""
    g = np.asarray(gamma_loc, float)
    normal = -0.5 * (g[:, None] + g[None, :])
    np.fill_diagonal(normal, 0.0)
    anomalous = -0.5 * (g[:, None] + g[None, :]) - 2.0 * gamma_c
    return np.block([[normal, anomalous], [anomalous, normal]])


def evolve_covariance(h, Delta, gamma_loc, gamma_c, G0, times, rtol=1e-10, atol=1e-12) -> np.ndarray:
    times = np.asarray(times, float)
    M = G0.shape[0]
    Hs = sp.csr_matrix(bdg_matrix(np.asarray(h, complex), np.asarray(Delta, complex)))
    Lam = damping_matrix(gamma_loc, gamma_c)

    def rhs(t, y):
        G = y.reshape(M, M)
        HG = Hs @ G
        GH = (Hs @ G.conj().T).conj().T  # G H = (H G^dag)^dag for Hermitian H
        return (-1j * (HG - GH) + Lam * G).ravel()

    y0 = np.asarray(G0, complex).ravel()
    if times[-1] == 0.0:
        return np.repeat(y0.reshape(1, M, M), len(times), axis=0)
    sol = solve_ivp(rhs, (0.0, times[-1]), y0, t_eval=times, method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T.reshape(len(times), M, M)


def product_covariance(occ: np.ndarray) -> np.ndarray:
    occ = np.asarray(occ, float)
    return np.diag(np.concatenate([1.0 - occ, occ])).astype(complex)


def occupation(G: np.ndarray) -> np.ndarray:
    N = G.shape[0] // 2
    return np.diag(G[N:, N:]).real


def hopping(G: np.ndarray, i: int, j: int) -> complex:
    """<c_i^dag c_j>."""
    N = G.shape[0] // 2
    return G[N + i, N + j]


def pairing(G: np.ndarray, i: int, j: int) -> complex:
    """<c_i c_j>."""
    N = G.shape[0] // 2
    return G[i, N + j]
