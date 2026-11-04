"""U(1) charge sectors B_q = {X : [S^z, X] = q X} and the protective collective rate."""

import numpy as np

from . import spins


def _charge_matrix(n: int) -> np.ndarray:
    m = spins.total_sz(n)
    return m[:, None] - m[None, :]


def charge_projector(n: int, X: np.ndarray, q: int) -> np.ndarray:
    """P_q(X): keep matrix elements X_bb' with M_b - M_b' = q."""
    return np.where(np.isclose(_charge_matrix(n), q), X, 0.0)


def charge_decompose(n: int, X: np.ndarray) -> dict:
    return {q: charge_projector(n, X, q) for q in range(-n, n + 1)}


def protective_collective_rate(Gamma: np.ndarray, tol: float = 1e-10) -> float:
    """Largest gamma with Gamma - gamma * 11^T PSD: 1/(1^T Gamma^+ 1), or 0 if 1 is not in range(Gamma)."""
    one = np.ones(len(Gamma))
    x = np.linalg.pinv(Gamma, hermitian=True) @ one
    if np.linalg.norm(Gamma @ x - one) > tol * np.sqrt(len(one)):
        return 0.0
    return float(1.0 / (one @ x))
