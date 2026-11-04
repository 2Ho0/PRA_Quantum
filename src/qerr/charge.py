"""U(1) charge sectors B_q = {X : [S^z, X] = q X}, superoperator charge parts, and the protective collective rate."""

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


def superop_charge_part(n: int, S: np.ndarray, Q: int) -> np.ndarray:
    """S_Q for a column-stacking superoperator (vec index a + D b <-> |a><b|): S_Q maps B_p into B_{p+Q}."""
    ch = _charge_matrix(n).ravel(order="F")
    return np.where(np.isclose(ch[:, None] - ch[None, :], Q), S, 0.0)


def superop_charges(n: int, S: np.ndarray, tol: float = 1e-12) -> dict:
    """{Q: ||S_Q||_F} for the nonzero superoperator charge parts."""
    out = {}
    for Q in range(-2 * n, 2 * n + 1):
        norm = np.linalg.norm(superop_charge_part(n, S, Q))
        if norm > tol:
            out[Q] = float(norm)
    return out


def protective_collective_rate(Gamma: np.ndarray, tol: float = 1e-10) -> float:
    """Largest gamma with Gamma - gamma * 11^T PSD: 1/(1^T Gamma^+ 1), or 0 if 1 is not in range(Gamma)."""
    one = np.ones(len(Gamma))
    x = np.linalg.pinv(Gamma, hermitian=True) @ one
    if np.linalg.norm(Gamma @ x - one) > tol * np.sqrt(len(one)):
        return 0.0
    return float(1.0 / (one @ x))
