"""Model builders: long-range XY Hamiltonian and a charge-definite basis of 1- and 2-local terms."""

from itertools import combinations
from typing import NamedTuple

import numpy as np

from .spins import pauli_op


class Term(NamedTuple):
    label: str
    op: np.ndarray
    abs_charge: int


def xy_couplings(n: int, J0: float = 1.0, alpha: float = 1.0) -> np.ndarray:
    d = np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(float)
    with np.errstate(divide="ignore"):
        J = np.where(d > 0, J0 / d**alpha, 0.0)
    return J


def xy_hamiltonian(n: int, J: np.ndarray, B: np.ndarray) -> np.ndarray:
    """H = 1/2 sum_{i<j} J_ij (xx + yy) + sum_i B_i z_i."""
    H = sum(B[i] * pauli_op(n, i, "z") for i in range(n))
    for i, j in combinations(range(n), 2):
        if J[i, j] != 0.0:
            H = H + 0.5 * J[i, j] * (pauli_op(n, i, "x") @ pauli_op(n, j, "x")
                                     + pauli_op(n, i, "y") @ pauli_op(n, j, "y"))
    return H


# Two-site Hermitian combinations of definite |q|: (coefficients over sigma^a_i sigma^b_j, |q|)
_PAIR_TERMS = {
    "zz": ({("z", "z"): 1.0}, 0),
    "xx+yy": ({("x", "x"): 0.5, ("y", "y"): 0.5}, 0),
    "xy-yx": ({("x", "y"): 0.5, ("y", "x"): -0.5}, 0),
    "xz": ({("x", "z"): 1.0}, 1),
    "zx": ({("z", "x"): 1.0}, 1),
    "yz": ({("y", "z"): 1.0}, 1),
    "zy": ({("z", "y"): 1.0}, 1),
    "xx-yy": ({("x", "x"): 0.5, ("y", "y"): -0.5}, 2),
    "xy+yx": ({("x", "y"): 0.5, ("y", "x"): 0.5}, 2),
}
_SITE_TERMS = {"z": 0, "x": 1, "y": 1}


def charge_basis_terms(n: int, max_range: int | None = None) -> list[Term]:
    terms = [Term(f"{a}{i}", pauli_op(n, i, a), q) for i in range(n) for a, q in _SITE_TERMS.items()]
    for i, j in combinations(range(n), 2):
        if max_range is not None and j - i > max_range:
            continue
        for name, (coeffs, q) in _PAIR_TERMS.items():
            op = sum(c * pauli_op(n, i, a) @ pauli_op(n, j, b) for (a, b), c in coeffs.items())
            terms.append(Term(f"{name}:{i},{j}", op, q))
    return terms


def delta_h(terms: list[Term], g: np.ndarray) -> np.ndarray:
    return sum(gm * t.op for gm, t in zip(g, terms))


def sample_coeffs(M: int, sigma, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """Antithetic Gaussian samples: rows (2k, 2k+1) are (g, -g), g ~ N(0, diag(sigma^2))."""
    g = rng.normal(size=(n_pairs, M)) * np.asarray(sigma)
    return np.stack([g, -g], axis=1).reshape(2 * n_pairs, M)
