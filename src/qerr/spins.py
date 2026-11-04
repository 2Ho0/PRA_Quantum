"""Many-body spin-1/2 operators in the computational basis.

Conventions: |0> is spin up (sigma^z = +1); site 0 is the leftmost Kronecker
factor, so basis index b = sum_i bit_i * 2**(n-1-i).
"""

from functools import reduce

import numpy as np

_SINGLE = {
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
    "+": np.array([[0, 1], [0, 0]], dtype=complex),
    "-": np.array([[0, 0], [1, 0]], dtype=complex),
}


def pauli_op(n: int, site: int, kind: str) -> np.ndarray:
    factors = [np.eye(2, dtype=complex)] * n
    factors[site] = _SINGLE[kind]
    return reduce(np.kron, factors)


def bits(n: int) -> np.ndarray:
    """bits[b, i] is the bit of site i in basis state b."""
    b = np.arange(2**n)
    return (b[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1


def zvals(n: int) -> np.ndarray:
    return 1 - 2 * bits(n)


def total_sz(n: int) -> np.ndarray:
    """Diagonal of S^z = sum_i sigma^z_i / 2."""
    return 0.5 * zvals(n).sum(axis=1)


def basis_state(n: int, bitstring: str) -> np.ndarray:
    rho = np.zeros((2**n, 2**n), dtype=complex)
    idx = int(bitstring, 2)
    rho[idx, idx] = 1.0
    return rho
