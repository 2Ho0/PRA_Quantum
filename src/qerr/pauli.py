"""Pauli strings as sorted tuples of (site, letter) pairs, letter in 'XYZ'; the identity is ()."""

from functools import reduce

import numpy as np
import scipy.sparse as sp

from qerr.spins import pauli_op

_MUL = {("X", "Y"): (1j, "Z"), ("Y", "X"): (-1j, "Z"), ("Y", "Z"): (1j, "X"),
        ("Z", "Y"): (-1j, "X"), ("Z", "X"): (1j, "Y"), ("X", "Z"): (-1j, "Y")}


def weight(p) -> int:
    return len(p)


def pmul(p, q):
    """p q = phase * r for Pauli strings p, q."""
    a, b = dict(p), dict(q)
    phase, out = 1, {}
    for site in sorted(set(a) | set(b)):
        x, y = a.get(site), b.get(site)
        if x is None or y is None:
            out[site] = x or y
        elif x != y:
            ph, z = _MUL[(x, y)]
            phase, out[site] = phase * ph, z
    return phase, tuple(sorted(out.items()))


def comm_i(p, q):
    """i[p, q] = coef * r with real coef (+-2) if p and q anticommute, else (0.0, None)."""
    ph, r = pmul(p, q)
    if pmul(q, p)[0] == ph:
        return 0.0, None
    return float((2j * ph).real), r  # i (pq - qp) = i (ph - (-ph)) r


def expectation_tables(strings, n: int):
    """Tables with <P_k> = sum_c phase[k, c] rho[c, idx[k, c]], since P|c> = phase(c) |c xor xmask>."""
    D = 2**n
    c = np.arange(D)
    idx = np.empty((len(strings), D), np.int64)
    phase = np.ones((len(strings), D), complex)
    for k, p in enumerate(strings):
        xmask = 0
        for site, a in p:
            bit = (c >> (n - 1 - site)) & 1
            if a in "XY":
                xmask |= 1 << (n - 1 - site)
            if a == "Y":
                phase[k] *= np.where(bit == 0, 1j, -1j)
            elif a == "Z":
                phase[k] *= np.where(bit == 0, 1.0, -1.0)
        idx[k] = c ^ xmask
    return idx, phase


def expectations(rho, idx, phase) -> np.ndarray:
    return (phase * rho[np.arange(rho.shape[0])[None, :], idx]).sum(axis=1).real


def psparse(p, n: int) -> sp.csr_matrix:
    """Sparse matrix of the string: P[c xor xmask, c] = phase(c)."""
    idx, phase = expectation_tables([p], n)
    D = 2**n
    return sp.csr_matrix((phase[0], (idx[0], np.arange(D))), shape=(D, D))


def pmatrix(p, n: int) -> np.ndarray:
    """Dense matrix of the string on n qubits (qerr.spins conventions)."""
    return reduce(np.matmul, [pauli_op(n, s, a.lower()) for s, a in p], np.eye(2**n, dtype=complex))
