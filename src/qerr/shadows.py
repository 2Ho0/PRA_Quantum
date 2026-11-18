"""Classical shadows from local random unitaries followed by z measurements (Huang, Kueng and Preskill 2020).

A shot applies a single-qubit unitary u to every qubit and measures z; the outcome b = 0 is the +1 eigenvalue of
sigma^z. For a unitary 2-design the single-qubit snapshot 3 u^dag |b><b| u - 1 is an unbiased estimate of the state,
so 3 (u O u^dag)_{bb} - Tr O estimates <O>, and products over different qubits estimate tensor products.
sigma^+ = |0><1| raises the magnetisation: <sigma^+> = rho[1, 0] = (<X> + i <Y>)/2.

Shots that share a unitary are not independent draws of the snapshot, so a squared modulus |<O>|^2 is estimated only
from pairs of shots taken with different unitaries (`cross_unit_square`).
"""

import numpy as np

SIGMA_PLUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
_PAULIS = (np.array([[0, 1], [1, 0]], dtype=complex), np.array([[0, -1j], [1j, 0]]), np.diag([1.0 + 0j, -1.0]))
# o = Tr(O P) / 2 for P = (I, X, Y, Z): coefficients of sigma^+ and sigma^- in <O> = sum_P o_P <P>
PAULI_COEF = {"+": np.array([0, 0.5, 0.5j, 0]), "-": np.array([0, 0.5, -0.5j, 0])}


def sigma_plus_table(u: np.ndarray) -> np.ndarray:
    """Snapshot values of sigma^+ for outcome 0 and 1: 3 (u sigma^+ u^dag)_{bb}, shape u.shape[:-2] + (2,)."""
    m = u @ SIGMA_PLUS @ np.conj(np.swapaxes(u, -1, -2))
    return 3.0 * np.diagonal(m, axis1=-2, axis2=-1)


def cross_unit_square(sums: np.ndarray, shots: int, weights=None) -> np.ndarray:
    """Unbiased |E x|^2 from per-unit sums S_u (axis 0) of `shots` shots each, using only shot pairs from different
    units: (|sum_u w_u S_u|^2 - sum_u w_u^2 |S_u|^2) / ((W^2 - sum_u w_u^2) shots^2), W = sum_u w_u.

    weights are unit multiplicities (bootstrap); copies of the same unit are never paired.
    """
    S = np.asarray(sums)
    w = np.ones(S.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    wb = w.reshape((-1,) + (1,) * (S.ndim - 1))
    total = np.sum(wb * S, axis=0)
    same = np.sum(wb**2 * np.abs(S) ** 2, axis=0)
    W = w.sum()
    return (np.abs(total) ** 2 - same) / ((W**2 - np.sum(w**2)) * shots**2)


# --------------------------------------------------------------------------- exact inversion for a fixed unitary set
# For a finite set of unitaries the measurement channel is not exactly depolarising, so the snapshot above is unbiased
# only on average over unitary draws. With one fixed set reused at every time, the deviation mixes Bloch components of
# different charge and biases decay ratios. The least-squares (exact) inverse of the channel of the actual set removes
# it: with p = (Tr(P Pi_b))_P the Pauli vector of the measured projector Pi_b = u^dag|b><b|u and
# C = sum_units sum_b p p^T, the estimate of <O> = sum_P o_P <P> on k qubits is (2^k / M) sum_shots (C^-1 o) . p.


def bloch_rows(u: np.ndarray) -> np.ndarray:
    """Bloch vector of u^dag|0> (the direction measured as outcome 0): m_a = (u sigma_a u^dag)_{00}."""
    return np.stack([np.real((u @ s @ np.conj(np.swapaxes(u, -1, -2)))[..., 0, 0]) for s in _PAULIS], axis=-1)


def shot_paulis(m: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """p[unit, shot, qubit, :] = (1, s m) with s = +1 for outcome 0 and -1 for outcome 1.

    m: (units, qubits, 3); bits: (units, shots, qubits)."""
    s = 1.0 - 2.0 * np.asarray(bits, dtype=float)
    out = np.empty(s.shape + (4,))
    out[..., 0] = 1.0
    out[..., 1:] = s[..., None] * m[:, None, :, :]
    return out


def exact_duals(m: np.ndarray, sets, ops) -> np.ndarray:
    """d_j = C_j^-1 o_j for each qubit set j and operator string ops[j] ("+", "-" per qubit), all sets of one size.

    C_j = sum_units kron_{q in set} 2 blockdiag(1, m m^T) (the sum over outcomes of p p^T factorises per unit)."""
    A = np.zeros(m.shape[:-1] + (4, 4))
    A[..., 0, 0] = 2.0
    A[..., 1:, 1:] = 2.0 * m[..., :, None] * m[..., None, :]
    out = []
    for S, op in zip(sets, ops):
        C = A[:, S[0]]
        for q in S[1:]:
            n = C.shape[1]
            C = np.einsum("kab,kcd->kacbd", C, A[:, q]).reshape(C.shape[0], n * 4, n * 4)
        o = PAULI_COEF[op[0]]
        for c in op[1:]:
            o = np.kron(o, PAULI_COEF[c])
        out.append(np.linalg.solve(C.sum(axis=0), o))
    return np.array(out)


def set_values(d: np.ndarray, p: np.ndarray, sets, shots_per_unit: int) -> np.ndarray:
    """Per-shot contributions x[..., j] = 2^k d_j . kron_q p[..., q] / shots_per_unit; their sum over all shots is the
    estimate of <O_j>. d: (sets, 4^k); p: (..., qubits, 4)."""
    idx = np.asarray(sets)
    k = idx.shape[1]
    D = d.reshape((len(idx),) + (4,) * k)
    cols = [p[..., idx[:, i], :] for i in range(k)]
    if k == 1:
        x = np.einsum("na,...na->...n", D, cols[0])
    elif k == 2:
        x = np.einsum("nab,...na,...nb->...n", D, *cols)
    elif k == 3:  # a loop of matrix products is about 2.5 times faster than the three-way einsum
        lead = p.shape[:-2]
        P = p.reshape((-1,) + p.shape[-2:])
        x = np.empty((P.shape[0], len(idx)), dtype=complex)
        for n, (a, b, c) in enumerate(idx):
            ab = (P[:, a, :, None] * P[:, b, None, :]).reshape(P.shape[0], 16)
            x[:, n] = np.einsum("sk,sk->s", ab @ D[n].reshape(16, 4), P[:, c, :])
        x = x.reshape(lead + (len(idx),))
    else:
        raise ValueError("sets of 1 to 3 qubits")
    return (2**k) * x / shots_per_unit


def variance_subtracted_square(x: np.ndarray) -> np.ndarray:
    """Unbiased |E sum x|^2 from per-shot contributions x (units, shots, sets) whose unit means may differ:
    |sum x|^2 - sum_units Var(S_unit), with Var(S_unit) = M/(M-1) sum_shots |x - mean_unit x|^2 (shots independent)."""
    M = x.shape[1]
    S = x.sum(axis=1)
    V = M / (M - 1) * np.sum(np.abs(x - x.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return np.abs(S.sum(axis=0)) ** 2 - V.sum(axis=0)
