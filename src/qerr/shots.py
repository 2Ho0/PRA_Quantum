"""Shot-level simulation of random-Pauli-basis measurements, the protocol of the BEQS 10-ion data.

pauli_tensor(rho, n) holds every Pauli expectation value, E[l_0, ..., l_{n-1}] = Tr[rho s_{l_0} x ... x s_{l_{n-1}}] with
l in (I, X, Y, Z) and site 0 the first tensor factor (the qerr.spins convention). A product-basis measurement with
letters L (0 = X, 1 = Y, 2 = Z per site, as in qerr.iondata) has outcome probabilities
    p(b) = 2^-n sum_S (-1)^{b . S} E_S,
where E_S carries L_i on the sites in S and I elsewhere: a Walsh-Hadamard transform of 2^n entries of E. The outcome
index b has bit 0 for eigenvalue +1 and site 0 as the most significant bit, so sampled spins match qerr.iondata.
"""

import numpy as np

from . import spins as _spins

_PAULI4 = np.array([np.eye(2), [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], complex)
_SITE = np.array([P.T.reshape(4) for P in _PAULI4])   # Tr[rho_site s_l] = sum_{rc} rho_rc (s_l)_cr, index 2r + c
_H = np.array([[1.0, 1.0], [1.0, -1.0]])


def pauli_tensor(rho, n) -> np.ndarray:
    """(4,)*n real tensor of all Pauli expectation values of rho, in O(n 4^n)."""
    A = np.asarray(rho, complex).reshape((2,) * (2 * n))
    A = A.transpose([ax for i in range(n) for ax in (i, n + i)]).reshape((4,) * n)
    for i in range(n):
        A = np.moveaxis(np.tensordot(_SITE, A, axes=([1], [i])), 0, i)
    return A.real


def setting_probabilities(E, bases) -> np.ndarray:
    """Outcome distribution (2^n,) of the product basis `bases` (0 = X, 1 = Y, 2 = Z per site)."""
    n = E.ndim
    sub = E[np.ix_(*[[0, int(b) + 1] for b in bases])]
    for i in range(n):
        sub = np.moveaxis(np.tensordot(_H, sub, axes=([1], [i])), 0, i)
    return sub.reshape(-1) / 2.0**n


def sample_spins(p, n_shots, rng, n) -> np.ndarray:
    """(n_shots, n) int8 spins (+1 for bit 0) drawn from the outcome distribution p."""
    p = np.clip(np.asarray(p, float), 0.0, None)
    idx = rng.choice(len(p), size=n_shots, p=p / p.sum())
    return (1 - 2 * _spins.bits(n)[idx]).astype(np.int8)


def resample_shots(spins, rng) -> np.ndarray:
    """Shots drawn with replacement within every leading cell: spins (..., n_shots, n) -> the same shape.

    Resampling the shots of every (time, setting) keeps the measurement design (every string stays measured in the
    same settings) and adds shot noise of the measured size to every Pauli estimate, with its correlations between
    strings measured in the same setting: one SIMEX level.
    """
    spins = np.asarray(spins)
    idx = rng.integers(0, spins.shape[-2], size=spins.shape[:-1])
    return np.take_along_axis(spins, idx[..., None], axis=-2)


def rotated_setting_probabilities(E, bases, phi) -> np.ndarray:
    """setting_probabilities of the state rotated by exp(-i phi S^z), from the Pauli tensor E of the unrotated state.

    The rotation mixes X and Y on every site (<X>' = cos phi <X> - sin phi <Y>, <Y>' = sin phi <X> + cos phi <Y>), so
    only the two needed components per site are formed: O(4^n) instead of a new Pauli tensor.
    """
    n = E.ndim
    c, s = np.cos(phi), np.sin(phi)
    rows = {0: [0.0, c, -s, 0.0], 1: [0.0, s, c, 0.0], 2: [0.0, 0.0, 0.0, 1.0]}
    sub = E
    for i in range(n):
        M = np.array([[1.0, 0.0, 0.0, 0.0], rows[int(bases[i])]])
        sub = np.moveaxis(np.tensordot(M, sub, axes=([1], [i])), 0, i)
    for i in range(n):
        sub = np.moveaxis(np.tensordot(_H, sub, axes=([1], [i])), 0, i)
    return sub.reshape(-1) / 2.0**n
