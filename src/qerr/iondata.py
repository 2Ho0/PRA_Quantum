"""Raw trapped-ion data of Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419 (CC BY 4.0).

Conventions (dataset ReadMe): every shot is a decimal number whose least significant bit is qubit 1;
bit 0 is the +1 eigenvalue of the measured Pauli, bit 1 the -1 eigenvalue.
10-ion data: two product initial states, 200 measurement settings ("random unitaries"), each with
4 x 50 shots at 21 times. The settings are random Pauli bases (Kraft et al., App. B 3) but are stored
only as indices, so `infer_bases` recovers them from the t = 0 statistics of the two known initial states:
  state 1, Bloch vector (sin 0.33 pi, 0, -cos 0.33 pi): X -> +0.86, Y -> 0, Z -> -0.51
  state 2, Bloch vector (0, 1, 0):                      Y -> +1, X and Z -> 0
"""

from pathlib import Path

import numpy as np
import scipy.io as sio

LETTERS = "XYZ"
Y_THRESHOLD, XZ_THRESHOLD = 0.5, 0.2


def decode_shots(dec: np.ndarray, n: int) -> np.ndarray:
    """Decimal shot records -> int8 spins (+1 for bit 0), last axis = qubit 1..n."""
    ints = np.asarray(dec).astype(np.int64)
    bits = (ints[..., None] >> np.arange(n, dtype=np.int64)) & 1
    return (1 - 2 * bits).astype(np.int8)


def _cells(path: Path) -> list:
    return [np.asarray(c) for c in sio.loadmat(path)["dataT"].flat]


def load_10ion(root, state: str, n_settings: int = 200, n_batches: int = 4):
    """times_ms (21,), spins (21, n_settings, 50 n_batches, 10) for folder root/state with DataHLRU1..n.mat.

    Only the first n_batches batches of 50 shots are used, so every setting carries equal weight
    (state 2, settings 1 and 59, hold 8 batches instead of the 4 described in the ReadMe).
    """
    root = Path(root)
    per_setting = []
    for k in range(1, n_settings + 1):
        cells = _cells(root / state / f"DataHLRU{k}.mat")[:n_batches]
        times = cells[0][:, 0].astype(float) / 1000.0
        per_setting.append(np.concatenate([c[:, 1:] for c in cells], axis=1))  # (21, 200)
    dec = np.stack(per_setting, axis=1)                                          # (21, S, 200)
    return times, decode_shots(dec, 10)


def load_neel51(path):
    """times_ms (11,), spins (11, 1000, 51) of the 51-ion Neel quench measured in the z basis."""
    cells = _cells(Path(path))
    times = cells[0][:, 0].astype(float) / 1000.0
    dec = np.concatenate([c[:, 1:] for c in cells], axis=1)
    return times, decode_shots(dec, 51)


def infer_bases(m1, m2, sem1, sem2):
    """Pauli basis (0 = X, 1 = Y, 2 = Z) of every (setting, qubit) from t = 0 means of states 1 and 2.

    Y if m2 > 0.5, else X if m1 > 0.2, else Z. margin = distance to the decisive threshold(s) in units of
    the larger SEM; values below ~3 mark settings that should be reported as ambiguous.
    """
    m1, m2 = np.asarray(m1, float), np.asarray(m2, float)
    se = np.maximum(np.asarray(sem1, float), np.asarray(sem2, float))
    is_y = m2 > Y_THRESHOLD
    bases = np.where(is_y, 1, np.where(m1 > XZ_THRESHOLD, 0, 2)).astype(np.int8)
    d_y = np.abs(m2 - Y_THRESHOLD)
    margin = np.where(is_y, d_y, np.minimum(d_y, np.abs(m1 - XZ_THRESHOLD))) / se
    return bases, margin


def _matching(bases, string):
    mask = np.ones(bases.shape[0], bool)
    for site, letter in string:
        mask &= bases[:, site] == LETTERS.index(letter)
    return np.flatnonzero(mask)


def _products(spins, idx, string):
    """(T, len(idx), shots) products of the outcomes on the string's sites."""
    sub = spins[:, idx]  # (T, n_idx, shots, n); two-step indexing keeps the axis order
    prod = np.ones(sub.shape[:3], np.int8)
    for site, _ in string:
        prod = prod * sub[..., site]
    return prod


def per_setting_means(spins, bases, string):
    """Indices of the settings that measure `string`, and their shot means, shape (n_set, T)."""
    idx = _matching(bases, string)
    return idx, _products(spins, idx, string).mean(axis=2).T


def pauli_means(spins, bases, strings):
    """Estimates of <P> for Pauli strings ((site, 'X'|'Y'|'Z'), ...) at every time: mean, SEM (T, S), n_settings."""
    T = spins.shape[0]
    mean, sem = np.full((T, len(strings)), np.nan), np.full((T, len(strings)), np.nan)
    nset = np.zeros(len(strings), int)
    for k, string in enumerate(strings):
        idx = _matching(bases, string)
        nset[k] = len(idx)
        if len(idx) == 0:
            continue
        prod = _products(spins, idx, string).reshape(T, -1).astype(float)
        mean[:, k] = prod.mean(axis=1)
        sem[:, k] = prod.std(axis=1, ddof=1) / np.sqrt(prod.shape[1])
    return mean, sem, nset
