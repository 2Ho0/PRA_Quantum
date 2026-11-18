"""Raw bitstring data of Andersen et al., Nature 638, 79 (2025); Zenodo 10.5281/zenodo.14060446 (CC BY 4.0).

The data files are Python pickles. pickle.load on downloaded data can run arbitrary code, so `safe_load` unpickles
with a find_class that returns only the numpy constructors needed for arrays (Python docs, "Restricting Globals");
any other global raises UnsafePickleError before it can be called. Plain containers and numbers need no globals.

A distribution is a dict {bitstring as a decimal integer: number of shots}. The excitation number of a shot is the
number of set bits. Which bit belongs to which qubit is not documented in the record, so per-bit marginals are
indexed by bit position (least significant first) and are not mapped to qubit coordinates here.
"""

import pickle
from pathlib import Path

import numpy as np

try:  # numpy >= 2
    from numpy._core.multiarray import _reconstruct, scalar
except ImportError:  # pragma: no cover - numpy 1.x
    from numpy.core.multiarray import _reconstruct, scalar


class UnsafePickleError(pickle.UnpicklingError):
    """The pickle asks for a global outside the whitelist."""


_ALLOWED = {
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
    ("numpy.core.multiarray", "_reconstruct"): _reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): _reconstruct,
    ("numpy.core.multiarray", "scalar"): scalar,
    ("numpy._core.multiarray", "scalar"): scalar,
}


class _WhitelistUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return _ALLOWED[(module, name)]
        except KeyError:
            raise UnsafePickleError(f"refused global {module}.{name}") from None


def safe_load(path):
    """Unpickle a downloaded data file, allowing only numpy array constructors as globals."""
    with open(Path(path), "rb") as fh:
        return _WhitelistUnpickler(fh).load()


def _keys_counts(dist: dict, n_qubits: int):
    counts = np.fromiter(dist.values(), dtype=np.int64, count=len(dist))
    if n_qubits <= 63:
        keys = np.fromiter(dist.keys(), dtype=np.uint64, count=len(dist))
        if keys.size and int(keys.max()) >> n_qubits:
            raise ValueError(f"a bitstring has more than {n_qubits} bits")
        return keys, counts
    raise ValueError("more than 63 qubits is not supported")


def popcount_histogram(dist: dict, n_qubits: int) -> np.ndarray:
    """Shots per excitation number n = 0..n_qubits (int64, length n_qubits + 1)."""
    keys, counts = _keys_counts(dist, n_qubits)
    hist = np.zeros(n_qubits + 1, dtype=np.int64)
    np.add.at(hist, np.bitwise_count(keys).astype(np.intp), counts)
    return hist


def bit_marginals(dist: dict, n_qubits: int):
    """(probability that bit k is set, k = 0..n_qubits-1 from the least significant bit; total shots)."""
    keys, counts = _keys_counts(dist, n_qubits)
    shots = int(counts.sum())
    p = np.empty(n_qubits)
    for k in range(n_qubits):
        p[k] = counts[((keys >> np.uint64(k)) & np.uint64(1)).astype(bool)].sum() / shots
    return p, shots


def sector_moments(hist, n0: int) -> dict:
    """Mean, variance and the share of the sector n0 of an excitation-number histogram, with standard errors.

    se_mean = sqrt(var / S); se_var = sqrt((mu4 - var_pop^2) / S), the large-S standard error of the sample variance;
    se_p0 binomial. var is the unbiased sample variance.
    """
    hist = np.asarray(hist, dtype=float)
    n = np.arange(hist.size, dtype=float)
    shots = hist.sum()
    mean = float(hist @ n / shots)
    d = n - mean
    var_pop = float(hist @ d**2 / shots)
    mu4 = float(hist @ d**4 / shots)
    var = var_pop * shots / (shots - 1)
    p0 = float(hist[n0] / shots)
    return {
        "shots": int(shots),
        "mean": mean,
        "var": var,
        "se_mean": float(np.sqrt(var / shots)),
        "se_var": float(np.sqrt(max(mu4 - var_pop**2, 0.0) / shots)),
        "p0": p0,
        "se_p0": float(np.sqrt(p0 * (1 - p0) / shots)),
    }
