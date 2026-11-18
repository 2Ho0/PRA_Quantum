import os
import pickle

import numpy as np
import pytest

from qerr.googledata import UnsafePickleError, bit_marginals, popcount_histogram, safe_load, sector_moments


class _CallsGetcwd:
    """Pickles as a call of os.getcwd: harmless if it ever ran, but outside the whitelist."""

    def __reduce__(self):
        return (os.getcwd, ())


def test_safe_load_refuses_global_outside_whitelist(tmp_path):
    path = tmp_path / "bad.pkl"
    path.write_bytes(pickle.dumps({"x": _CallsGetcwd()}, protocol=4))
    with pytest.raises(UnsafePickleError):
        safe_load(path)


def test_safe_load_reads_numpy_arrays_lists_and_dicts(tmp_path):
    obj = {
        "times": np.linspace(0.0, 30.0, 4),
        "flags": np.array([[True, False], [False, True]]),
        "qubits": [(2, 6), (2, 5)],
        "dists": [[{3: 5, 12: 1}, {2**34 + 1: 7}]],
    }
    path = tmp_path / "ok.pkl"
    path.write_bytes(pickle.dumps(obj, protocol=4))
    got = safe_load(path)
    assert np.array_equal(got["times"], obj["times"]) and got["flags"].dtype == bool
    assert got["qubits"] == obj["qubits"] and got["dists"] == obj["dists"]


def test_popcount_histogram_counts_excitations():
    dist = {0b0011: 5, 0b0101: 2, 0b0000: 1, 0b1111: 3}
    assert popcount_histogram(dist, 4).tolist() == [1, 0, 7, 0, 3]


def test_popcount_histogram_handles_35_qubit_keys():
    dist = {2**35 - 1: 2, (2**35 - 1) ^ (1 << 17): 4, 1 << 34: 1}
    h = popcount_histogram(dist, 35)
    assert h[35] == 2 and h[34] == 4 and h[1] == 1 and h.sum() == 7


def test_bit_marginals_count_each_bit_position():
    dist = {0b0011: 5, 0b0101: 2, 0b0000: 1, 0b1111: 3}
    p, shots = bit_marginals(dist, 4)
    assert shots == 11
    assert np.allclose(p, np.array([10, 8, 5, 3]) / 11)  # index k = bit k (least significant first)


def test_sector_moments_match_direct_sample_statistics():
    rng = np.random.default_rng(1)
    n = rng.binomial(10, 0.4, size=20000)
    hist = np.bincount(n, minlength=11)
    m = sector_moments(hist, n0=4)
    assert m["shots"] == 20000
    assert np.isclose(m["mean"], n.mean()) and np.isclose(m["var"], n.var(ddof=1))
    assert np.isclose(m["p0"], np.mean(n == 4))
    # standard errors: sd/sqrt(S) for the mean, sqrt((mu4 - var^2)/S) for the variance (large-S form)
    mu4 = np.mean((n - n.mean()) ** 4)
    assert np.isclose(m["se_mean"], np.sqrt(n.var(ddof=1) / n.size), rtol=1e-6)
    assert np.isclose(m["se_var"], np.sqrt((mu4 - n.var() ** 2) / n.size), rtol=1e-3)
    assert np.isclose(m["se_p0"], np.sqrt(m["p0"] * (1 - m["p0"]) / n.size))
