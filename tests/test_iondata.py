from pathlib import Path

import numpy as np
import pytest

from qerr.iondata import decode_shots, infer_bases, load_10ion, pauli_means, per_setting_means

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.mark.skipif(not (RAW / "state1_2" / "state2").exists(), reason="run data/fetch_beqs.py first")
def test_load_10ion_uses_four_batches_per_setting():
    # state 2 settings 1 and 59 carry 8 batches; the ReadMe format is 4 batches of 50 shots
    for state in ("state1", "state2"):
        times, spins = load_10ion(RAW / "state1_2", state)
        assert times.shape == (21,) and spins.shape == (21, 200, 200, 10)

BLOCH1 = np.array([np.sin(0.33 * np.pi), 0.0, -np.cos(0.33 * np.pi)])  # state 1, (x, y, z)
BLOCH2 = np.array([0.0, 1.0, 0.0])                                      # state 2


def sample(rng, p_plus, shots):
    """+-1 outcomes with P(+1) = p_plus; p_plus has shape (S, n) -> (S, shots, n)."""
    return np.where(rng.random((p_plus.shape[0], shots, p_plus.shape[1])) < p_plus[:, None, :], 1, -1)


def test_decode_readme_example():
    assert decode_shots(np.array([101]), 10)[0].tolist() == [-1, 1, -1, 1, 1, -1, -1, 1, 1, 1]


def test_neel_decimal_decodes_alternating():
    s = decode_shots(np.array([1501199875790165.0]), 51)[0]
    assert s[0] == -1 and np.all(s[1:] == -s[:-1])


def test_infer_bases_recovers_synthetic():
    rng = np.random.default_rng(0)
    bases = rng.integers(0, 3, size=(200, 10))
    flip = 0.02
    s1 = sample(rng, (1 + (1 - 2 * flip) * BLOCH1[bases]) / 2, 200)
    s2 = sample(rng, (1 + (1 - 2 * flip) * BLOCH2[bases]) / 2, 200)
    sem = lambda s: s.std(axis=1) / np.sqrt(s.shape[1])
    inferred, margin = infer_bases(s1.mean(axis=1), s2.mean(axis=1), sem(s1), sem(s2))
    assert np.array_equal(inferred, bases)
    assert margin.min() > 3


def test_infer_bases_flags_ambiguous():
    se = np.array([[0.07]])
    _, margin = infer_bases(np.array([[0.21]]), np.array([[0.0]]), se, se)
    assert margin[0, 0] < 3


def test_pauli_means_unbiased():
    rng = np.random.default_rng(1)
    T, S, shots, n = 2, 300, 200, 4
    bases = rng.integers(0, 3, size=(S, n))
    bloch = np.array([[0.6, 0.0, -0.8], [0.0, 0.6, 0.8], [0.8, -0.6, 0.0], [0.0, 0.0, 1.0]])
    p = (1 + bloch[np.arange(n)[None, :], bases]) / 2
    spins = np.stack([sample(rng, p, shots) for _ in range(T)]).astype(np.int8)
    strings = [((0, "X"),), ((1, "Y"), (2, "X")), ((0, "Z"), (1, "Z"), (3, "Z"))]
    mean, sem, nset = pauli_means(spins, bases, strings)
    for k, exact in enumerate([0.6, 0.6 * 0.8, -0.8 * 0.8 * 1.0]):
        assert nset[k] > 0
        assert np.all(np.abs(mean[:, k] - exact) < 4 * sem[:, k])
    idx, per = per_setting_means(spins, bases, strings[1])
    assert per.shape == (len(idx), T) and len(idx) == nset[1]
    assert np.allclose(per.mean(axis=0), mean[:, 1])
