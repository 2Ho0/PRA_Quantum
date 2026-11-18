from pathlib import Path

import numpy as np
import pytest

from qerr.mpembadata import decode, load_dephasing, load_unitaries
from qerr.shadows import (bloch_rows, cross_unit_square, exact_duals, set_values, shot_paulis, sigma_plus_table,
                          variance_subtracted_square)

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "mpemba2024"


def haar(rng, n):
    """n Haar-random 2x2 unitaries."""
    z = (rng.normal(size=(n, 2, 2)) + 1j * rng.normal(size=(n, 2, 2))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r, axis1=-2, axis2=-1)
    return q * (d / np.abs(d))[:, None, :]


def test_decode_readme_example():
    # ReadMe: 1918 is the bit string 011101111110 for 12 ions, qubit 1 at the least significant (rightmost) bit
    assert decode(np.array([1918]), 12)[0].tolist() == [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0]


def test_sigma_plus_table_vanishes_for_z_measurements():
    assert np.allclose(sigma_plus_table(np.eye(2)[None, None]), 0)


def test_sigma_plus_shadow_is_unbiased_over_haar_unitaries():
    rng = np.random.default_rng(3)
    r = np.array([0.6, -0.3, 0.5])  # Bloch vector of a mixed single-qubit state
    rho = 0.5 * np.array([[1 + r[2], r[0] - 1j * r[1]], [r[0] + 1j * r[1], 1 - r[2]]])
    u = haar(rng, 40000)
    p = np.real(np.einsum("nab,bc,nac->na", u, rho, u.conj()))  # Born probabilities of z outcomes after u
    table = sigma_plus_table(u[:, None])[:, 0]                    # (n, 2): estimate for outcome 0 and 1
    est = np.mean(np.sum(p * table, axis=1))
    assert abs(est - rho[1, 0]) < 0.02  # <sigma^+> = Tr(rho |0><1|) = rho[1, 0] = (r_x + i r_y)/2


def test_cross_unit_square_matches_explicit_pairs_from_different_units():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))  # 4 units x 3 shots
    w = np.array([2, 0, 1, 3])                                    # bootstrap multiplicities
    copies = [(u, k) for u in range(4) for _ in range(w[u]) for k in range(3)]
    num = sum(x[u, k] * np.conj(x[v, l]) for (u, k) in copies for (v, l) in copies if u != v)
    den = sum(1 for (u, _) in copies for (v, _) in copies if u != v)
    got = cross_unit_square(x.sum(axis=1), 3, weights=w)
    assert np.isclose(got, np.real(num / den))


def test_cross_unit_square_removes_the_unit_effect_bias():
    rng = np.random.default_rng(7)
    mu, n_units, shots, reps = 0.2 + 0.1j, 50, 10, 4000
    naive, cross = [], []
    for _ in range(reps):
        unit = mu + 0.3 * (rng.normal(size=(n_units, 1)) + 1j * rng.normal(size=(n_units, 1)))
        x = unit + rng.normal(size=(n_units, shots))
        naive.append(abs(x.mean()) ** 2)
        cross.append(cross_unit_square(x.sum(axis=1), shots))
    assert abs(np.mean(cross) - abs(mu) ** 2) < 0.004
    assert np.mean(naive) - abs(mu) ** 2 > 0.005  # the naive estimate is biased upward


def _random_state(rng, n):
    a = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))
    r = a @ a.conj().T
    return r / np.trace(r)


@pytest.mark.parametrize("ops", ["+", "++", "+-", "+++"])
def test_exact_duals_recover_the_expectation_for_a_fixed_small_unitary_set(ops):
    # with the exact outcome probabilities of a correlated state, the estimator returns <O> exactly, for any fixed set
    rng = np.random.default_rng(11)
    n_units, m = 100, len(ops)  # 25 units leave the 3-qubit channel rank-deficient (62 of 64)
    u = haar(rng, n_units * m).reshape(n_units, m, 2, 2)
    rho = _random_state(rng, m)
    sp = np.array([[0, 1], [0, 0]], complex)
    single = {"+": sp, "-": sp.conj().T}
    O = single[ops[0]]
    for c in ops[1:]:
        O = np.kron(O, single[c])
    target = np.trace(rho @ O)
    d = exact_duals(bloch_rows(u), [tuple(range(m))], [ops])[0]
    total = 0.0
    for k in range(n_units):
        U = u[k, 0]
        for q in range(1, m):
            U = np.kron(U, u[k, q])
        probs = np.real(np.diag(U @ rho @ U.conj().T))  # outcome b as an m-bit string, qubit 0 most significant
        for b in range(2**m):
            bits = np.array([(b >> (m - 1 - q)) & 1 for q in range(m)], dtype=np.uint8)
            p = shot_paulis(bloch_rows(u[k][None]), bits[None, None, :])[0, 0]  # (m, 4)
            total += probs[b] * set_values(d[None, :], p[None], [tuple(range(m))], shots_per_unit=1)[0, 0]
    assert np.isclose(total, target, atol=1e-10)


def test_variance_subtracted_square_is_unbiased_with_unit_dependent_means():
    rng = np.random.default_rng(13)
    n_units, shots, reps = 40, 30, 3000
    mu = (rng.normal(size=n_units) + 1j * rng.normal(size=n_units)) * 0.05  # per-unit means (one set)
    vals = []
    for _ in range(reps):
        x = mu[:, None, None] / shots + (rng.normal(size=(n_units, shots, 1)) + 1j * rng.normal(size=(n_units, shots, 1))) * 0.02
        vals.append(variance_subtracted_square(x)[0])
    assert abs(np.mean(vals) - abs(mu.sum()) ** 2) < 0.02 * abs(mu.sum()) ** 2 + 2e-4


@pytest.mark.skipif(not (RAW / "MpembaDataUpload").exists(), reason="run data/fetch_mpemba2024.py first")
def test_load_dephasing_and_unitaries_shapes():
    times, bits = load_dephasing(RAW, "0p5")
    assert times.tolist() == list(range(9)) and bits.shape == (9, 300, 30, 12) and bits.dtype == np.uint8
    u = load_unitaries(RAW)
    assert u.shape == (500, 12, 2, 2)
    assert np.allclose(np.einsum("nqab,nqcb->nqac", u, u.conj()), np.eye(2))
