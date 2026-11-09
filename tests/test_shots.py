import numpy as np

from qerr.pauli import pmatrix

PAULI = {"X": np.array([[0, 1], [1, 0]], complex), "Y": np.array([[0, -1j], [1j, 0]], complex),
         "Z": np.array([[1, 0], [0, -1]], complex)}
TO_Z = {"X": np.array([[1, 1], [1, -1]], complex) / np.sqrt(2),          # rows: eigenvectors of X (+1, -1)
        "Y": np.array([[1, -1j], [1, 1j]], complex) / np.sqrt(2),         # rows: eigenvectors of Y (+1, -1)
        "Z": np.eye(2, dtype=complex)}


def random_density(D, rng):
    X = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    rho = X @ X.conj().T
    return rho / np.trace(rho)


def test_pauli_tensor_matches_traces():
    from qerr.shots import pauli_tensor

    n = 3
    rho = random_density(2**n, np.random.default_rng(0))
    E = pauli_tensor(rho, n)
    letters = "IXYZ"
    for idx in np.ndindex(*(4,) * n):
        string = tuple((i, letters[l]) for i, l in enumerate(idx) if l)
        direct = np.trace(rho).real if not string else np.trace(pmatrix(string, n) @ rho).real
        assert abs(E[idx] - direct) < 1e-12


def test_setting_probabilities_match_rotated_diagonal():
    from qerr.shots import pauli_tensor, setting_probabilities

    n = 3
    rng = np.random.default_rng(1)
    rho = random_density(2**n, rng)
    E = pauli_tensor(rho, n)
    for _ in range(10):
        bases = rng.integers(0, 3, n)                     # 0 = X, 1 = Y, 2 = Z, as in qerr.iondata
        U = np.array([[1.0]])
        for b in bases:
            U = np.kron(U, TO_Z["XYZ"[b]])
        direct = np.real(np.diag(U @ rho @ U.conj().T))  # outcome index: bit 0 (site 0 most significant) = +1
        assert np.abs(setting_probabilities(E, bases) - direct).max() < 1e-12


def test_sampled_spins_reproduce_expectations():
    from qerr.shots import pauli_tensor, sample_spins, setting_probabilities

    n = 3
    rng = np.random.default_rng(2)
    rho = random_density(2**n, rng)
    E = pauli_tensor(rho, n)
    bases = np.array([0, 1, 2])
    spins = sample_spins(setting_probabilities(E, bases), 200000, rng, n)   # (shots, n), +1 for bit 0
    assert spins.shape == (200000, n) and spins.dtype == np.int8
    for site, letter in enumerate("XYZ"):
        expect = np.trace(pmatrix(((site, letter),), n) @ rho).real
        assert abs(spins[:, site].mean() - expect) < 5 / np.sqrt(200000)
    pair = np.trace(pmatrix(((0, "X"), (2, "Z")), n) @ rho).real
    assert abs((spins[:, 0] * spins[:, 2]).mean() - pair) < 5 / np.sqrt(200000)


def test_resample_shots_draws_with_replacement_within_each_setting_and_time():
    from qerr.shots import resample_shots

    T, K, S = 3, 4, 50
    labels = np.arange(T * K * S).reshape(T, K, S, 1)  # every shot carries the index of its (time, setting) cell
    out = resample_shots(labels, np.random.default_rng(7))
    assert out.shape == labels.shape and out.dtype == labels.dtype
    cell = out[..., 0] // S
    assert np.array_equal(cell, np.arange(T * K).reshape(T, K, 1) + np.zeros((1, 1, S), int))
    assert any(len(np.unique(out[t, k, :, 0])) < S for t in range(T) for k in range(K))  # with replacement


def test_rotated_setting_probabilities_match_a_rotated_state():
    from qerr.shots import pauli_tensor, rotated_setting_probabilities, setting_probabilities
    from qerr.spins import total_sz

    n = 3
    rng = np.random.default_rng(4)
    rho = random_density(2**n, rng)
    m = total_sz(n)
    E = pauli_tensor(rho, n)
    for phi in (0.0, 0.37, -1.2):
        rotated = rho * np.exp(-1j * phi * (m[:, None] - m[None, :]))  # exp(-i phi S^z) rho exp(i phi S^z)
        for bases in (np.array([0, 1, 2]), np.array([1, 1, 0]), np.array([2, 0, 1])):
            direct = setting_probabilities(pauli_tensor(rotated, n), bases)
            assert np.abs(rotated_setting_probabilities(E, bases, phi) - direct).max() < 1e-12
