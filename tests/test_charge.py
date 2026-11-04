import numpy as np

from qerr import spins
from qerr.charge import charge_decompose, charge_projector, protective_collective_rate
from qerr.lindblad import OpenModel, evolve, lindblad_rhs
from qerr.models import charge_basis_terms, sample_coeffs, xy_couplings, xy_hamiltonian


def random_complex(D, rng):
    return rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))


def test_charge_projector_nonhermitian():
    n = 3
    rng = np.random.default_rng(0)
    X = random_complex(8, rng)
    parts = charge_decompose(n, X)
    assert np.allclose(sum(parts.values()), X)
    Sz = np.diag(spins.total_sz(n)).astype(complex)
    for q, Xq in parts.items():
        assert np.allclose(Sz @ Xq - Xq @ Sz, q * Xq)
    assert np.allclose(charge_projector(n, X, 1), parts[1])


def test_basis_terms_have_declared_charge():
    n = 4
    terms = charge_basis_terms(n, max_range=2)
    # 4 sites: 3 nearest-neighbour + 2 next-nearest-neighbour pairs
    assert len(terms) == 3 * n + 9 * 5
    for t in terms:
        assert np.allclose(t.op, t.op.conj().T), t.label
        support = {q for q, X in charge_decompose(n, t.op).items() if np.linalg.norm(X) > 1e-12}
        assert support == ({0} if t.abs_charge == 0 else {t.abs_charge, -t.abs_charge}), t.label
    counts = np.bincount([t.abs_charge for t in terms])
    assert list(counts) == [n + 3 * 5, 2 * n + 4 * 5, 2 * 5]


def test_xy_model_is_u1_covariant():
    n = 3
    rng = np.random.default_rng(1)
    G = rng.normal(size=(n, n))
    model = OpenModel(H=xy_hamiltonian(n, xy_couplings(n), B=rng.normal(size=n)),
                      Gamma=0.1 * G @ G.T, emission=0.1 * rng.random(n))
    X = random_complex(8, rng)
    for p in range(-n, n + 1):
        LXp = lindblad_rhs(model, charge_projector(n, X, p))
        for q in range(-n, n + 1):
            if q != p:
                assert np.linalg.norm(charge_projector(n, LXp, q)) < 1e-12


def test_protective_rate_formula():
    Gamma = 0.3 * np.ones((3, 3)) + np.diag([0.1, 0.2, 0.05])
    one = np.ones(3)
    g = protective_collective_rate(Gamma)
    assert np.isclose(g, 1.0 / (one @ np.linalg.solve(Gamma, one)))
    assert np.linalg.eigvalsh(Gamma - g * np.outer(one, one)).min() >= -1e-10


def test_protective_rate_singular():
    assert protective_collective_rate(np.diag([1.0, 0.0, 1.0])) == 0.0


def test_collective_damping_rate():
    n, gamma_c, t = 3, 0.4, 1.3
    model = OpenModel(H=np.zeros((8, 8), complex), Gamma=gamma_c * np.ones((n, n)), emission=np.zeros(n))
    X = random_complex(8, np.random.default_rng(2))
    for q in (1, 2, 3):
        Xq = charge_projector(n, X, q)
        assert np.allclose(evolve(model, Xq, [0.0, t])[-1], np.exp(-2 * gamma_c * q**2 * t) * Xq, atol=1e-9)


def test_sample_coeffs_antithetic():
    g = sample_coeffs(5, 0.1, 4, np.random.default_rng(3))
    assert g.shape == (8, 5)
    assert np.allclose(g[0::2], -g[1::2])
