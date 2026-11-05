import numpy as np

from qerr import spins
from qerr.lindblad import OpenModel, evolve, lindblad_rhs, superop


def random_model(n, rng, n_extra_jumps=1):
    D = 2**n
    A = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    G = rng.normal(size=(n, n))
    jumps = tuple(
        0.3 * (rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)))
        for _ in range(n_extra_jumps)
    )
    return OpenModel(
        H=(A + A.conj().T) / 2,
        Gamma=0.1 * (G @ G.T),
        emission=0.1 * rng.random(n),
        jumps=jumps,
    )


def random_state(D, rng):
    A = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


def random_hermitian(D, rng):
    A = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    return A + A.conj().T


def test_pauli_algebra():
    n = 3
    for site in range(n):
        x, y, z = (spins.pauli_op(n, site, k) for k in "xyz")
        assert np.allclose(x @ y, 1j * z)
    ket0, ket1 = np.array([1, 0]), np.array([0, 1])
    assert np.allclose(spins.pauli_op(1, 0, "+"), np.outer(ket0, ket1))
    assert np.allclose(spins.pauli_op(1, 0, "-"), np.outer(ket1, ket0))
    assert np.allclose(spins.pauli_op(1, 0, "z") @ ket0, ket0)
    # site 0 is the leftmost Kronecker factor
    assert np.allclose(spins.basis_state(2, "10"), np.diag([0, 0, 1, 0]))


def test_rhs_matches_superop():
    rng = np.random.default_rng(1)
    model = random_model(3, rng)
    rho = random_state(8, rng)
    lhs = lindblad_rhs(model, rho).reshape(-1, order="F")
    rhs = superop(model) @ rho.reshape(-1, order="F")
    assert np.linalg.norm(lhs - rhs) < 1e-12


def test_trace_and_hermiticity_preserved():
    rng = np.random.default_rng(2)
    model = random_model(3, rng)
    rhos = evolve(model, random_state(8, rng), np.linspace(0, 2.0, 5))
    for r in rhos:
        assert abs(np.trace(r) - 1) < 1e-9
        assert np.linalg.norm(r - r.conj().T) < 1e-9
        assert np.linalg.eigvalsh((r + r.conj().T) / 2).min() > -1e-8


def test_heisenberg_matches_schrodinger():
    rng = np.random.default_rng(3)
    model = random_model(3, rng)
    rho0 = random_state(8, rng)
    O = random_hermitian(8, rng)
    times = np.linspace(0, 1.5, 4)
    rhos = evolve(model, rho0, times)
    Os = evolve(model, O, times, heisenberg=True)
    for r, Ot in zip(rhos, Os):
        assert abs(np.trace(O @ r) - np.trace(Ot @ rho0)) < 1e-8


def test_evolve_nonhermitian_linear():
    rng = np.random.default_rng(4)
    model = random_model(2, rng)
    X, Y = rng.normal(size=(4, 4)), rng.normal(size=(4, 4))
    t = np.array([0.0, 0.7])
    a = evolve(model, X + 1j * Y, t)
    b = evolve(model, X, t) + 1j * evolve(model, Y, t)
    assert np.max(np.abs(a - b)) < 1e-9


def test_evolve_releases_solver_memory():
    # solve_ivp keeps its stage arrays in reference cycles; repeated short calls on large states must not
    # accumulate them (observed: +300 MB per call for 10 qubits before the fix)
    import tracemalloc

    n = 8
    rng = np.random.default_rng(0)
    A = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))
    model = OpenModel(H=(A + A.conj().T) / 50, Gamma=0.1 * np.eye(n), emission=0.05 * np.ones(n))
    rho = np.eye(2**n, dtype=complex) / 2**n
    tracemalloc.start()
    try:
        rho = evolve(model, rho, [0.0, 0.05], rtol=1e-7, atol=1e-9)[-1]
        first = tracemalloc.get_traced_memory()[0]
        for _ in range(4):
            rho = evolve(model, rho, [0.0, 0.05], rtol=1e-7, atol=1e-9)[-1]
        growth = tracemalloc.get_traced_memory()[0] - first
    finally:
        tracemalloc.stop()
    assert growth < 5 * 2**20


def test_evolve_td_frees_solver_cycles():
    """Large evolutions collect the solver's reference cycles, which hold its stage arrays (~300 MB per call at n = 10)."""
    import gc

    from qerr.lindblad import evolve_td

    n = 7
    model = OpenModel(H=np.diag(np.linspace(0.0, 1.0, 2**n)).astype(complex), Gamma=0.1 * np.eye(n), emission=np.zeros(n))
    gc.collect()
    evolve_td(model, np.eye(2**n, dtype=complex) / 2**n, [0.0, 0.01], np.ones((n, n)), lambda t: 0.1)
    assert gc.collect() == 0
