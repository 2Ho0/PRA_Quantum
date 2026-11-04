import itertools

import numpy as np

from qerr.learning import Ansatz, adjoint_terms, fit, integral_system, learn, to_openmodel
from qerr.lindblad import adjoint_rhs, evolve
from qerr.pauli import comm_i, expectation_tables, expectations, pmatrix, pmul, psparse, weight


def all_strings(n, max_weight):
    out = []
    for w in range(1, max_weight + 1):
        for sites in itertools.combinations(range(n), w):
            for letters in itertools.product("XYZ", repeat=w):
                out.append(tuple(zip(sites, letters)))
    return out


def xy_ansatz(n):
    pairs = list(itertools.combinations(range(n), 2))
    h = [((i, a), (j, b)) for i, j in pairs for a, b in ("XX", "YY", "XY", "YX")]
    h += [((i, a),) for i in range(n) for a in "XYZ"]
    deph = [(i, i) for i in range(n)] + pairs
    return Ansatz(n=n, h_terms=h, deph_pairs=deph, emission_sites=list(range(n)))


def random_theta(ans, rng):
    theta = np.zeros(ans.n_params)
    for k, p in enumerate(ans.h_terms):
        if weight(p) == 2:
            (i, a), (j, b) = p
            theta[k] = (0.5 / abs(i - j) if a == b else 0.05) * (1 + 0.2 * rng.normal())
        else:
            theta[k] = 0.3 * rng.normal() if p[0][1] == "Z" else 0.03 * rng.normal()
    A = 0.1 * rng.normal(size=(ans.n, ans.n))
    G = A @ A.T + 0.04 * np.ones((ans.n, ans.n))
    k0 = len(ans.h_terms)
    for m, (i, j) in enumerate(ans.deph_pairs):
        theta[k0 + m] = G[i, j]
    theta[k0 + len(ans.deph_pairs):] = 0.01 + 0.01 * rng.random(len(ans.emission_sites))
    return theta


def product_state(bloch):
    rho = np.array([[1.0]])
    for x, y, z in bloch:
        rho = np.kron(rho, 0.5 * np.array([[1 + z, x - 1j * y], [x + 1j * y, 1 - z]]))
    return rho


def exact_series(model, n, bloch_states, times, strings):
    series = {}
    mats = {p: pmatrix(p, n) for p in strings}
    for s, bloch in enumerate(bloch_states):
        rhos = evolve(model, product_state(bloch), times)
        series[(s, ())] = np.ones(len(times))
        for p, P in mats.items():
            series[(s, p)] = np.einsum("ij,tji->t", P, rhos).real
    return series


def test_comm_i_single_site():
    assert comm_i(((0, "X"),), ((0, "Y"),)) == (-2.0, ((0, "Z"),))
    assert comm_i(((0, "Y"),), ((0, "Z"),)) == (-2.0, ((0, "X"),))
    assert comm_i(((0, "Z"),), ((0, "X"),)) == (-2.0, ((0, "Y"),))
    assert comm_i(((0, "X"),), ((1, "Y"),)) == (0.0, None)
    assert pmul(((0, "X"), (1, "Z")), ((0, "X"),)) == (1, ((1, "Z"),))


def test_adjoint_terms_match_superop():
    n = 3
    rng = np.random.default_rng(0)
    ans = xy_ansatz(n)
    theta = random_theta(ans, rng)
    model = to_openmodel(theta, ans)
    X = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    rho = X @ X.conj().T
    rho /= np.trace(rho)
    for O in all_strings(n, 2):
        direct = np.trace(adjoint_rhs(model, pmatrix(O, n)) @ rho)
        terms = sum(theta[k] * c * np.trace(pmatrix(S, n) @ rho) for k, c, S in adjoint_terms(O, ans))
        assert abs(direct - terms) < 1e-10, O


def recovery_error(times, t_learn):
    n = 4
    rng = np.random.default_rng(3)
    ans = xy_ansatz(n)
    theta = random_theta(ans, rng)
    bloch_states = [[(np.sin(1.04), 0.0, -np.cos(1.04))] * n, [(0.0, 1.0, 0.0)] * n]
    series = exact_series(to_openmodel(theta, ans), n, bloch_states, times, all_strings(n, 3))
    A, y, _ = integral_system(all_strings(n, 2), ans, series, times, t_learn, states=(0, 1))
    return np.abs(fit(A, y) - theta).max() / np.abs(theta).max()


def test_recovers_parameters_exact_data():
    assert recovery_error(np.linspace(0.0, 1.5, 151), 1.5) < 1e-3


def test_expectations_match_pmatrix():
    n = 3
    rng = np.random.default_rng(4)
    X = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    rho = X @ X.conj().T
    strings = all_strings(n, 3)
    idx, ph = expectation_tables(strings, n)
    direct = [np.trace(pmatrix(p, n) @ rho).real for p in strings]
    assert np.allclose(expectations(rho, idx, ph), direct)
    assert all(np.allclose(psparse(p, n).toarray(), pmatrix(p, n)) for p in strings)


def test_defect_correction_removes_coarse_grid_bias():
    n = 4
    rng = np.random.default_rng(3)
    ans = xy_ansatz(n)
    theta = random_theta(ans, rng)
    bloch_states = [[(np.sin(1.04), 0.0, -np.cos(1.04))] * n, [(0.0, 1.0, 0.0)] * n]
    times = np.array([0.0, 0.5, 1.0, 1.5])
    series = exact_series(to_openmodel(theta, ans), n, bloch_states, times, all_strings(n, 3))
    rho0s = [product_state(b) for b in bloch_states]
    thetas = learn(series, ans, all_strings(n, 2), times, 1.5, rho0s, n_iter=3)
    errs = [np.abs(t - theta).max() / np.abs(theta).max() for t in thetas]
    print(f"coarse grid (dt = 0.5): max relative parameter error per iteration {np.round(errs, 4)}")
    assert errs[0] > 0.05 and errs[-1] < 0.01
