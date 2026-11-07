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


# --- single-site Kossakowski parameters, structured and PSD-constrained fits -------------------------------

PAULI1 = {"X": np.array([[0, 1], [1, 0]], complex), "Y": np.array([[0, -1j], [1j, 0]], complex),
          "Z": np.array([[1, 0], [0, -1]], complex)}


def site_op(n, site, a):
    return pmatrix(((site, a),), n)


def random_site_noise(rng, scale=0.05):
    """Random PSD 3x3 Kossakowski matrix over (X, Y, Z)."""
    F = scale * (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    return F @ F.conj().T


def jumps_from(n, site, K):
    w, V = np.linalg.eigh(K)
    ops = [site_op(n, site, a) for a in "XYZ"]
    return [np.sqrt(max(lam, 0.0)) * sum(v[a] * ops[a] for a in range(3)) for lam, v in zip(w, V.T)]


def noise_theta(K):
    return [K[0, 0].real, K[1, 1].real, K[0, 1].real, K[0, 1].imag, K[0, 2].real, K[0, 2].imag,
            K[1, 2].real, K[1, 2].imag]


def noisy_model(n, rng):
    """Dense model with XY Hamiltonian, Z dephasing Gamma and site noise K_e; theta in the noise-site ansatz."""
    from qerr.lindblad import OpenModel

    base = xy_ansatz(n)
    ans = Ansatz(n=n, h_terms=base.h_terms, deph_pairs=base.deph_pairs, emission_sites=[],
                 noise_sites=list(range(n)))
    th0 = random_theta(base, rng)
    k0 = len(base.h_terms)
    H = sum(th0[k] * pmatrix(P, n) for k, P in enumerate(base.h_terms))
    G = np.zeros((n, n))
    for m, (i, j) in enumerate(base.deph_pairs):
        G[i, j] = G[j, i] = th0[k0 + m]
    Ks = [random_site_noise(rng) for _ in range(n)]
    jumps = [J for e, K in enumerate(Ks) for J in jumps_from(n, e, K)]
    model = OpenModel(H=H, Gamma=G, emission=np.zeros(n), jumps=tuple(jumps))
    theta = np.concatenate([th0[:k0], np.zeros(len(base.deph_pairs)), np.concatenate([noise_theta(K) for K in Ks])])
    for m, (i, j) in enumerate(base.deph_pairs):
        theta[k0 + m] = G[i, j] + (Ks[i][2, 2].real if i == j else 0.0)
    return model, ans, theta


def random_state(D, rng):
    X = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    rho = X @ X.conj().T
    return rho / np.trace(rho)


def test_site_noise_adjoint_matches_superop():
    from qerr.lindblad import lindblad_rhs

    n = 2
    rng = np.random.default_rng(7)
    model, ans, theta = noisy_model(n, rng)
    rho = random_state(4, rng)
    Lrho = lindblad_rhs(model, rho)
    for O in all_strings(n, 2):
        direct = np.trace(pmatrix(O, n) @ Lrho)
        terms = sum(theta[k] * c * (np.trace(pmatrix(S, n) @ rho) if S != () else 1.0)
                    for k, c, S in adjoint_terms(O, ans))
        assert abs(direct - terms) < 1e-10, O


def test_correlated_adjoint_matches_collective_jump():
    """Single-ion block S = c plus the cross-ion block c is the collective dissipator D[sum_k (a X_k + b Y_k)]."""
    from qerr.learning import NOISE_KEYS, correlated_adjoint
    from qerr.lindblad import OpenModel, lindblad_rhs

    n, a, b = 3, 0.6, -0.8
    rng = np.random.default_rng(11)
    D = 2**n
    J = sum(a * pmatrix(((k, "X"),), n) + b * pmatrix(((k, "Y"),), n) for k in range(n))
    model = OpenModel(H=np.zeros((D, D)), Gamma=np.zeros((n, n)), emission=np.zeros(n), jumps=(J,))
    rho = random_state(D, rng)
    Lrho = lindblad_rhs(model, rho)
    ans = Ansatz(n=n, h_terms=[], deph_pairs=[], noise_sites=list(range(n)))
    theta = np.zeros(ans.n_params)
    for m in range(n):
        theta[len(NOISE_KEYS) * m:len(NOISE_KEYS) * m + 3] = [a * a, b * b, a * b]  # SXX, SYY, SXY
    c = {"cXX": a * a, "cYY": b * b, "cXY": a * b}
    expect = lambda S: np.trace(pmatrix(S, n) @ rho) if S != () else 1.0
    for O in all_strings(n, 3):
        direct = np.trace(pmatrix(O, n) @ Lrho)
        single = sum(theta[k] * w * expect(S) for k, w, S in adjoint_terms(O, ans))
        cross = sum(c[key] * w * expect(S) for key in c for S, w in correlated_adjoint(O, key).items())
        assert abs(direct - single - cross) < 1e-12, O


def test_site_noise_reproduces_emission():
    n, gam = 2, 0.37
    rng = np.random.default_rng(8)
    em = Ansatz(n=n, h_terms=[], deph_pairs=[], emission_sites=[0])
    no = Ansatz(n=n, h_terms=[], deph_pairs=[], noise_sites=[0])
    th_no = np.zeros(no.n_params)
    th_no[:4] = [gam / 4, gam / 4, 0.0, gam / 4]
    rho = random_state(4, rng)
    expect = lambda S: np.trace(pmatrix(S, n) @ rho) if S != () else 1.0
    for O in all_strings(n, 2):
        a = sum(gam * c * expect(S) for k, c, S in adjoint_terms(O, em))
        b = sum(th_no[k] * c * expect(S) for k, c, S in adjoint_terms(O, no))
        assert abs(a - b) < 1e-12, O


def test_noise_charges_match_superop():
    from qerr.charge import superop_charges
    from qerr.learning import NOISE_KEYS, noise_charges

    n = 1
    I = np.eye(2)

    def superop_of(K):
        S = np.zeros((4, 4), complex)
        for a, A in enumerate("XYZ"):
            for b, B in enumerate("XYZ"):
                PA, PB = PAULI1[A], PAULI1[B]
                BA = PB @ PA
                S += K[a, b] * (np.kron(PB.T, PA) - 0.5 * np.kron(I, BA) - 0.5 * np.kron(BA.T, I))
        return S

    combos = noise_charges()
    for combo, charges in combos.items():
        K = np.zeros((3, 3), complex)
        for key, sign in combo:
            kind, A, B = key[0], "XYZ".index(key[1]), "XYZ".index(key[2])
            if kind == "S":
                K[A, B] += sign
                if A != B:
                    K[B, A] += sign
            else:
                K[A, B] += 1j * sign
                K[B, A] -= 1j * sign
        assert set(superop_charges(n, superop_of(K))) == charges, combo
    used = {key for combo in combos for key, _ in combo}
    assert used == set(NOISE_KEYS)


def test_site_noise_ansatz_full_rank():
    n = 2
    rng = np.random.default_rng(9)
    model, ans, theta = noisy_model(n, rng)
    times = np.linspace(0.0, 1.5, 151)
    bloch_states = [[(np.sin(1.04), 0.0, -np.cos(1.04))] * n, [(0.0, 1.0, 0.0)] * n]
    series = exact_series(model, n, bloch_states, times, all_strings(n, 2))
    A, y, _ = integral_system(all_strings(n, 2), ans, series, times, 1.5)
    assert np.linalg.matrix_rank(A) == ans.n_params
    est = fit(A, y)
    assert np.abs(est - theta).max() / np.abs(theta).max() < 1e-3


def test_structured_fit_recovers_gamma_c():
    from qerr.learning import fit_bounded, gamma_matrix, structured_map, to_openmodel

    n = 3
    rng = np.random.default_rng(10)
    ans = xy_ansatz(n)
    theta = random_theta(ans, rng)
    gc, g = 0.3, np.array([0.1, 0.2, 0.05])
    G = gc * np.ones((n, n)) + np.diag(g)
    k0 = len(ans.h_terms)
    for m, (i, j) in enumerate(ans.deph_pairs):
        theta[k0 + m] = G[i, j]
    times = np.linspace(0.0, 1.5, 151)
    bloch_states = [[(np.sin(1.04), 0.0, -np.cos(1.04))] * n, [(0.0, 1.0, 0.0)] * n]
    series = exact_series(to_openmodel(theta, ans), n, bloch_states, times, all_strings(n, 3))
    A, y, _ = integral_system(all_strings(n, 2), ans, series, times, 1.5)
    P = structured_map(ans)
    lower = np.full(P.shape[1], -np.inf)
    lower[k0 + 1:k0 + 1 + n] = 0.0
    th_s = fit_bounded(A @ P, y, lower)
    assert abs(th_s[k0] - gc) < 1e-3
    assert np.abs(th_s[k0 + 1:k0 + 1 + n] - g).max() < 1e-3
    G_back = gamma_matrix(P @ th_s, ans)
    assert np.allclose(G_back, th_s[k0] * np.ones((n, n)) + np.diag(th_s[k0 + 1:k0 + 1 + n]))


def test_psd_fit_is_feasible_and_optimal():
    from qerr.learning import fit_psd, gamma_matrix

    n = 3
    rng = np.random.default_rng(11)
    ans = xy_ansatz(n)
    A = rng.normal(size=(400, ans.n_params))
    theta = random_theta(ans, rng)
    est = fit_psd(A, A @ theta, ans)
    assert np.abs(est - theta).max() < 1e-6
    k0, nd = len(ans.h_terms), len(ans.deph_pairs)
    bad = theta.copy()
    bad[k0] -= 0.3  # one negative eigenvalue: clipping it is feasible but not optimal
    y = A @ bad + 0.01 * rng.normal(size=A.shape[0])
    est = fit_psd(A, y, ans)
    assert np.linalg.eigvalsh(gamma_matrix(est, ans)).min() > -1e-9
    ls = fit(A, y)
    w, V = np.linalg.eigh(gamma_matrix(ls, ans))
    Gc = (V * np.clip(w, 0, None)) @ V.T
    clip = ls.copy()
    for m, (i, j) in enumerate(ans.deph_pairs):
        clip[k0 + m] = Gc[i, j]
    other = np.r_[0:k0, k0 + nd:ans.n_params]
    clip[other] = np.linalg.lstsq(A[:, other], y - A[:, k0:k0 + nd] @ clip[k0:k0 + nd], rcond=None)[0]
    obj = lambda th: np.sum((A @ th - y) ** 2)
    assert obj(est) < obj(clip) - 1e-3 * obj(clip)
    # KKT for min f over Gamma >= 0: the gradient Z is PSD, complementary to Gamma*, and zero for free parameters
    grad = 2 * A.T @ (A @ est - y)
    Z = np.zeros((n, n))
    for m, (i, j) in enumerate(ans.deph_pairs):
        Z[i, j] = Z[j, i] = grad[k0 + m] if i == j else 0.5 * grad[k0 + m]
    scale = np.abs(grad).max() + np.abs(2 * A.T @ y).max()
    assert np.abs(grad[other]).max() < 1e-8 * scale
    assert np.linalg.eigvalsh(Z).min() > -1e-6 * scale
    assert abs(np.sum(Z * gamma_matrix(est, ans))) < 1e-6 * scale


def test_fit_psd_rejects_rank_deficient_profile():
    """Collinear non-dephasing columns (e.g. emission and noise sites together) would corrupt the profiling."""
    import pytest
    from qerr.learning import fit_psd

    n = 3
    rng = np.random.default_rng(12)
    ans = xy_ansatz(n)
    A = rng.normal(size=(200, ans.n_params))
    A[:, 1] = A[:, 0]  # two Hamiltonian columns made identical
    with pytest.raises(ValueError, match="rank"):
        fit_psd(A, A @ random_theta(ans, rng), ans)


# --- learning windows that start after t = 0 (time-resolved generators) ------------------------------------

def late_window_problem(times):
    n = 4
    rng = np.random.default_rng(3)
    ans = xy_ansatz(n)
    theta = random_theta(ans, rng)
    bloch_states = [[(np.sin(1.04), 0.0, -np.cos(1.04))] * n, [(0.0, 1.0, 0.0)] * n]
    series = exact_series(to_openmodel(theta, ans), n, bloch_states, times, all_strings(n, 3))
    return ans, theta, series, [product_state(b) for b in bloch_states]


def test_window_fit_recovers_parameters_on_late_window():
    ans, theta, series, _ = late_window_problem(np.linspace(0.0, 3.0, 301))
    A, y, rows = integral_system(all_strings(4, 2), ans, series, np.linspace(0.0, 3.0, 301), 3.0, t_start=1.5)
    assert min(t for _, _, t in rows) > 1.5 + 1e-9
    assert np.abs(fit(A, y) - theta).max() / np.abs(theta).max() < 1e-3


def test_window_defect_correction_on_late_window():
    times = np.arange(0.0, 3.01, 0.5)
    ans, theta, series, rho0s = late_window_problem(times)
    thetas = learn(series, ans, all_strings(4, 2), times, 3.0, rho0s, n_iter=3, t_start=1.0)
    errs = [np.abs(t - theta).max() / np.abs(theta).max() for t in thetas]
    print(f"late window [1, 3], dt = 0.5: max relative parameter error per iteration {np.round(errs, 4)}")
    assert errs[0] > 0.05 and errs[-1] < 0.01


def test_unit_weight_reproduces_the_unweighted_system():
    times = np.arange(0.0, 3.01, 0.5)
    ans, _, series, _ = late_window_problem(times)
    A, y, rows = integral_system(all_strings(4, 2), ans, series, times, 3.0, t_start=1.0)
    Aw, yw, rows_w = integral_system(all_strings(4, 2), ans, series, times, 3.0, t_start=1.0,
                                     weight=lambda s: np.ones_like(s))
    assert rows_w == rows and np.array_equal(yw, y)
    assert np.abs(Aw - A).max() < 1e-12


def test_time_weighted_column_recovers_a_growing_collective_rate():
    """gamma_c(t) = a + b t (white plus quasi-static noise) is linear in (a, b): weight the collective column by t."""
    from qerr.lindblad import evolve_td

    n = 3
    rng = np.random.default_rng(8)
    ans = xy_ansatz(n)
    theta = random_theta(ans, rng)
    a, b = 0.2, 0.4
    times = np.linspace(0.0, 1.5, 151)
    model = to_openmodel(theta, ans)
    mats = {p: pmatrix(p, n) for p in all_strings(n, 3)}
    series = {}
    for s, bloch in enumerate([[(np.sin(1.04), 0.0, -np.cos(1.04))] * n, [(0.0, 1.0, 0.0)] * n]):
        rhos = evolve_td(model, product_state(bloch), times, np.ones((n, n)), lambda t: a + b * t)
        for p, P in mats.items():
            series[(s, p)] = np.einsum("ij,tji->t", P, rhos).real
    obs = all_strings(n, 2)
    A, y, _ = integral_system(obs, ans, series, times, 1.5)
    Aw, _, _ = integral_system(obs, ans, series, times, 1.5, weight=lambda s: s)
    k0, nd = len(ans.h_terms), len(ans.deph_pairs)
    est = fit(np.column_stack([A, Aw[:, k0:k0 + nd].sum(axis=1)]), y)
    truth = theta.copy()
    truth[k0:k0 + nd] += a  # the constant part of the collective rate adds a to every Gamma_ij
    assert abs(est[-1] - b) < 1e-3
    assert np.abs(est[:-1] - truth).max() / np.abs(truth).max() < 1e-3


def test_trajectory_defect_matches_quadrature_defect():
    from qerr.learning import model_trajectory, needed_strings, quadrature_defect, refine, trajectory_defect

    n = 3
    ans = xy_ansatz(n)
    model = to_openmodel(random_theta(ans, np.random.default_rng(5)), ans)
    rho0s = [product_state([(0.0, 1.0, 0.0)] * n)]
    strings = needed_strings(all_strings(n, 2), ans)
    times = np.arange(0.0, 3.01, 0.5)
    direct = quadrature_defect(model, rho0s, strings, times, 2.5, n_sub=8, t_start=1.0)
    fine = refine(times, 8)
    via = trajectory_defect(fine, model_trajectory(model, rho0s, strings, fine), strings, times, 2.5, t_start=1.0)
    assert set(via) == set(direct)
    assert max(np.abs(direct[k] - via[k]).max() for k in direct) < 1e-7
    unit = trajectory_defect(fine, model_trajectory(model, rho0s, strings, fine), strings, times, 2.5, t_start=1.0,
                             weight=lambda s: np.ones_like(s))
    assert max(np.abs(unit[k] - via[k]).max() for k in via) < 1e-12
