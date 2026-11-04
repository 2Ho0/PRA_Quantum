import numpy as np

from qerr.lindblad import OpenModel, evolve
from qerr.quadratic import (bdg_operator, bdg_to_majorana, charge_part, heisenberg_evolve, majorana_to_bdg,
                            number_operator, quad_norm)
from test_fermions import jw_annihilators

N = 3


def ladder(n):
    c = jw_annihilators(n)
    return c, [x.conj().T for x in c]


def mb_quadratic(M, n):
    """Many-body sum_ab M_ab Psi_a Psi_b^dag with Psi = (c, c^dag)."""
    c, cd = ladder(n)
    psi = c + cd
    return sum(M[a, b] * psi[a] @ psi[b].conj().T for a in range(2 * n) for b in range(2 * n))


def mb_majorana(A, n):
    """Many-body q(A) = (i/4) sum_cd A_cd w_c w_d with c_j = (w_j + i w_{n+j}) / 2."""
    c, cd = ladder(n)
    w = [c[j] + cd[j] for j in range(n)] + [1j * (cd[j] - c[j]) for j in range(n)]
    return 0.25j * sum(A[x, y] * w[x] @ w[y] for x in range(2 * n) for y in range(2 * n))


def random_model(rng, n):
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    B = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return 0.5 * (A + A.conj().T), 0.3 * (B - B.T)


def random_antisym(rng, n, complex_=False):
    X = rng.normal(size=(2 * n, 2 * n))
    if complex_:
        X = X + 1j * rng.normal(size=(2 * n, 2 * n))
    return X - X.T


def traceless(X):
    return X - np.trace(X) / X.shape[0] * np.eye(X.shape[0])


def test_bdg_operator_matches_many_body():
    rng = np.random.default_rng(1)
    h, Delta = random_model(rng, N)
    c, cd = ladder(N)
    H = sum(h[i, j] * cd[i] @ c[j] for i in range(N) for j in range(N))
    pair = sum(Delta[i, j] * cd[i] @ cd[j] for i in range(N) for j in range(N))
    H = H + 0.5 * (pair + pair.conj().T)
    assert np.allclose(mb_quadratic(bdg_operator(h, Delta), N), H)
    for i in range(N):
        assert np.allclose(mb_quadratic(number_operator(N, i), N), cd[i] @ c[i])


def test_majorana_roundtrip_operator():
    rng = np.random.default_rng(2)
    M = rng.normal(size=(2 * N, 2 * N)) + 1j * rng.normal(size=(2 * N, 2 * N))
    A = bdg_to_majorana(M)
    assert np.allclose(A, -A.T)
    assert np.allclose(mb_majorana(A, N) + 0.5 * np.trace(M) * np.eye(2**N), mb_quadratic(M, N))
    assert np.allclose(mb_quadratic(majorana_to_bdg(A), N), mb_majorana(A, N))


def test_commutator_rule():
    rng = np.random.default_rng(3)
    A1, A2 = random_antisym(rng, N, True), random_antisym(rng, N, True)
    X1, X2 = mb_majorana(A1, N), mb_majorana(A2, N)
    assert np.allclose(X1 @ X2 - X2 @ X1, 1j * mb_majorana(A1 @ A2 - A2 @ A1, N))


def test_quad_norm_exact_for_hermitian():
    rng = np.random.default_rng(4)
    for _ in range(5):
        A = random_antisym(rng, N)
        X = mb_majorana(A, N)
        assert np.allclose(X, X.conj().T)
        assert abs(quad_norm(A) - np.abs(np.linalg.eigvalsh(X)).max()) < 1e-10
        assert abs(quad_norm(1j * A) - np.abs(np.linalg.eigvalsh(X)).max()) < 1e-10


def test_heisenberg_duality():
    rng = np.random.default_rng(5)
    h, Delta = random_model(rng, N)
    gamma_loc, gamma_c, t = np.array([0.1, 0.2, 0.3]), 0.4, 0.7
    c, cd = ladder(N)
    H = sum(h[i, j] * cd[i] @ c[j] for i in range(N) for j in range(N))
    pair = sum(Delta[i, j] * cd[i] @ cd[j] for i in range(N) for j in range(N))
    H = H + 0.5 * (pair + pair.conj().T)
    n_ops = [cd[i] @ c[i] for i in range(N)]
    jumps = tuple(np.sqrt(g) * n for g, n in zip(gamma_loc, n_ops)) + (np.sqrt(gamma_c) * sum(n_ops),)
    model = OpenModel(H=H, Gamma=np.zeros((N, N)), emission=np.zeros(N), jumps=jumps)
    for A0 in (bdg_to_majorana(number_operator(N, 1)), random_antisym(rng, N), random_antisym(rng, N, True)):
        X_t = evolve(model, mb_majorana(A0, N), [0.0, t], heisenberg=True)[-1]
        A_t = heisenberg_evolve(h, Delta, gamma_loc, gamma_c, A0, [0.0, t])[-1]
        assert np.allclose(traceless(X_t), traceless(mb_majorana(A_t, N)), atol=1e-8)


def test_charge_part_selects_pairing():
    rng = np.random.default_rng(6)
    A = random_antisym(rng, N)
    c, cd = ladder(N)
    Ntot = sum(cd[i] @ c[i] for i in range(N))
    for q in (0, 2, -2):
        X = mb_majorana(charge_part(A, q), N)
        assert np.allclose(Ntot @ X - X @ Ntot, q * X)
    assert np.allclose(charge_part(A, 0) + charge_part(A, 2) + charge_part(A, -2), A)
