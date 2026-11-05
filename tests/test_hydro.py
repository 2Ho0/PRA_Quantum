import numpy as np

from qerr.fermions import evolve_covariance, product_covariance
from qerr.quadratic import bdg_to_majorana, heisenberg_evolve, number_operator


def chain(N, amp):
    h = np.zeros((N, N), complex)
    for i in range(N - 1):
        h[i, i + 1] = h[i + 1, i] = amp
    return h


def test_duhamel_pairing_reproduces_the_error():
    """Delta(t) = int_0^t <i[dH, O_u]>_{rho'_{t-u}} du with O_u the reference and rho' the perturbed evolution."""
    from qerr.hydro import coefficient_matrix, commutator_coefficients, pairing, two_point

    N, g, eps, c = 24, 0.7, 0.05, 11
    rng = np.random.default_rng(0)
    occ = rng.random(N)
    t_max, n_t = 4.0, 801
    grid = np.linspace(0.0, t_max, n_t)
    A_u = heisenberg_evolve(chain(N, -1.0), np.zeros((N, N)), g * np.ones(N), 0.0,
                            bdg_to_majorana(number_operator(N, c)).real, grid, rtol=1e-11, atol=1e-13)
    G1 = evolve_covariance(chain(N, -1.0 + eps), np.zeros((N, N)), g * np.ones(N), 0.0, product_covariance(occ),
                           grid, rtol=1e-11, atol=1e-13)
    G0 = evolve_covariance(chain(N, -1.0), np.zeros((N, N)), g * np.ones(N), 0.0, product_covariance(occ),
                           grid[[0, -1]], rtol=1e-11, atol=1e-13)
    dh = chain(N, eps)
    integrand = np.array([pairing(commutator_coefficients(dh, coefficient_matrix(A_u[k])), two_point(G1[n_t - 1 - k]))
                          for k in range(n_t)])
    assert np.abs(integrand.imag).max() < 1e-12
    via = np.trapezoid(integrand.real, grid)
    direct = (two_point(G1[-1]) - two_point(G0[-1]))[c, c].real
    assert abs(via - direct) < 1e-6 * abs(direct) + 1e-12


def test_two_sided_factors_bound_the_pairing():
    from qerr.hydro import commutator_coefficients, observable_factors, pairing, state_factors

    N = 30
    rng = np.random.default_rng(1)
    for _ in range(20):
        X = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
        w = X + X.conj().T
        Y = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
        C = 0.1 * (Y + Y.conj().T) + np.diag(rng.random(N))
        dh = chain(N, 1.0) * rng.random(N)[:, None]
        dh = 0.5 * (dh + dh.T)  # random real symmetric bond amplitudes
        a, m = observable_factors(w, dh)
        gamma, kappa = state_factors(C)
        assert abs(pairing(commutator_coefficients(dh, w), C)) <= a * gamma + m * kappa + 1e-12
