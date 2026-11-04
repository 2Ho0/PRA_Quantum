import numpy as np

from qerr.bounds import (beqs_bias_bound, beqs_concentration_bound, beqs_expected_bound,
                         charged_observable_bound, forgetting_bound, opnorm, zeno_bound)
from qerr.lindblad import OpenModel, evolve, expect


def test_bounds_limits():
    t = np.array([0.0, 0.5, 2.0])
    for b in (beqs_expected_bound(t, 0.1, 10), beqs_concentration_bound(t, 0.1, 10),
              beqs_bias_bound(t, 1.0), charged_observable_bound(t, 1.0, 0.5, 1),
              zeno_bound(t, 1.0, {1: 0.3, -1: 0.3}, 0.5)):
        assert b[0] == 0.0 and np.all(np.isfinite(b))
    # gamma_c -> 0 recovers the closed-system forms
    assert np.allclose(charged_observable_bound(t, 1.0, 1e-14, 1), 2 * t, rtol=1e-6)
    assert np.allclose(zeno_bound(t, 1.0, {1: 0.3, -1: 0.3}, 1e-9), 2 * 0.6 * t**2, rtol=1e-3)
    # large gamma_c t: linear growth with slope 2 ||dH|| sum_q ||dH_q|| / (gamma_c q^2)
    big = zeno_bound(np.array([100.0]), 1.0, {2: 0.5, -2: 0.5}, 10.0)
    assert np.isclose(big[0], 2 * 100 * 1.0 / (10 * 4), rtol=0.01)


def test_forgetting_bound_dominates():
    rng = np.random.default_rng(5)
    n, D = 3, 8
    A = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    B = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    G = rng.normal(size=(n, n))
    H, dH = 0.3 * (A + A.conj().T), 0.05 * (B + B.conj().T)
    model = OpenModel(H=H, Gamma=0.2 * G @ G.T, emission=0.3 * np.ones(n))
    perturbed = OpenModel(H=H + dH, Gamma=model.Gamma, emission=model.emission)
    psi = rng.normal(size=D) + 1j * rng.normal(size=D)
    rho0 = np.outer(psi, psi.conj()) / np.vdot(psi, psi)
    O = np.diag(rng.normal(size=D)).astype(complex)
    times = np.linspace(0, 6, 25)
    delta = np.abs(expect(evolve(perturbed, rho0, times), O) - expect(evolve(model, rho0, times), O))
    fb = forgetting_bound(model, O, dH, times)
    assert np.all(delta <= fb + 1e-9)
    assert np.all(fb <= beqs_bias_bound(times, opnorm(dH), opnorm(O)) + 1e-9)
