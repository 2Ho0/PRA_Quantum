import numpy as np

from qerr import spins
from qerr.charge import charge_projector
from qerr.lindblad import OpenModel, evolve
from qerr.models import xy_couplings, xy_hamiltonian


def small_model(n=4, gamma_loc=0.05, seed=1):
    rng = np.random.default_rng(seed)
    H = xy_hamiltonian(n, xy_couplings(n, 1.0, 1.0), B=0.3 * rng.normal(size=n))
    return OpenModel(H=H, Gamma=gamma_loc * np.eye(n), emission=np.zeros(n))


def random_density(D, rng):
    X = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    rho = X @ X.conj().T
    return rho / np.trace(rho)


def telegraph_chi(q, a, lam, t):
    """Closed form for symmetric telegraph noise +-a with switching rate lam in each direction."""
    om = np.sqrt(complex(lam**2 - (q * a) ** 2))
    return (np.exp(-lam * t) * (np.cosh(om * t) + lam / om * np.sinh(om * t))).real


def test_chain_chi_matches_closed_forms():
    from qerr.noise import ehrenfest, quasi_static, telegraph

    t = np.linspace(0.0, 4.0, 9)
    for q in (1, 2, 3):
        assert np.allclose(telegraph(0.7, 0.4).chi(q, t), telegraph_chi(q, 0.7, 0.4, t), atol=1e-12)
        assert np.allclose(ehrenfest(0.8, 1.5, 12).chi(q, t), telegraph_chi(q, 0.8 / np.sqrt(12), 1 / 3.0, t) ** 12,
                           atol=1e-12)
        assert np.allclose(quasi_static(0.5, 80).chi(q, t), np.exp(-0.5 * (q * 0.5 * t) ** 2), atol=1e-10)


def test_ehrenfest_correlation_is_ou():
    from scipy.linalg import expm

    from qerr.noise import ehrenfest

    noise = ehrenfest(0.8, 1.5, 12)
    assert np.allclose(noise.W.sum(axis=0), 0.0) and np.allclose(noise.W @ noise.pi, 0.0)
    for tau in (0.0, 0.7, 2.0):
        corr = noise.xi @ expm(noise.W * tau) @ (noise.xi * noise.pi)
        assert abs(corr - 0.8**2 * np.exp(-tau / 1.5)) < 1e-10


def test_averaged_evolution_is_sector_damping():
    """Lemma G.0: for a covariant model the noise multiplies each charge sector by chi_q(t), exactly."""
    from qerr.noise import averaged_evolve, ehrenfest, quasi_static, telegraph

    n = 4
    model = small_model(n)
    rho0 = random_density(2**n, np.random.default_rng(2))
    times = np.array([0.0, 0.5, 1.3, 2.0])
    free = evolve(model, rho0, times)
    for noise in (telegraph(0.9, 0.3), ehrenfest(0.6, 0.8, 6), quasi_static(0.7, 40)):
        avg = averaged_evolve(model, rho0, times, noise)
        for m, t in enumerate(times):
            pred = sum(noise.chi(q, t) * charge_projector(n, free[m], q) for q in range(-n, n + 1))
            assert np.abs(avg[m] - pred).max() < 1e-9, (noise, t)


def test_fast_ehrenfest_noise_approaches_lindblad_collective_dephasing():
    from qerr.noise import averaged_evolve, ehrenfest

    n = 3
    model = small_model(n, seed=4)
    rho0 = random_density(2**n, np.random.default_rng(5))
    perturbed = OpenModel(H=model.H + 0.2 * sum(spins.pauli_op(n, i, "x") for i in range(n)), Gamma=model.Gamma,
                          emission=model.emission)
    gamma_c, tau_c = 0.3, 0.01  # sigma^2 tau_c = 2 gamma_c
    times = np.linspace(0.0, 2.0, 5)
    avg = averaged_evolve(perturbed, rho0, times, ehrenfest(np.sqrt(2 * gamma_c / tau_c), tau_c, 30))
    markov = evolve(OpenModel(H=perturbed.H, Gamma=model.Gamma + gamma_c * np.ones((n, n)),
                              emission=model.emission), rho0, times)
    assert np.abs(avg - markov).max() < 0.02
