"""Error bounds: BEQS (Kraft et al. 2026, Eqs. 11, 12, 14) and the theory note's Theorems A, B2, B3.

For the tight versions of Theorems A/B3 pass osc(O) as O_norm; ||O|| is always valid.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from .lindblad import OpenModel, evolve


def opnorm(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, 2))


def osc(A: np.ndarray) -> float:
    """min_c ||A - c 1|| for Hermitian A: half the spectral spread."""
    ev = np.linalg.eigvalsh((A + A.conj().T) / 2)
    return float((ev[-1] - ev[0]) / 2)


def beqs_expected_bound(t, sigma, M, O_norm=1.0):
    return np.expm1(2 * np.asarray(t, float) ** 2 * sigma**2 * M) * O_norm


def beqs_concentration_bound(t, sigma, M, delta=0.05, O_norm=1.0):
    c = np.sqrt(2 * np.log(2 / delta))
    return 2 * np.asarray(t, float) * np.sqrt(M) * c * sigma * O_norm


def beqs_bias_bound(t, dH_norm, O_norm=1.0):
    return 2 * np.asarray(t, float) * dH_norm * O_norm


def forgetting_bound(model: OpenModel, O: np.ndarray, dH: np.ndarray, times, n_fine: int = 401) -> np.ndarray:
    """Theorem A: int_0^t ||[dH, e^{u L^dag}(O)]|| du on a fine trapezoid grid."""
    times = np.asarray(times, float)
    fine = np.linspace(0.0, times[-1], n_fine)
    Os = evolve(model, O, fine, heisenberg=True)
    integrand = np.array([opnorm(dH @ Ou - Ou @ dH) for Ou in Os])
    return np.interp(times, fine, cumulative_trapezoid(integrand, fine, initial=0.0))


def _relax(a, t):
    """(1 - e^{-a t}) / a, finite as a -> 0."""
    a, t = np.broadcast_arrays(np.asarray(a, float), np.asarray(t, float))
    small = a * t < 1e-8
    return np.where(small, t * (1 - a * t / 2), -np.expm1(-a * t) / np.where(small, 1.0, a))


def _phi(a, t):
    """t/a - (1 - e^{-a t})/a^2 = int_0^t ds int_0^s du e^{-a(s-u)}, finite as a -> 0."""
    a, t = np.broadcast_arrays(np.asarray(a, float), np.asarray(t, float))
    x = a * t
    small = x < 1e-4
    safe_a = np.where(small, 1.0, a)
    exact = (x + np.expm1(-x)) / safe_a**2
    series = t**2 / 2 - a * t**3 / 6 + a**2 * t**4 / 24
    return np.where(small, series, exact)


def charged_observable_bound(t, dH_norm, gamma_c, q, O_norm=1.0):
    """Theorem B2 for one charged component: ||dH|| ||O_q|| (1 - e^{-2 gamma_c q^2 t}) / (gamma_c q^2)."""
    return 2 * dH_norm * O_norm * _relax(2 * gamma_c * q**2, t)


def zeno_bound(t, dH_norm, dHq_norms: dict, gamma_c, O_norm=1.0):
    """Theorem B3: sum_{q != 0} 4 osc(O) ||dH|| ||dH_q|| phi(2 gamma_c q^2, t); dHq_norms holds both signs."""
    t = np.asarray(t, float)
    total = np.zeros_like(t)
    for q, norm_q in dHq_norms.items():
        if q != 0:
            total = total + 4 * O_norm * dH_norm * norm_q * _phi(2 * gamma_c * q**2, t)
    return total
