"""Two-sided (Heisenberg x Schrodinger) forgetting for number-conserving quadratic fermions (Theorem F).

Number-conserving quadratic operators are O = sum_xy w_xy c_x^dag c_y (+ const) and states enter through the two-point
matrix C_xy = <c_x^dag c_y>. For dH = sum dh_xy c_x^dag c_y, i[dH, O] = sum M_xy c_x^dag c_y with M = i[dh, w], and
<i[dH, O]>_rho = sum_xy M_xy C_xy. The Duhamel identity
    Delta(t) = int_0^t <i[dH, O_u]>_{rho'_{t-u}} du    (O_u: reference Heisenberg, rho': perturbed state)
splits for a nearest-neighbour dh (bond amplitudes e_b) into a density part and a coherence part:
    sum_x M_xx n_x = i sum_b e_b (w_{x+1,x} - w_{x,x+1}) (n_x - n_{x+1})          (summation by parts)
so that |<i[dH, O]>_rho| <= a(w) gamma(C) + m(w) kappa(C), with the observable factors
    a = sum_b |e_b| |w_{x+1,x} - w_{x,x+1}|,   m = sum_{x != y} |M_xy|,
and the state factors gamma = max_b |n_x - n_{x+1}| (largest density gradient) and kappa = max_{x != y} |C_xy|
(largest coherence). Both kinds of factor decay diffusively, so their convolution is time-uniform.
"""

import numpy as np

from .quadratic import majorana_to_bdg


def coefficient_matrix(A: np.ndarray) -> np.ndarray:
    """w with q(A) = sum_xy w_xy c_x^dag c_y + const, for a number-conserving Majorana matrix A."""
    N = A.shape[0] // 2
    return 2.0 * majorana_to_bdg(A)[N:, N:]


def two_point(G: np.ndarray) -> np.ndarray:
    """C_xy = <c_x^dag c_y> from the covariance G_ab = <Psi_a Psi_b^dag> of qerr.fermions."""
    N = G.shape[0] // 2
    return G[N:, N:]


def commutator_coefficients(dh: np.ndarray, w: np.ndarray) -> np.ndarray:
    """M with i[dH, O] = sum M_xy c_x^dag c_y."""
    return 1j * (dh @ w - w @ dh)


def pairing(M: np.ndarray, C: np.ndarray) -> complex:
    """<sum M_xy c_x^dag c_y> in the state with two-point matrix C."""
    return np.sum(M * C)


def observable_factors(w: np.ndarray, dh: np.ndarray):
    """(a, m) for a nearest-neighbour dh: the density-pairing and coherence-pairing weights of i[dH, O]."""
    e = np.abs(np.diag(dh, 1))
    a = float(np.sum(e * np.abs(np.diag(w, -1) - np.diag(w, 1))))
    M = commutator_coefficients(dh, w)
    m = float(np.abs(M).sum() - np.abs(np.diag(M)).sum())
    return a, m


def state_factors(C: np.ndarray):
    """(gamma, kappa): the largest density gradient and the largest coherence of the state."""
    n = np.real(np.diag(C))
    off = np.abs(C - np.diag(np.diag(C)))
    return float(np.abs(np.diff(n)).max()), float(off.max())
