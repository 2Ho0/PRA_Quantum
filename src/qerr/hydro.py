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

import gc

import numpy as np
from scipy.integrate import solve_ivp

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


def nn_commutator(e: np.ndarray, X: np.ndarray) -> np.ndarray:
    """i[h, X] for the real nearest-neighbour hopping h_{x,x+1} = h_{x+1,x} = e_x, by array shifts (O(N^2))."""
    hX = np.zeros_like(X)
    hX[:-1] += e[:, None] * X[1:]
    hX[1:] += e[:, None] * X[:-1]
    Xh = np.zeros_like(X)
    Xh[:, :-1] += X[:, 1:] * e[None, :]
    Xh[:, 1:] += X[:, :-1] * e[None, :]
    return 1j * (hX - Xh)


def evolve_two_point(e, g, X0, times, rtol=1e-10, atol=1e-12, blocks=10) -> np.ndarray:
    """X(t) for dX/dt = i[h, X] - Lambda o X, Lambda_xy = (g_x + g_y)/2 off the diagonal and 0 on it.

    For real nearest-neighbour hopping (amplitudes e) with number dephasing g, this one equation evolves both the
    two-point matrix C_xy = <c_x^dag c_y> of the state and the coefficient matrix w of O = sum w_xy c_x^dag c_y in
    the Heisenberg picture (the pairing sum_xy w_xy C_xy is invariant). The time grid is integrated in `blocks`
    pieces so that only one block of samples is held at a time.
    """
    times = np.asarray(times, float)
    e, g = np.asarray(e, float), np.asarray(g, float)
    N = X0.shape[0]
    Lam = 0.5 * (g[:, None] + g[None, :])
    np.fill_diagonal(Lam, 0.0)

    def rhs(_, y):
        X = y.reshape(N, N)
        return (nn_commutator(e, X) - Lam * X).ravel()

    out = np.empty((len(times), N, N), complex)
    X = np.asarray(X0, complex)
    start = 0
    if times[0] == 0.0:
        out[0] = X
        start = 1
    t_prev = 0.0
    for block in np.array_split(np.arange(start, len(times)), min(blocks, max(1, len(times) - start))):
        if len(block) == 0:
            continue
        sol = solve_ivp(rhs, (t_prev, times[block[-1]]), X.ravel(), t_eval=times[block], method="DOP853", rtol=rtol,
                        atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        out[block] = sol.y.T.reshape(len(block), N, N)
        X, t_prev = out[block[-1]].copy(), times[block[-1]]
        del sol
        gc.collect()  # solver stage arrays sit in reference cycles
    return out


def state_factors(C: np.ndarray):
    """(gamma, kappa): the largest density gradient and the largest coherence of the state."""
    n = np.real(np.diag(C))
    off = np.abs(C - np.diag(np.diag(C)))
    return float(np.abs(np.diff(n)).max()), float(off.max())
