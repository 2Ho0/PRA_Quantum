"""Classical collective noise H_xi(t) = xi(t) S^z with xi a finite-state Markov chain (Kubo's stochastic Liouville equation).

A chain has states xi_k, generator W (dp/dt = W p, columns summing to zero) and stationary distribution pi. The
noise-averaged state is sum_k rho_k(t), where
    d rho_k/dt = L(rho_k) - i xi_k [S^z, rho_k] + sum_j W_kj rho_j,    rho_k(0) = pi_k rho_0,
which is exact for the chain started in its stationary state. Quasi-static noise is the case W = 0 on Gauss-Hermite
nodes; telegraph noise is a two-state chain; a sum of n independent telegraph processes (the Ehrenfest chain) has the
Ornstein-Uhlenbeck correlation sigma^2 exp(-|tau|/tau_c) exactly and tends to the Gaussian OU process as n grows.
White noise with E xi(t) xi(t') = 4 gamma_c delta(t - t') averages to the collective dephasing D_c(gamma_c) =
-2 gamma_c [S^z, [S^z, .]] of qerr.lindblad (Gamma = gamma_c 11^T).
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.special import comb

from . import spins
from .lindblad import OpenModel, lindblad_rhs


@dataclass(frozen=True, eq=False)
class ChainNoise:
    xi: np.ndarray
    W: np.ndarray
    pi: np.ndarray

    def chi(self, q, t):
        """chi_q(t) = E exp(-i q int_0^t xi) from the stationary state (Feynman-Kac for the chain)."""
        ts = np.atleast_1d(np.asarray(t, float))
        M = self.W - 1j * q * np.diag(self.xi)
        out = np.array([np.sum(expm(s * M) @ self.pi) for s in ts])
        return out if np.ndim(t) else out[0]


def quasi_static(sigma, n_nodes=60, width=8.0) -> ChainNoise:
    """Static Gaussian field xi ~ N(0, sigma^2) on a uniform grid over +-width sigma (trapezoid weights).

    Any finite set of static fields revives: a charge-q coherence returns at t = 2 pi / (q dxi), dxi the grid step.
    A uniform grid pushes that time out to 2 pi (n_nodes - 1) / (2 width q sigma), far beyond Gauss-Hermite nodes.
    """
    xi = np.linspace(-width * sigma, width * sigma, n_nodes)
    w = np.exp(-0.5 * (xi / sigma) ** 2)
    return ChainNoise(xi=xi, W=np.zeros((n_nodes, n_nodes)), pi=w / w.sum())


def revival_time(noise: ChainNoise, q) -> float:
    """First revival 2 pi / (q dxi) of a quasi-static grid (inf for a chain with transitions)."""
    if np.any(noise.W):
        return np.inf
    return 2.0 * np.pi / (abs(q) * np.min(np.diff(noise.xi)))


def telegraph(a, lam) -> ChainNoise:
    """xi = +-a, switching at rate lam in each direction (correlation a^2 exp(-2 lam |tau|))."""
    return ChainNoise(xi=np.array([a, -a]), W=lam * np.array([[-1.0, 1.0], [1.0, -1.0]]), pi=np.array([0.5, 0.5]))


def ehrenfest(sigma, tau_c, n) -> ChainNoise:
    """Sum of n independent telegraph processes of amplitude sigma/sqrt(n): OU correlation, Gaussian as n -> inf."""
    a, lam = sigma / np.sqrt(n), 1.0 / (2.0 * tau_c)
    k = np.arange(n + 1)
    W = np.zeros((n + 1, n + 1))
    for j in k:  # column j: rates out of the state with j telegraphs at +a
        if j < n:
            W[j + 1, j] = (n - j) * lam
        if j > 0:
            W[j - 1, j] = j * lam
        W[j, j] = -n * lam
    return ChainNoise(xi=a * (2.0 * k - n), W=W, pi=comb(n, k) / 2.0**n)


def averaged_evolve(model: OpenModel, rho0, times, noise: ChainNoise, rtol=1e-10, atol=1e-12) -> np.ndarray:
    """Noise-averaged e^{t L} rho0 under the extra collective field xi(t) S^z, at each time, shape (T, D, D)."""
    times = np.asarray(times, float)
    D = rho0.shape[0]
    m = spins.zvals(model.n).sum(axis=1) / 2.0
    dm = m[:, None] - m[None, :]
    K = len(noise.xi)
    y0 = (noise.pi[:, None, None] * np.asarray(rho0, complex)[None]).ravel()

    stacked = isinstance(model.H, np.ndarray) and not model.jumps and not np.any(model.emission)
    phase = -1j * noise.xi[:, None, None] * dm[None]
    W_nonzero = bool(np.any(noise.W))

    def rhs(_, y):
        R = y.reshape(K, D, D)
        if stacked:  # dense H, sigma^z dephasing only: one batched matrix product for all noise states
            out = -1j * (model.H @ R - R @ model.H) + (model._elementwise[None] + phase) * R
        else:
            out = np.stack([lindblad_rhs(model, R[k]) for k in range(K)]) + phase * R
        if W_nonzero:
            out += np.einsum("kj,jab->kab", noise.W, R)
        return out.ravel()

    if times[-1] == 0.0:
        return np.repeat(np.asarray(rho0, complex)[None], len(times), axis=0)
    sol = solve_ivp(rhs, (0.0, times[-1]), y0, t_eval=times, method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T.reshape(len(times), K, D, D).sum(axis=1)
