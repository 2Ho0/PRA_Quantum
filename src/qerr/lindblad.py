"""Dense Lindblad simulator for small spin systems, in both pictures.

Generator (Schrodinger picture):
    L(X) = -i[H, X]
           + sum_ij Gamma_ij (Z_i X Z_j - 1/2 {Z_j Z_i, X})      sigma^z dephasing
           + sum_i e_i (s-_i X s+_i - 1/2 {s+_i s-_i, X})        emission, s- = |1><0|
           + sum_k (J_k X J_k^dag - 1/2 {J_k^dag J_k, X})        extra jumps
Dephasing and the emission anticommutators act elementwise in the z basis,
which keeps the right-hand side cheap.
"""

import gc
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.integrate import solve_ivp

from . import spins


@dataclass(frozen=True, eq=False)
class OpenModel:
    H: np.ndarray
    Gamma: np.ndarray
    emission: np.ndarray
    jumps: tuple = ()

    @property
    def n(self) -> int:
        return int(round(np.log2(self.H.shape[0])))

    @cached_property
    def _elementwise(self) -> np.ndarray:
        """Real (D, D) factor W with D_Gamma(X) + emission anticommutators = W * X."""
        z = spins.zvals(self.n).astype(float)
        a = z @ self.Gamma
        q = np.sum(a * z, axis=1)
        w = -0.5 * (q[:, None] + q[None, :] - 2.0 * a @ z.T)
        p0 = 1.0 - spins.bits(self.n)
        w -= 0.5 * (p0 @ self.emission)[:, None] + 0.5 * (p0 @ self.emission)[None, :]
        return w

    @cached_property
    def _emission_maps(self) -> list:
        """(rate, idx0, idx1): basis states with bit_i = 0 and their bit-flipped partners."""
        b = spins.bits(self.n)
        maps = []
        for i, rate in enumerate(self.emission):
            if rate != 0.0:
                idx0 = np.flatnonzero(b[:, i] == 0)
                maps.append((rate, np.ix_(idx0, idx0), np.ix_(idx0 + 2 ** (self.n - 1 - i), idx0 + 2 ** (self.n - 1 - i))))
        return maps

    @cached_property
    def _jumps(self) -> list:
        return [(J, J.conj().T, J.conj().T @ J) for J in self.jumps]


def lindblad_rhs(model: OpenModel, X: np.ndarray) -> np.ndarray:
    out = -1j * (model.H @ X - X @ model.H) + model._elementwise * X
    for rate, sel0, sel1 in model._emission_maps:
        out[sel1] += rate * X[sel0]
    for J, Jd, JdJ in model._jumps:
        out += J @ X @ Jd - 0.5 * (JdJ @ X + X @ JdJ)
    return out


def adjoint_rhs(model: OpenModel, A: np.ndarray) -> np.ndarray:
    out = 1j * (model.H @ A - A @ model.H) + model._elementwise * A
    for rate, sel0, sel1 in model._emission_maps:
        out[sel0] += rate * A[sel1]
    for J, Jd, JdJ in model._jumps:
        out += Jd @ A @ J - 0.5 * (JdJ @ A + A @ JdJ)
    return out


def evolve(model: OpenModel, X0: np.ndarray, times, heisenberg: bool = False,
           rtol: float = 1e-10, atol: float = 1e-12) -> np.ndarray:
    """Return e^{tL}(X0) (or e^{tL^dag}(X0) if heisenberg) at each time, shape (T, D, D)."""
    times = np.asarray(times, dtype=float)
    D = X0.shape[0]
    y0 = np.asarray(X0, dtype=complex).ravel()
    if times[-1] == 0.0:
        return np.repeat(y0.reshape(1, D, D), len(times), axis=0)
    rhs = adjoint_rhs if heisenberg else lindblad_rhs
    sol = solve_ivp(lambda t, y: rhs(model, y.reshape(D, D)).ravel(), (0.0, times[-1]), y0,
                    t_eval=times, method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    out = sol.y.T.reshape(len(times), D, D).copy()
    del sol
    if D >= 128:
        gc.collect()  # the solver's stage arrays sit in reference cycles; free them now, not at the next GC pass
    return out


def evolve_td(model: OpenModel, X0: np.ndarray, times, Gamma_td: np.ndarray, rate,
              rtol: float = 1e-10, atol: float = 1e-12) -> np.ndarray:
    """Schrodinger-picture evolution under L + rate(t) D_{Gamma_td}: a time-dependent sigma^z dephasing on top of L."""
    times = np.asarray(times, dtype=float)
    D = X0.shape[0]
    W = OpenModel(H=np.zeros((D, D)), Gamma=Gamma_td, emission=np.zeros(model.n))._elementwise
    y0 = np.asarray(X0, dtype=complex).ravel()
    sol = solve_ivp(lambda t, y: (lindblad_rhs(model, y.reshape(D, D)) + rate(t) * W * y.reshape(D, D)).ravel(),
                    (0.0, times[-1]), y0, t_eval=times, method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T.reshape(len(times), D, D)


def expect(Xs: np.ndarray, O: np.ndarray) -> np.ndarray:
    """Tr[O X_t] for each t."""
    return np.einsum("ij,tji->t", O, Xs)


def superop(model: OpenModel) -> np.ndarray:
    """Dense column-stacking superoperator, built independently from Kronecker products."""
    n, H = model.n, model.H
    I = np.eye(H.shape[0])

    def dissipator(J, rate):
        JdJ = J.conj().T @ J
        return rate * (np.kron(J.conj(), J) - 0.5 * np.kron(I, JdJ) - 0.5 * np.kron(JdJ.T, I))

    L = -1j * (np.kron(I, H) - np.kron(H.T, I))
    Z = [spins.pauli_op(n, i, "z") for i in range(n)]
    for i in range(n):
        for j in range(n):
            if model.Gamma[i, j] != 0.0:
                ZZ = Z[j] @ Z[i]
                L = L + model.Gamma[i, j] * (np.kron(Z[j].T, Z[i]) - 0.5 * np.kron(I, ZZ) - 0.5 * np.kron(ZZ.T, I))
        if model.emission[i] != 0.0:
            L = L + dissipator(spins.pauli_op(n, i, "-"), model.emission[i])
    for J in model.jumps:
        L = L + dissipator(J, 1.0)
    return L
