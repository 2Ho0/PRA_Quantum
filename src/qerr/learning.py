"""Integral-method learning of a Lindbladian from time series of Pauli expectation values.

Model: H = sum_k theta_k P_k; dephasing D_Gamma(X) = sum_ij Gamma_ij (Z_i X Z_j - 1/2 {Z_j Z_i, X}) with
symmetric Gamma (parameters Gamma_ii and Gamma_ij, i < j, each of the latter entering twice); emission
with jump sqrt(gamma_e) sigma^-_e, sigma^- = |1><0| (the qerr.lindblad convention). For every observable O,
  <O>(t) - <O>(0) = sum_k theta_k  int_0^t <G_k(O)>(s) ds,   L^dag(O) = sum_k theta_k G_k(O),
which is linear in theta; the integrals are taken over a cubic spline through the samples with t <= t_learn.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline

from qerr.lindblad import OpenModel, evolve
from qerr.pauli import comm_i, expectation_tables, expectations, pmul, psparse


@dataclass
class Ansatz:
    n: int
    h_terms: list
    deph_pairs: list
    emission_sites: list = field(default_factory=list)

    @property
    def n_params(self) -> int:
        return len(self.h_terms) + len(self.deph_pairs) + len(self.emission_sites)


def _anticommutes_z(O, site) -> bool:
    return dict(O).get(site) in ("X", "Y")


def adjoint_terms(O, ans: Ansatz):
    """L^dag(O) = sum theta_k * coef * S over the returned (k, coef, S); S = () is the identity."""
    terms = []
    for k, P in enumerate(ans.h_terms):
        c, R = comm_i(P, O)
        if R is not None:
            terms.append((k, c, R))
    k0 = len(ans.h_terms)
    for m, (i, j) in enumerate(ans.deph_pairs):
        if not (_anticommutes_z(O, i) and _anticommutes_z(O, j)):
            continue
        if i == j:
            terms.append((k0 + m, -2.0, O))
        else:
            ph, R = pmul(((i, "Z"), (j, "Z")), O)
            terms.append((k0 + m, float((-4 * ph).real), R))
    k1 = k0 + len(ans.deph_pairs)
    for m, e in enumerate(ans.emission_sites):
        letter = dict(O).get(e)
        if letter in ("X", "Y"):
            terms.append((k1 + m, -0.5, O))
        elif letter == "Z":
            rest = tuple((s, a) for s, a in O if s != e)
            terms += [(k1 + m, -1.0, rest), (k1 + m, -1.0, O)]
    return terms


def _spline_integrals(tw, vals):
    spline = CubicSpline(tw, vals)
    return np.array([spline.integrate(0.0, t) for t in tw])


def integral_system(obs, ans: Ansatz, series, times, t_learn, states=(0, 1), defect=None):
    """Rows (state, O, t_m) for 0 < t_m <= t_learn; rows needing an unavailable series are dropped.

    `defect[(state, S)]` (from `quadrature_defect`) is added to the spline integral of <S>.
    """
    times = np.asarray(times, float)
    win = times <= t_learn + 1e-12
    tw = times[win]
    cache = {}

    def integral(state, S):
        key = (state, S)
        if key not in cache:
            vals = np.ones(len(tw)) if S == () else series.get(key, np.full(len(times), np.nan))[win]
            if np.any(~np.isfinite(vals)):
                cache[key] = np.full(len(tw), np.nan)
            else:
                cache[key] = _spline_integrals(tw, vals) + (0.0 if defect is None else defect.get(key, 0.0))
        return cache[key]

    A, y, rows = [], [], []
    for s in states:
        for O in obs:
            vals = series.get((s, O))
            if vals is None or np.any(~np.isfinite(vals[win])):
                continue
            block = np.zeros((len(tw), ans.n_params))
            for k, c, S in adjoint_terms(O, ans):
                block[:, k] += c * integral(s, S)
            for m in range(1, len(tw)):
                if np.all(np.isfinite(block[m])):
                    A.append(block[m])
                    y.append(vals[win][m] - vals[win][0])
                    rows.append((s, O, tw[m]))
    return np.array(A), np.array(y), rows


def fit(A, y):
    return np.linalg.lstsq(A, y, rcond=None)[0]


def needed_strings(obs, ans: Ansatz) -> list:
    return sorted({S for O in obs for _, _, S in adjoint_terms(O, ans) if S != ()})


def quadrature_defect(model, rho0s, strings, times, t_learn, n_sub=16, rtol=1e-9, atol=1e-11):
    """defect[(s, S)][m] = int_0^{t_m} <S>_model exactly minus the spline quadrature of its grid samples.

    The model is integrated on a grid refined n_sub times; adding the defect to the spline integrals of the
    data removes the discretisation bias of a coarse time grid to the extent that the model is accurate.
    """
    times = np.asarray(times, float)
    tw = times[times <= t_learn + 1e-12]
    fine = np.concatenate([np.linspace(a, b, n_sub + 1)[:-1] for a, b in zip(tw[:-1], tw[1:])] + [tw[-1:]])
    coarse_at = np.searchsorted(fine, tw)
    idx, ph = expectation_tables(strings, model.H.shape[0].bit_length() - 1)
    defect = {}
    for s, rho in enumerate(rho0s):
        vals = [expectations(rho, idx, ph)]
        for a, b in zip(fine[:-1], fine[1:]):
            rho = evolve(model, rho, [0.0, b - a], rtol=rtol, atol=atol)[-1]
            vals.append(expectations(rho, idx, ph))
        vals = np.array(vals)                                   # (n_fine, n_strings)
        exact = CubicSpline(fine, vals, axis=0)
        exact_int = np.array([exact.integrate(0.0, t) for t in tw])
        coarse = CubicSpline(tw, vals[coarse_at], axis=0)
        coarse_int = np.array([coarse.integrate(0.0, t) for t in tw])
        for k, S in enumerate(strings):
            defect[(s, S)] = exact_int[:, k] - coarse_int[:, k]
    return defect


def learn(series, ans: Ansatz, obs, times, t_learn, rho0s, n_iter=3, states=(0, 1), **defect_options):
    """Integral-method fit followed by n_iter defect-corrected refits; returns the estimate of every round.

    defect_options (n_sub, rtol, atol) are passed to `quadrature_defect`.
    """
    A, y, _ = integral_system(obs, ans, series, times, t_learn, states)
    thetas = [fit(A, y)]
    strings = needed_strings(obs, ans)
    for _ in range(n_iter):
        defect = quadrature_defect(to_openmodel(thetas[-1], ans, sparse=True), rho0s, strings, times, t_learn,
                                   **defect_options)
        A, y, _ = integral_system(obs, ans, series, times, t_learn, states, defect=defect)
        thetas.append(fit(A, y))
    return thetas


def gamma_matrix(theta, ans: Ansatz) -> np.ndarray:
    k0 = len(ans.h_terms)
    G = np.zeros((ans.n, ans.n))
    for m, (i, j) in enumerate(ans.deph_pairs):
        G[i, j] = G[j, i] = theta[k0 + m]
    return G


def to_openmodel(theta, ans: Ansatz, gamma=None, sparse=False) -> OpenModel:
    """OpenModel of the learned generator; `gamma` overrides the dephasing matrix (e.g. its PSD projection).

    sparse=True stores H as a CSR matrix, which `evolve` accepts and which is much faster for n = 10.
    """
    H = sum(theta[k] * psparse(P, ans.n) for k, P in enumerate(ans.h_terms)).tocsr()
    G = gamma_matrix(theta, ans) if gamma is None else gamma
    return OpenModel(H=H if sparse else H.toarray(), Gamma=G, emission=_emission(theta, ans))


def _emission(theta, ans: Ansatz) -> np.ndarray:
    k1 = len(ans.h_terms) + len(ans.deph_pairs)
    emission = np.zeros(ans.n)
    for m, e in enumerate(ans.emission_sites):
        emission[e] = theta[k1 + m]
    return emission
