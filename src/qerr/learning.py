"""Integral-method learning of a Lindbladian from time series of Pauli expectation values.

Model: H = sum_k theta_k P_k; dephasing D_Gamma(X) = sum_ij Gamma_ij (Z_i X Z_j - 1/2 {Z_j Z_i, X}) with
symmetric Gamma (parameters Gamma_ii and Gamma_ij, i < j, each of the latter entering twice); emission
with jump sqrt(gamma_e) sigma^-_e, sigma^- = |1><0| (the qerr.lindblad convention); optional single-site noise
D(rho) = sum_{A,B in XYZ} K_AB (A rho B - 1/2 {BA, rho}) with K = S + iT (S symmetric, T antisymmetric, K_ZZ kept
in Gamma), 8 parameters per noise site in the order NOISE_KEYS. For every observable O,
  <O>(t) - <O>(0) = sum_k theta_k  int_0^t <G_k(O)>(s) ds,   L^dag(O) = sum_k theta_k G_k(O),
which is linear in theta; the integrals are taken over a cubic spline through the samples with t <= t_learn.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear

from qerr.lindblad import OpenModel, evolve
from qerr.pauli import comm_i, expectation_tables, expectations, pmul, psparse


NOISE_KEYS = ("SXX", "SYY", "SXY", "TXY", "SXZ", "TXZ", "SYZ", "TYZ")


def _unit_kossakowski(key):
    K = np.zeros((3, 3), complex)
    a, b = "XYZ".index(key[1]), "XYZ".index(key[2])
    if key[0] == "S":
        K[a, b] = K[b, a] = 1.0
    else:
        K[a, b], K[b, a] = 1j, -1j
    return K


_NOISE_UNITS = {key: _unit_kossakowski(key) for key in NOISE_KEYS}


@dataclass
class Ansatz:
    n: int
    h_terms: list
    deph_pairs: list
    emission_sites: list = field(default_factory=list)
    noise_sites: list = field(default_factory=list)

    @property
    def n_params(self) -> int:
        return len(self.h_terms) + len(self.deph_pairs) + len(self.emission_sites) + len(NOISE_KEYS) * len(
            self.noise_sites)


def noise_charges() -> dict:
    """Charge-definite combinations ((key, sign), ...) of the noise parameters and their superoperator charges.

    In the jump basis (sigma+, sigma-, sigma^z) the term K'_ab has charge q_a - q_b; emission and absorption rates
    are SXX + SYY +- 2 TXY.
    """
    return {(("SXX", 1), ("SYY", 1)): {0}, (("TXY", 1),): {0},
            (("SXX", 1), ("SYY", -1)): {-2, 2}, (("SXY", 1),): {-2, 2},
            (("SXZ", 1),): {-1, 1}, (("TXZ", 1),): {-1, 1}, (("SYZ", 1),): {-1, 1}, (("TYZ", 1),): {-1, 1}}


def _kossakowski_adjoint(O, site, K) -> dict:
    """D^dag(O) = sum_AB K_AB (B O A - 1/2 {BA, O}) for single-site Paulis at `site`, as {string: coefficient}."""
    out = {}

    def add(phase, R, c):
        out[R] = out.get(R, 0.0) + c * phase

    for a, A in enumerate("XYZ"):
        for b, B in enumerate("XYZ"):
            k = K[a, b]
            if k == 0:
                continue
            pa, pb = ((site, A),), ((site, B),)
            ph1, R1 = pmul(pb, O)
            ph2, R2 = pmul(R1, pa)
            add(ph1 * ph2, R2, k)
            ph_ba, C = pmul(pb, pa)
            ph3, R3 = pmul(C, O)
            add(ph_ba * ph3, R3, -0.5 * k)
            ph4, R4 = pmul(O, C)
            add(ph_ba * ph4, R4, -0.5 * k)
    return out


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
    k2 = k1 + len(ans.emission_sites)
    for m, e in enumerate(ans.noise_sites):
        if dict(O).get(e) is None:
            continue  # unital: noise at a site outside supp(O) does not act on O
        for u, key in enumerate(NOISE_KEYS):
            for R, c in _kossakowski_adjoint(O, e, _NOISE_UNITS[key]).items():
                if abs(c) > 1e-12:
                    assert abs(c.imag) < 1e-12, (O, key, c)
                    terms.append((k2 + len(NOISE_KEYS) * m + u, float(c.real), R))
    return terms


_GL_X, _GL_W = np.polynomial.legendre.leggauss(4)  # exact for (cubic spline) x (polynomial weight of degree <= 4)


def _weighted_cumulative(knots, spline, weight):
    """int_{knots[0]}^{knots[i]} weight(s) spline(s) ds for every knot, by 4-point Gauss-Legendre on each interval."""
    a, b = knots[:-1], knots[1:]
    s = 0.5 * (a + b)[:, None] + 0.5 * (b - a)[:, None] * _GL_X[None]
    vals = spline(s)                                              # (n_int, 4) + value shape
    w = (0.5 * (b - a)[:, None] * _GL_W[None] * weight(s)).reshape(s.shape + (1,) * (vals.ndim - 2))
    pieces = (w * vals).sum(axis=1)
    return np.concatenate([np.zeros((1,) + pieces.shape[1:]), np.cumsum(pieces, axis=0)])


def _spline_integrals(tw, vals, weight=None):
    spline = CubicSpline(tw, vals)
    if weight is None:
        return np.array([spline.integrate(tw[0], t) for t in tw])
    return _weighted_cumulative(tw, spline, weight)


def _window(times, t_start, t_learn):
    return (times >= t_start - 1e-12) & (times <= t_learn + 1e-12)


def integral_system(obs, ans: Ansatz, series, times, t_learn, states=(0, 1), defect=None, t_start=0.0,
                    weight=None):
    """Rows (state, O, t_m) for t_start < t_m <= t_learn; rows needing an unavailable series are dropped.

    A row reads <O>(t_m) - <O>(t_start) = sum_k theta_k int_{t_start}^{t_m} <G_k(O)>, so a window that starts after
    t = 0 estimates the generator acting there. `defect[(state, S)]` (from `quadrature_defect` or
    `trajectory_defect` for the same window and weight) is added to the spline integral of <S>.
    With `weight` w(s), every column integral becomes int w(s) <G_k(O)>(s) ds (the rows are unchanged): a parameter
    with time profile w, such as the slope b of a collective rate a + b s, is a weighted column.
    """
    times = np.asarray(times, float)
    win = _window(times, t_start, t_learn)
    tw = times[win]
    cache = {}

    def integral(state, S):
        key = (state, S)
        if key not in cache:
            vals = np.ones(len(tw)) if S == () else series.get(key, np.full(len(times), np.nan))[win]
            if np.any(~np.isfinite(vals)):
                cache[key] = np.full(len(tw), np.nan)
            else:
                cache[key] = _spline_integrals(tw, vals, weight) + (0.0 if defect is None else defect.get(key, 0.0))
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


def refine(times, n_sub) -> np.ndarray:
    """The grid `times` with every interval split into n_sub equal steps."""
    times = np.asarray(times, float)
    return np.concatenate([np.linspace(a, b, n_sub + 1)[:-1] for a, b in zip(times[:-1], times[1:])] + [times[-1:]])


def model_trajectory(model, rho0s, strings, fine, rtol=1e-9, atol=1e-11) -> list:
    """[(len(fine), len(strings)) array of <S>_model at the times `fine`, for each initial state]; starts at t = 0."""
    idx, ph = expectation_tables(strings, model.H.shape[0].bit_length() - 1)
    out = []
    for rho in rho0s:
        if fine[0] > 0.0:
            rho = evolve(model, rho, [0.0, fine[0]], rtol=rtol, atol=atol)[-1]
        vals = [expectations(rho, idx, ph)]
        for a, b in zip(fine[:-1], fine[1:]):
            rho = evolve(model, rho, [0.0, b - a], rtol=rtol, atol=atol)[-1]
            vals.append(expectations(rho, idx, ph))
        out.append(np.array(vals))
    return out


def trajectory_defect(fine, vals, strings, times, t_learn, t_start=0.0, weight=None) -> dict:
    """defect[(s, S)][m] = int_{t_start}^{t_m} <S>_model minus the spline quadrature of its samples on `times`.

    `vals` is the output of `model_trajectory` on a grid `fine` that contains every point of `times` in the window;
    the integral over `fine` stands for the exact one. One trajectory thus serves every window inside it.
    With `weight`, both integrals carry w(s), matching `integral_system(..., weight=w)`.
    """
    times, fine = np.asarray(times, float), np.asarray(fine, float)
    tw = times[_window(times, t_start, t_learn)]
    fw = _window(fine, tw[0], tw[-1])
    at = np.searchsorted(fine, tw - 1e-12)
    assert np.allclose(fine[at], tw, atol=1e-12, rtol=0.0), "the fine grid must contain the window's grid points"
    sub = np.searchsorted(fine[fw], tw - 1e-12)
    defect = {}
    for s, v in enumerate(vals):
        exact = CubicSpline(fine[fw], v[fw], axis=0)
        coarse = CubicSpline(tw, v[at], axis=0)
        if weight is None:
            diff = np.array([exact.integrate(tw[0], t) - coarse.integrate(tw[0], t) for t in tw])
        else:
            diff = _weighted_cumulative(fine[fw], exact, weight)[sub] - _weighted_cumulative(tw, coarse, weight)
        for k, S in enumerate(strings):
            defect[(s, S)] = diff[:, k]
    return defect


def quadrature_defect(model, rho0s, strings, times, t_learn, n_sub=16, rtol=1e-9, atol=1e-11, t_start=0.0):
    """defect[(s, S)][m] = int_{t_start}^{t_m} <S>_model exactly minus the spline quadrature of its grid samples.

    The model is integrated on a grid refined n_sub times; adding the defect to the spline integrals of the
    data removes the discretisation bias of a coarse time grid to the extent that the model is accurate.
    """
    times = np.asarray(times, float)
    fine = refine(times[_window(times, t_start, t_learn)], n_sub)
    vals = model_trajectory(model, rho0s, strings, fine, rtol=rtol, atol=atol)
    return trajectory_defect(fine, vals, strings, times, t_learn, t_start=t_start)


def learn(series, ans: Ansatz, obs, times, t_learn, rho0s, n_iter=3, states=(0, 1), t_start=0.0, **defect_options):
    """Integral-method fit on (t_start, t_learn] followed by n_iter defect-corrected refits; every round's estimate.

    defect_options (n_sub, rtol, atol) are passed to `quadrature_defect`.
    """
    A, y, _ = integral_system(obs, ans, series, times, t_learn, states, t_start=t_start)
    thetas = [fit(A, y)]
    strings = needed_strings(obs, ans)
    for _ in range(n_iter):
        defect = quadrature_defect(to_openmodel(thetas[-1], ans, sparse=True), rho0s, strings, times, t_learn,
                                   t_start=t_start, **defect_options)
        A, y, _ = integral_system(obs, ans, series, times, t_learn, states, defect=defect, t_start=t_start)
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
    if ans.noise_sites:
        raise NotImplementedError("single-site noise is not simulated here; fit with a precomputed defect")
    H = sum(theta[k] * psparse(P, ans.n) for k, P in enumerate(ans.h_terms)).tocsr()
    G = gamma_matrix(theta, ans) if gamma is None else gamma
    return OpenModel(H=H if sparse else H.toarray(), Gamma=G, emission=_emission(theta, ans))


def _emission(theta, ans: Ansatz) -> np.ndarray:
    k1 = len(ans.h_terms) + len(ans.deph_pairs)
    emission = np.zeros(ans.n)
    for m, e in enumerate(ans.emission_sites):
        emission[e] = theta[k1 + m]
    return emission


def site_kossakowski(theta, ans: Ansatz) -> np.ndarray:
    """(n, 3, 3) single-site Kossakowski matrices over (X, Y, Z); K_ZZ = Gamma_ii, other entries from the noise sites."""
    K = np.zeros((ans.n, 3, 3), complex)
    K[:, 2, 2] = np.diag(gamma_matrix(theta, ans))
    k2 = len(ans.h_terms) + len(ans.deph_pairs) + len(ans.emission_sites)
    for m, e in enumerate(ans.noise_sites):
        block = theta[k2 + len(NOISE_KEYS) * m:k2 + len(NOISE_KEYS) * (m + 1)]
        K[e] += sum(v * _NOISE_UNITS[key] for v, key in zip(block, NOISE_KEYS))
    return K


def flip_rates(K: np.ndarray):
    """Emission and absorption rates (the sigma-/sigma+ diagonal of K in the charge basis): SXX + SYY +- 2 TXY."""
    S = K[..., 0, 0].real + K[..., 1, 1].real
    T = K[..., 0, 1].imag
    return S + 2 * T, S - 2 * T


def structured_map(ans: Ansatz) -> np.ndarray:
    """P with theta = P theta_s; theta_s replaces the dephasing block by (gamma_c, g_1..g_n), Gamma = gamma_c 11^T + diag(g)."""
    k0, nd = len(ans.h_terms), len(ans.deph_pairs)
    rest = ans.n_params - k0 - nd
    P = np.zeros((ans.n_params, k0 + 1 + ans.n + rest))
    P[:k0, :k0] = np.eye(k0)
    for m, (i, j) in enumerate(ans.deph_pairs):
        P[k0 + m, k0] = 1.0
        if i == j:
            P[k0 + m, k0 + 1 + i] = 1.0
    P[k0 + nd:, k0 + 1 + ans.n:] = np.eye(rest)
    return P


def fit_bounded(A, y, lower):
    """Least squares with lower bounds (-inf for free parameters), by bounded-variable least squares."""
    return lsq_linear(A, y, bounds=(np.asarray(lower, float), np.inf), method="bvls", tol=1e-12).x


def fit_psd(A, y, ans: Ansatz, n_iter=20000, tol=1e-12):
    """Least squares subject to Gamma(theta) PSD.

    The other parameters are profiled out exactly. The dephasing block is solved by FISTA with adaptive restart in
    coordinates x_ii = Gamma_ii, x_ij = sqrt(2) Gamma_ij, where the Euclidean projection is eigenvalue clipping of
    Gamma. The best iterate is kept, so the result is never worse than the clipped unconstrained solution.
    """
    n, k0, nd = ans.n, len(ans.h_terms), len(ans.deph_pairs)
    assert nd == n * (n + 1) // 2, "fit_psd needs the full dephasing matrix in the ansatz"
    gcols = np.arange(k0, k0 + nd)
    ocols = np.r_[0:k0, k0 + nd:ans.n_params]
    Ao = A[:, ocols]
    Q, R = np.linalg.qr(Ao)
    r = np.abs(np.diag(R))
    if r.min() <= 1e-10 * r.max():
        raise ValueError("fit_psd: the non-dephasing columns are rank deficient (e.g. emission and noise sites "
                         "together), so they cannot be profiled out")
    B = A[:, gcols] - Q @ (Q.T @ A[:, gcols])
    c = y - Q @ (Q.T @ y)
    s = np.array([1.0 if i == j else np.sqrt(0.5) for i, j in ans.deph_pairs])
    Bs = B * s
    BtB, Btc = Bs.T @ Bs, Bs.T @ c
    L = np.linalg.eigvalsh(BtB)[-1]
    rows = np.array([i for i, _ in ans.deph_pairs])
    cols = np.array([j for _, j in ans.deph_pairs])

    def project(x):
        G = np.zeros((n, n))
        G[rows, cols] = x * s
        G[cols, rows] = x * s
        w, V = np.linalg.eigh(G)
        return ((V * np.clip(w, 0.0, None)) @ V.T)[rows, cols] / s

    def objective(x):
        r = Bs @ x - c
        return r @ r

    x = project(np.linalg.lstsq(Bs, c, rcond=None)[0])
    best, best_obj = x, objective(x)
    z, t = x, 1.0
    for _ in range(n_iter):
        grad = BtB @ z - Btc
        x_new = project(z - grad / L)
        if grad @ (x_new - x) > 0:  # adaptive restart (O'Donoghue and Candes)
            z, t = x, 1.0
            continue
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        step = np.linalg.norm(x_new - x)
        x, t = x_new, t_new
        obj = objective(x)
        if obj < best_obj:
            best, best_obj = x, obj
        if step <= tol * max(1.0, np.linalg.norm(x)):
            break
    gamma = best * s
    theta = np.zeros(ans.n_params)
    theta[gcols] = gamma
    theta[ocols] = np.linalg.lstsq(Ao, y - A[:, gcols] @ gamma, rcond=None)[0]
    return theta
