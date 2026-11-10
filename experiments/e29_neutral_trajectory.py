"""E29: learning the U(1)-symmetric Hamiltonian from the neutral trajectories of the window, with a posterior band
(preregistered)

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0), analysed
here from the public raw shots.
Background (Q29-Q31). The late neutral predictions have no validated band. E25: the band of S carries its spread but
not the bias of the currents (XY - YX), which S sets to zero because the integral method does not identify them (their
least-squares slope against the truth is 0.26, E24b). E27: marginalising currents drawn from an independent prior of
the least-squares rms gives a band that is 4-6 times too wide in Z_i and flip-flop; the truth's currents move the late
neutral record less than all 200 prior draws (E27b).
Why the neutral trajectories (Lemma G.0). Collective noise of any spectrum multiplies the charge-q blocks of the state
and leaves its charge-0 blocks alone. The neutral combinations (Z_i, Z_iZ_j, (XX + YY)/2, (XY - YX)/2) of a U(1)-
symmetric generator read only the charge-0 blocks, so they do not see the collective noise, white or quasi-static. A
detuning drift across the settings is a global z rotation, which commutes with the symmetric dynamics; it leaves the
neutral combinations unchanged in expectation and adds only variance. The neutral trajectories therefore depend on
the U(1)-symmetric Hamiltonian (couplings J, currents D, z fields up to their mean) and on the initial state alone. A
forward-model fit of them has no errors-in-variables: its regressors are model predictions, not measured correlators.
Design probe (synthetic only, before this commit; linearised at the control truth, diagonal shot weights of the
data): the window's neutral trajectories make 24 of the 45 current directions data-dominated (prior-whitened
information > 1). The posterior predictive sd of the late neutral classes is 0.013, 0.010 and 0.008 (Z_i, Z_iZ_j,
flip-flop), against 0.053, 0.025 and 0.055 from the independent current prior alone (E27 band 0.049, 0.026, 0.049).
Without noise, the truth's currents leave 0.61, 0.78 and 0.64 posterior sd of late error after the update; draws from
the prior leave 0.36-0.85, so the truth is a typical draw under this posterior.
Estimator T (fixed here).
  Parameters x = (J_ij, D_ij for the 45 pairs, h_i for the 10 ions), with XX = YY = J/2, XY = -YX = D/2 and z fields h.
      The mean of h commutes with the symmetric dynamics and has no effect on the neutral classes; it gets a weak
      prior (sd 1 rad/ms) around its start value.
  Data: the 145 neutral combinations of both initial states at t = 0.5, 1.0 and 1.5 ms (870 values). Noise: per Pauli
      string the setting-level standard error (sd of the per-setting means over the settings that measure it,
      divided by the root of their number), at least the binomial shot error; combinations add these in quadrature
      (the strings of one combination come from different settings).
  Initial state: the product state of the t = 0 single-ion means of the same data set.
  Prior: D_ij ~ N(0, tau_g^2) by distance group g (d = 1, 2, >= 3); flat on J and h. tau_g is the rms of the
      least-squares currents of the same data set in the group (E27's prior scale), fixed.
  Fit: Levenberg-Marquardt on the whitened window residual plus the prior term, Jacobian by forward differences
      (1 rad/s), exact propagation in the magnetisation sectors (charge-0 blocks only). Start: J and h of the
      least-squares Xb15uf fit of the same data set (oracle defect on synthetic data, as E25 and E27), D = 0. Stop when
      the largest step is below 0.1 rad/s, at most 20 iterations.
  Posterior: Laplace, Sigma = b (J^T W J + P)^-1 at the fit, with the Birge factor b = max(1, chi^2 / nu),
      nu = n - tr(J^T W J (J^T W J + P)^-1).
  Band: K = 48 draws, x_k ~ N(x_hat, Sigma) and initial Bloch vectors ~ N(measured, setting-level sem^2) per ion
      (normalised to the unit ball as in E6), propagated exactly to t = 2.0 ... 10 ms. Central prediction: x_hat from
      the measured initial state. Band sd = sd over the draws.
  On the data only (the synthetic worlds are U(1)-symmetric): the U(1)-breaking terms of S (uniform transverse field,
      flips S_51 and D5/2 emission) and the collective noise enter as a fixed offset Delta(t) = [S's full prediction
      (E25 step 1)] - [exact neutral propagation of S's U(1)-symmetric part], subtracted from the window data before
      the fit and added to the central prediction and to every draw. The uncertainty of the uniform transverse field
      enters every draw through the linear response of the full model at S (finite differences of +5 rad/s in h_x
      and in h_y, full Lindblad prediction as in E25), with (h_x, h_y) drawn from the covariance of 24 shot-resampled
      least-squares fits of the data (E25's resampling, flips and emission fixed).
Choices made before this commit, on synthetic data only (one world of arm Y15 with its own seed, drift on and off;
  nothing of it enters the results):
  - Route checks: the charge-0 block propagation agrees with the E11f sector propagator to 1.6e-15; a uniform z field
    of 300 rad/s changes the neutral classes by 1.3e-15; the setting-averaged drift world with collective noise gives
    the noise-free neutral trajectories to 1.9e-15 (Lemma G.0 and the drift argument above, numerically).
  - The prior scale. With tau_g by type-II maximum likelihood (EM), tau for d = 1 fell from 100 to 16 rad/s in 20
    iterations without converging (the world's currents have rms 101 rad/s), and the band was too narrow: factors
    1.47, 1.51, 1.52 (Z_i, Z_iZ_j, flip-flop) with drift, 1.06, 1.26, 1.16 without. With tau_g fixed at the
    least-squares rms the same world gave 0.97, 0.99, 1.04 with drift and 0.89, 1.05, 1.00 without, converging in 6-7
    iterations (about 250 s). Reading (not tested separately; consistent with E27b, where the Y15 currents moved the
    late record less than all prior draws): the Y15 currents lie mostly in directions the window does not identify,
    so the evidence, which reads the identified directions, underestimates their scale. The least-squares rms is kept
    as the prior scale; it is the scale of current configurations that the integral method does not rule out.
  - The drift's correlated errors across the pairs of one setting are not in the weights; the same world gave
    calibrated factors with and without drift, so no correlated weighting is added.
Step A (data): T and its band; factors of the neutral classes on t > 1.5 ms.
Step B (synthetic, end to end; the confirmatory part, blind to the data's late record). The E11f control truth (Y15
  couplings and z fields, quasi-static collective noise beta_s = 38.2 rad/s per ms, no transverse fields, flips or
  emission) with a share sigma_d^2 / 2 of beta_s drifting across the settings, sigma_d = 216.9 rad/s (E27 step A),
  sampled like the data (200 settings x 200 shots, both measured initial states).
  arm "redrawn" (confirmatory): R = 48 realisations, each with its own currents D ~ N(0, tau*^2) per pair, tau* the
      data's least-squares current rms per distance group (104.9, 105.3, 71.0 rad/s, E27 step A). The realisations do
      not share a truth, so the pooled factor of a calibrated band concentrates near 1 (E27b: a shared truth does not).
  arm "Y15" (reported only): 8 realisations with the control truth's own currents (shared truth, as E25 and E27).
  Every realisation: least squares (start), T, its band, and the exact late neutral truth.
Fixed before the run:
  rule S3 (step B, arm "redrawn"): for each neutral class (Z_i, Z_iZ_j, flip-flop), the pooled factor
      sqrt(sum (central - truth)^2 / sum band sd^2) over the 48 realisations, both states, t = 2.0 ... 10 ms and the
      class's combinations, lies in [0.80, 1.25] (two-sided: a band too wide fails like one too narrow).
  rule B'' (step A): Holm-Bonferroni at familywise 5% over the three neutral classes, one-sided, with p-values of T's
      data factors (band plus data shot noise) from their null (pooled factors of draws from the band covariance plus
      shot noise, 4000 draws); it passes if no class is rejected.
  guard (step A): the late temporal variance (t >= 4 ms) of T's central neutral prediction, averaged over the six
      class x state cells, is >= 0.8 times that of Xb15uf as learned (E20).
  Verdict: the procedure is validated if rule S3 passes; the data are consistent with it if rule B'' passes and the
      guard holds.
  Plan fixed with the rules: if S3 passes and B'' passes, T and its band become the method for the late neutral
      predictions. If S3 passes and B'' rejects a class, step 3 of E25 (an initial slip, run parameters with priors)
      is the next test. If S3 fails, no further estimator is built for the late neutral band; the manuscript reports
      that a validated late neutral band was not achieved.
  Reported with them: for S3, the null distribution of the pooled factor (one held-out draw of each realisation's own
      band as the truth, 2000 draws), its realisation-bootstrap interval, the coverage of the nominal 95% interval
      (|central - truth| <= 1.96 sd) and the factor per time block (2-4, 4.5-7, 7.5-10 ms); for arm Y15, the pooled
      factor, bias rms / band rms and spread / band rms; parameter level: coupling scale (sum J J_T / sum J_T^2) and
      J_nn error of least squares and T, the slope of the learned currents against the true ones (T and least
      squares), tau against tau*, chi^2 / nu and the Birge factor; on the data: lambda_T = sum J_T J_LS /
      sum J_LS^2 against S (1.045), S_D (1.054) and the late-data values of E23d (1.069, 1.104), tau, chi^2 / nu,
      the factors of S (E25 band) and S_D (E27 band), and the pooled neutral z-score per time.
Limits stated before the run: T fits only the neutral sector; the charged classes keep S (or S_D). On the data the
  U(1)-breaking terms enter as an offset computed at S's symmetric part, which neglects their interaction with the
  change from S to T (second order). The truth of step B is U(1)-symmetric, so step B does not test the offset.
  Correlations between combinations that share settings are not in the weights; step B includes them through the
  exact shot sampling. The t = 0 per-setting excess of the data is not modelled.
Status: the late record of the data has been used to shape earlier estimators (E24b, E25), not T: T uses only the
  window, and its rules and design were fixed before any fit of T to the data.
Run: `--data` (step A), `--synthetic` (step B), `--analyse`; `--smoke` runs one Y15-arm realisation with its own seed
  and saves nothing (design check). Workers: E29_WORKERS (default 10).
"""

import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from common import CAT, INK, MUTED, RESULTS, save_figure, save_json, style  # noqa: E402
from e6_ion_learning import N, PAIRS, ansatz, observables, per_setting_tables  # noqa: E402
from e6c_validation import load  # noqa: E402
from e11_model_selection import EMISSION_D52  # noqa: E402
from e11f_nonmarkov_truths import Propagator, hamiltonian, sectors  # noqa: E402
from e14_shot_calibration import truth_theta  # noqa: E402
from e20_flip_models import S_51  # noqa: E402
from e24_eiv_correction import N_WIN, NEUTRAL, T_LEARN, data_inputs, holm, strings_list  # noqa: E402
from e25_structured_eiv import class_matrices, indices, predict, solve, to_theta  # noqa: E402
from e27_marginal_band import BETA_TRUTH, K0, World, current_prior, current_scale, data_excess, drift_sigma  # noqa: E402
from qerr.pauli import expectation_tables, expectations  # noqa: E402
from qerr.shots import resample_shots  # noqa: E402

R_RED, R_Y15, K_BAND, N_NULL_S, N_NULL_B, R_XY, SHOTS = 48, 8, 48, 2000, 4000, 24, 200
EPS, C_H, TAU_FLOOR, MAX_IT, DH = 1e-3, 1.0, 1e-3, 20, 5e-3  # rad/ms
MARGIN = (0.80, 1.25)
WORKERS = int(os.environ.get("E29_WORKERS", "10"))
PARTS = RESULTS / "e29_parts"
SEEDS = {"redrawn": 290000, "Y15": 295000, "smoke": 299000, "band": 291000, "xy": 292000}

# ---------------- the neutral sector ----------------
OBS = observables()
CS = class_matrices()
CN = np.hstack([CS[n] for n in NEUTRAL])                       # 435 x 145: Z_i (10), Z_iZ_j (45), flip-flop (90)
USED = np.flatnonzero(np.abs(CN).sum(axis=1) > 0)
NSTR = [OBS[m] for m in USED]                                  # the 235 strings of the neutral combinations
CNS = CN[USED]
SLICE = {"q=0  Z_i": slice(0, 10), "q=0  Z_iZ_j": slice(10, 55), "q=0  flip-flop": slice(55, 145)}
SINGLES = [((i, a),) for i in range(N) for a in "XYZ"]
PERM, SLICES = sectors()
INV = np.argsort(PERM)
_IDX, _PH = expectation_tables(NSTR, N)
IDXP, PHP = INV[_IDX[:, PERM]], _PH[:, PERM]                  # tables in the sector-sorted basis
FF, CU, _ = indices()
_K = {p: m for m, p in enumerate(ansatz().h_terms)}
ZF = np.array([_K[((i, "Z"),)] for i in range(N)])
XF = np.array([_K[((i, "X"),)] for i in range(N)])
YF = np.array([_K[((i, "Y"),)] for i in range(N)])
DIST = np.array([j - i for i, j in PAIRS])
GROUP = np.where(DIST == 1, 0, np.where(DIST == 2, 1, 2))
NN = np.flatnonzero(DIST == 1)
NX = 100


def theta_from(x):
    """x = (J, D, h) in rad/ms -> the 210 Hamiltonian coefficients of the U(1)-symmetric part."""
    th = np.zeros(K0)
    th[FF[:, 0]] = th[FF[:, 1]] = 0.5 * x[:45]
    th[CU[:, 0]], th[CU[:, 1]] = 0.5 * x[45:90], -0.5 * x[45:90]
    th[ZF] = x[90:100]
    return th


def x_from(theta):
    return np.r_[theta[FF].sum(axis=1), theta[CU[:, 0]] - theta[CU[:, 1]], theta[ZF]]


def product_rho(bloch):
    """Product state of the Bloch vectors (N, 3), each normalised to the unit ball (as e6_ion_learning.product_state)."""
    rho = np.array([[1.0 + 0j]])
    for r in bloch:
        r = r / max(1.0, float(np.linalg.norm(r)))
        rho = np.kron(rho, 0.5 * np.array([[1 + r[2], r[0] - 1j * r[1]], [r[0] + 1j * r[1], 1 - r[2]]]))
    return rho


class Neutral:
    """Exact neutral combinations (2 states, times, 145) of U(1)-symmetric Hamiltonians from two product states."""

    def __init__(self, bloch):
        self.blocks = []
        for b in bloch:
            rp = product_rho(b)[np.ix_(PERM, PERM)]
            self.blocks.append([rp[sl, sl].copy() for sl in SLICES])

    def __call__(self, x, tgrid):
        H = hamiltonian(np.r_[theta_from(x), np.zeros(ansatz().n_params - K0)]).toarray()[np.ix_(PERM, PERM)]
        eig = [np.linalg.eigh(H[sl, sl]) for sl in SLICES]
        out = np.empty((2, len(tgrid), CNS.shape[1]))
        full = np.zeros((2**N, 2**N), complex)
        rows = np.arange(2**N)[None, :]
        for s, blocks in enumerate(self.blocks):
            tl = [V.conj().T @ r @ V for (_, V), r in zip(eig, blocks)]
            for ti, t in enumerate(tgrid):
                for (w, V), r, sl in zip(eig, tl, SLICES):
                    p = np.exp(-1j * w * t)
                    full[sl, sl] = V @ (p[:, None] * r * p.conj()[None, :]) @ V.conj().T
                out[s, ti] = (PHP * full[rows, IDXP]).sum(axis=1).real @ CNS
        return out


def neutral_reference(x, bloch, tgrid):
    """The same with the full sector propagator of E11f (route check)."""
    H = hamiltonian(np.r_[theta_from(x), np.zeros(ansatz().n_params - K0)]).toarray()
    idx, ph = expectation_tables(NSTR, N)
    return np.array([[expectations(P(t), idx, ph) @ CNS for t in tgrid]
                     for P in (Propagator(H, product_rho(b), (PERM, SLICES)) for b in bloch)])


# ---------------- data of one data set ----------------
def neutral_data(shots, bases):
    """Window neutral combinations (2, N_WIN, 145), their standard errors, and the t = 0 Bloch vectors and sem (2, N, 3)."""
    tab = per_setting_tables(shots, bases, NSTR + [p for p in SINGLES if p not in NSTR])
    mean = np.empty((2, N_WIN, len(NSTR)))
    var = np.empty_like(mean)
    single_m, single_e = {}, {}
    for p, (idx, per) in tab.items():
        n = len(idx)
        for s, m in enumerate(per):
            mu = m.mean(axis=0)
            e2 = np.maximum(m.var(axis=0, ddof=1) / n, np.clip(1.0 - mu**2, 0.0, None) / (n * SHOTS))
            if p in NSTR:
                k = NSTR.index(p)
                mean[s, :, k], var[s, :, k] = mu, e2
            if p in SINGLES:
                single_m[(s, p)], single_e[(s, p)] = mu[0], np.sqrt(e2[0])
    y = mean @ CNS
    sem = np.sqrt(var @ CNS**2)
    bloch = np.array([[[single_m[(s, ((i, a),))] for a in "XYZ"] for i in range(N)] for s in (0, 1)])
    bsem = np.array([[[single_e[(s, ((i, a),))] for a in "XYZ"] for i in range(N)] for s in (0, 1)])
    return y, sem, bloch, bsem


def data_bloch():
    """(2, N, 3) t = 0 single-ion means of the data: the initial state of every synthetic truth (as E11f, E27)."""
    times, s1, s2, bases = load()
    return neutral_data((s1[:N_WIN], s2[:N_WIN]), bases)[2]


# ---------------- estimator T ----------------
def prior_precision(tau, h_mean_weight=1.0 / C_H**2):
    P = np.zeros((NX, NX))
    P[45:90, 45:90] = np.diag(1.0 / tau[GROUP] ** 2)
    P[90:, 90:] = h_mean_weight * np.ones((N, N)) / N   # weak prior on the mean z field only
    return P


def log_evidence(log_tau, x, m, r, A, Jr):
    """Laplace log evidence of the prior scales at the linearisation (x, whitened residual r, A = Jw^T Jw, Jr = Jw^T r)."""
    tau = np.exp(log_tau)
    P = prior_precision(tau)
    H = A + P
    g = Jr - P @ (x - m)
    val = r @ r + (x - m) @ P @ (x - m) - g @ np.linalg.solve(H, g)
    return float(-0.5 * val - 0.5 * np.linalg.slogdet(H)[1] - np.sum(np.log(tau[GROUP])))


def fit_t(model, y, sem, x_start, tau_start, tw, log=None, tau_mode="fixed"):
    """Estimator T: (x_hat, Sigma, tau_hat, info dict). tau_mode: "evidence" (type-II ML), "em" or "fixed"."""
    from scipy.optimize import minimize
    w = 1.0 / sem.ravel()
    m = x_start.copy()
    m[45:90] = 0.0
    tau = np.maximum(np.asarray(tau_start, float), TAU_FLOOR)
    x = x_start.copy()
    x[45:90] = 0.0
    f = model(x, tw).ravel()
    em = tau_mode != "fixed"

    def objective(xx, ff, P):
        r = (y.ravel() - ff) * w
        return float(r @ r + (xx - m) @ P @ (xx - m)), r

    P = prior_precision(tau)
    obj, r = objective(x, f, P)
    mu, it, step = 1e-3, 0, np.inf
    for it in range(1, MAX_IT + 1):
        Jac = np.empty((f.size, NX))
        for k in range(NX):
            e = np.zeros(NX)
            e[k] = EPS
            Jac[:, k] = (model(x + e, tw).ravel() - f) / EPS
        Jw = Jac * w[:, None]
        A = Jw.T @ Jw
        if tau_mode == "evidence" and it >= 2:
            Jr = Jw.T @ r
            lo, hi = np.log(TAU_FLOOR), np.log(1.0)
            res = minimize(lambda lt: -log_evidence(lt, x, m, r, A, Jr), np.log(tau), method="L-BFGS-B",
                           bounds=[(lo, hi)] * 3)
            new = np.exp(res.x)
            dtau = float(np.max(np.abs(new / tau - 1.0)))
            tau = new
            P = prior_precision(tau)
            obj, r = objective(x, f, P)
        elif tau_mode == "em" and it >= 3:
            Sig = np.linalg.inv(A + P)
            new = np.array([np.sqrt(np.mean(x[45:90][GROUP == g] ** 2 + np.diag(Sig)[45:90][GROUP == g]))
                            for g in range(3)])
            new = np.maximum(new, TAU_FLOOR)
            dtau = float(np.max(np.abs(new / tau - 1.0)))
            tau = new
            P = prior_precision(tau)
            obj, r = objective(x, f, P)
        else:
            dtau = np.inf
        g = Jw.T @ r - P @ (x - m)
        for _ in range(12):
            H = A + P
            delta = np.linalg.solve(H + mu * np.diag(np.diag(H)), g)
            f_new = model(x + delta, tw).ravel()
            obj_new, r_new = objective(x + delta, f_new, P)
            if obj_new < obj:
                x, f, obj, r = x + delta, f_new, obj_new, r_new
                mu = max(mu / 3.0, 1e-7)
                break
            mu *= 4.0
        step = float(np.max(np.abs(delta)))
        if log:
            log(f"  iteration {it}: objective {obj:.1f}, step {step * 1e3:.3f} rad/s, tau {np.round(tau * 1e3, 1)} rad/s")
        if step < 1e-4 and (dtau < 0.01 or not em):
            break
    Jac = np.empty((f.size, NX))
    for k in range(NX):
        e = np.zeros(NX)
        e[k] = EPS
        Jac[:, k] = (model(x + e, tw).ravel() - f) / EPS
    Jw = Jac * w[:, None]
    A = Jw.T @ Jw
    Sig0 = np.linalg.inv(A + P)
    chi2 = float(r @ r)
    nu = float(f.size - np.trace(A @ Sig0))
    birge = max(1.0, chi2 / nu)
    return x, birge * Sig0, tau, {"chi2": chi2, "nu": nu, "birge": birge, "iterations": it, "last step": step}


def band_draws(x_hat, Sig, bloch, bsem, tl, rng, offset=None):
    """(K_BAND, 2, len(tl), 145) posterior-predictive draws and the central prediction (2, len(tl), 145)."""
    L = np.linalg.cholesky(0.5 * (Sig + Sig.T) + 1e-14 * np.eye(NX))
    central = Neutral(bloch)(x_hat, tl)
    draws = np.empty((K_BAND, *central.shape))
    for k in range(K_BAND):
        xk = x_hat + L @ rng.normal(size=NX)
        bk = bloch + bsem * rng.normal(size=bloch.shape)
        draws[k] = Neutral(bk)(xk, tl)
    if offset is not None:
        central, draws = central + offset, draws + offset[None]
    return central, draws


# ---------------- step B: synthetic ----------------
def truth_x(arm, r):
    x = x_from(truth_theta()[:K0])
    if arm == "redrawn":
        tau_star = current_prior()[1]
        x[45:90] = np.random.default_rng(SEEDS[arm] + 13 * r).normal(size=45) * tau_star[GROUP]
    return x


def syn_job(args):
    arm, r = args
    path = PARTS / f"{arm}_{r}.npz"
    if path.exists():
        return dict(np.load(path))
    PARTS.mkdir(exist_ok=True)
    out = realisation(arm, r)
    np.savez_compressed(path, **out)
    return out


def realisation(arm, r, drift=True, tau_mode="fixed", log=None):
    t0 = time.time()
    times, s1, s2, bases = load()
    tw, strings, tl = times[:N_WIN], strings_list(), times[N_WIN:]
    xt = truth_x("Y15" if arm == "smoke" else arm, r)
    sig = drift_sigma(data_excess()) if drift else 0.0
    W = World(theta_from(xt), 0.0, BETA_TRUTH - 0.5 * sig**2, sig, tw)
    rng = np.random.default_rng(SEEDS[arm] + 7907 * r + 1)
    shots = W.shots(rng, bases)
    d, dt = W.defects(strings, T_LEARN)
    th_ls = to_theta(solve(shots, bases, strings, tw, d, dt, 0.0, 0.0), 0.0)[0]
    y, sem, bloch, bsem = neutral_data(shots, bases)
    model = Neutral(bloch)
    x_hat, Sig, tau, info = fit_t(model, y[:, 1:], sem[:, 1:], x_from(th_ls), current_scale(th_ls)[1], tw[1:], log, tau_mode)
    band_seed = SEEDS["band"] + 31 * r + {"redrawn": 0, "Y15": 7, "smoke": 11}[arm]
    central, draws = band_draws(x_hat, Sig, bloch, bsem, tl, np.random.default_rng(band_seed))
    truth = Neutral(data_bloch())(xt, tl)
    out = dict(x_hat=x_hat, Sigma=Sig, tau=tau, x_true=xt, x_ls=x_from(th_ls), central=central, truth=truth,
               draws=draws.astype(np.float32), bloch=bloch, chi2=info["chi2"], nu=info["nu"], birge=info["birge"],
               iterations=info["iterations"], seconds=time.time() - t0, y=y, sem=sem, x_start=x_from(th_ls))
    print(f"{arm} realisation {r}: {time.time() - t0:.0f} s, {info['iterations']} iterations, "
          f"chi2/nu {info['chi2'] / info['nu']:.2f}", flush=True)
    return out


# ---------------- step A: the data ----------------
_W = {}


def _init_xy(inputs):
    _W.update(inputs=inputs)


def xy_job(k):
    """Uniform transverse field (h_x, h_y) of one shot-resampled least-squares fit of the data (E25's resampling)."""
    shots, bases, strings, tw, d, dt = _W["inputs"]
    rng = np.random.default_rng(SEEDS["xy"] + 7919 * k)
    rep = tuple(resample_shots(s, rng) for s in shots)
    th = to_theta(solve(rep, bases, strings, tw, d, dt, S_51, EMISSION_D52), EMISSION_D52)[0]
    return np.array([th[XF].mean(), th[YF].mean()])


def step_a():
    times, s1, s2, bases = load()
    tw, tl = times[1:N_WIN], times[N_WIN:]
    e25 = np.load(RESULTS / "e25_step1.npz")
    th_s, b_s, cen_s = e25["theta"], float(e25["beta"]), e25["central"]
    y, sem, bloch, bsem = neutral_data((s1[:N_WIN], s2[:N_WIN]), bases)
    model = Neutral(bloch)
    delta = cen_s @ CN - model(x_from(th_s[:K0]), times)            # U(1)-breaking terms and noise of S, (2, 21, 145)
    th_ls = to_theta(np.load(RESULTS / "e24_step2.npz")["phi0"], EMISSION_D52)[0]
    t0 = time.time()
    x_hat, Sig, tau, info = fit_t(model, y[:, 1:] - delta[:, 1:N_WIN], sem[:, 1:], x_from(th_ls[:K0]),
                                  current_prior()[1], tw, log=lambda s: print(s, flush=True))
    print(f"step A fit: {time.time() - t0:.0f} s, chi2/nu {info['chi2'] / info['nu']:.2f}", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_xy, initargs=(data_inputs(),)) as pool:
        hxy = np.array(list(pool.map(xy_job, range(R_XY))))
    jobs = []
    for tag, F in (("e29_hx", XF), ("e29_hy", YF)):
        th = th_s.copy()
        th[F] += DH
        jobs += [(tag, th, b_s, s, S_51, EMISSION_D52) for s in (0, 1)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        preds = list(pool.map(predict, jobs))
    Rx = (np.stack(preds[:2]) - cen_s) @ CN / DH
    Ry = (np.stack(preds[2:]) - cen_s) @ CN / DH
    rng = np.random.default_rng(SEEDS["band"] + 999)
    central, draws = band_draws(x_hat, Sig, bloch, bsem, tl, rng, offset=delta[:, N_WIN:])
    dh = rng.normal(size=(K_BAND, 2)) @ np.linalg.cholesky(np.cov(hxy.T)).T
    draws = draws + dh[:, 0, None, None, None] * Rx[None, :, N_WIN:] + dh[:, 1, None, None, None] * Ry[None, :, N_WIN:]
    np.savez_compressed(RESULTS / "e29_stepA.npz", x_hat=x_hat, Sigma=Sig, tau=tau, chi2=info["chi2"], nu=info["nu"],
                        birge=info["birge"], iterations=info["iterations"], central=central, draws=draws,
                        delta=delta, Rx=Rx, Ry=Ry, hxy=hxy, x_ls=x_from(th_ls[:K0]), x_s=x_from(th_s[:K0]),
                        bloch=bloch, bsem=bsem, y=y, sem=sem)


# ---------------- step B ----------------
def step_b():
    jobs = [("redrawn", r) for r in range(R_RED)] + [("Y15", r) for r in range(R_Y15)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(syn_job, jobs))
    print(f"step B: {len(jobs)} realisations, {time.time() - t0:.0f} s", flush=True)


# ---------------- analysis ----------------
def route_checks():
    times, s1, s2, bases = load()
    b = data_bloch()
    x = x_from(truth_theta()[:K0])
    tg = times[[1, 10]]
    out = {"neutral propagation against the E11f sector propagator, max |difference|":
           float(np.abs(Neutral(b)(x, tg) - neutral_reference(x, b, tg)).max())}
    xs = x.copy()
    xs[90:] += 0.3
    out["uniform z-field shift of 300 rad/s, max |difference| of the neutral classes"] = float(
        np.abs(Neutral(b)(xs, tg) - Neutral(b)(x, tg)).max())
    sig = drift_sigma(data_excess())
    W = World(theta_from(x), 0.0, BETA_TRUTH - 0.5 * sig**2, sig, times[:N_WIN])
    traj = W.trajectory(NSTR, times[1:N_WIN])
    out["setting-averaged drift world against the neutral propagation (window), max |difference|"] = float(
        max(np.abs(traj[s] @ CNS - Neutral(b)(x, times[1:N_WIN])[s]).max() for s in (0, 1)))
    from e9_symmetry_data import extended_ansatz
    out["extended ansatz shares the 210 Hamiltonian terms"] = bool(extended_ansatz().h_terms[:K0] == ansatz().h_terms)
    return out


def class_sums(p, cls, tsel=slice(None)):
    """Per-realisation sums: (central - truth)^2, band variance, held-out draw deviations (K,), coverage fraction."""
    sl = SLICE[cls]
    c, t = p["central"][:, tsel, sl], p["truth"][:, tsel, sl]
    d = p["draws"][:, :, tsel, sl].astype(float)
    var = d.var(axis=0, ddof=1)
    dev = c - t
    return (float(np.sum(dev**2)), float(np.sum(var)), np.sum((d - c[None]) ** 2, axis=(1, 2, 3)),
            float(np.mean(np.abs(dev) <= 1.96 * np.sqrt(var))))


def parameter_level(parts):
    xt = np.array([p["x_true"] for p in parts])
    out = {}
    JT = xt[0, :45]
    n = len(parts)
    for est, key in (("least squares", "x_ls"), ("T", "x_hat")):
        x = np.array([p[key] for p in parts])
        sc = x[:, :45] @ JT / (JT @ JT)
        je = x[:, :45][:, NN].mean(axis=1) / JT[NN].mean() - 1.0
        Dt, Dh = xt[:, 45:90].ravel(), x[:, 45:90].ravel()
        out[est] = {"coupling scale over the truth: mean, se": [float(sc.mean()), float(sc.std(ddof=1) / np.sqrt(n))],
                    "J_nn relative error: mean, se": [float(je.mean()), float(je.std(ddof=1) / np.sqrt(n))],
                    "currents: slope against the truth (through 0), correlation": [
                        float(Dh @ Dt / (Dt @ Dt)), float(np.corrcoef(Dh, Dt)[0, 1])]}
    return out


def step_b_summary(parts, arm):
    times = load()[0]
    tl = times[N_WIN:]
    blocks = {"2.0-4.0 ms": tl <= 4.0 + 1e-9, "4.5-7.0 ms": (tl > 4.0 + 1e-9) & (tl <= 7.0 + 1e-9),
              "7.5-10 ms": tl > 7.0 + 1e-9}
    rng = np.random.default_rng(2929)
    row = {}
    R = len(parts)
    for cls in NEUTRAL:
        sums = [class_sums(p, cls) for p in parts]
        num = np.array([s[0] for s in sums])
        den = np.array([s[1] for s in sums])
        held = np.array([s[2] for s in sums])                     # (R, K)
        k = rng.integers(0, K_BAND, size=(N_NULL_S, R))
        null = np.sqrt(held[np.arange(R)[None, :], k].sum(axis=1) / den.sum())
        boot = np.array([np.sqrt(num[i].sum() / den[i].sum()) for i in rng.integers(0, R, size=(2000, R))])
        r = {"pooled factor": float(np.sqrt(num.sum() / den.sum())),
             "null 2.5%, 50%, 97.5%": np.quantile(null, [0.025, 0.5, 0.975]).tolist(),
             "realisation bootstrap 2.5%, 97.5%": np.quantile(boot, [0.025, 0.975]).tolist(),
             "coverage of the nominal 95% interval": float(np.mean([s[3] for s in sums])),
             "pooled factor per time block": {}}
        for name, m in blocks.items():
            sb = [class_sums(p, cls, np.flatnonzero(m)) for p in parts]
            r["pooled factor per time block"][name] = float(np.sqrt(sum(s[0] for s in sb) / sum(s[1] for s in sb)))
        sl = SLICE[cls]
        dev = np.array([p["central"][..., sl] - p["truth"][..., sl] for p in parts])
        band2 = np.mean([p["draws"][..., sl].astype(float).var(axis=0, ddof=1).mean() for p in parts])
        r["bias rms / band rms"] = float(np.sqrt(np.mean(dev.mean(axis=0) ** 2) / band2))
        r["spread across realisations / band rms"] = float(np.sqrt(np.mean(dev.var(axis=0, ddof=1)) / band2))
        row[cls] = r
    tau = np.array([p["tau"] for p in parts]) * 1e3
    cn = np.array([p["chi2"] / p["nu"] for p in parts])
    out = {"per neutral class": row, "parameter level": parameter_level(parts),
           "tau (least-squares rms, rad/s) per distance group: mean, sd": [[float(a), float(b)] for a, b in
                                                            zip(tau.mean(axis=0), tau.std(axis=0, ddof=1))],
           "chi^2 / nu: mean, min, max": [float(cn.mean()), float(cn.min()), float(cn.max())],
           "Birge factor: mean, max": [float(np.mean([p["birge"] for p in parts])),
                                       float(np.max([p["birge"] for p in parts]))],
           "iterations: mean, max": [float(np.mean([p["iterations"] for p in parts])),
                                     int(np.max([p["iterations"] for p in parts]))],
           "seconds per realisation: mean": float(np.mean([p["seconds"] for p in parts]))}
    if arm == "redrawn":
        out["rule S3 per neutral class"] = {c: bool(MARGIN[0] <= row[c]["pooled factor"] <= MARGIN[1]) for c in NEUTRAL}
        out["rule S3 passes"] = bool(all(out["rule S3 per neutral class"].values()))
    return out


def step_a_summary():
    z = np.load(RESULTS / "e11_model_selection.npz")
    e20 = np.load(RESULTS / "e20_flip_models.npz")
    a = np.load(RESULTS / "e29_stepA.npz")
    times = z["times"]
    data = (z["data"] @ CN)[:, N_WIN:]
    sem = np.sqrt((z["sem"] ** 2) @ CN**2)[:, N_WIN:]
    central, draws = a["central"], a["draws"]
    var = draws.var(axis=0, ddof=1)
    rng = np.random.default_rng(2930)
    dev_b = draws - draws.mean(axis=0)
    f, p, nulls = {}, [], {}
    for cls in NEUTRAL:
        sl = SLICE[cls]
        tot = var[..., sl] + sem[..., sl] ** 2
        f[cls] = float(np.sqrt(np.mean((data[..., sl] - central[..., sl]) ** 2) / np.mean(tot)))
        null = np.empty(N_NULL_B)
        for i in range(N_NULL_B):
            d = np.tensordot(rng.normal(size=K_BAND) / np.sqrt(K_BAND - 1), dev_b[..., sl], axes=1)
            d = d + rng.normal(size=d.shape) * sem[..., sl]
            null[i] = np.sqrt(np.mean(d**2) / np.mean(tot))
        nulls[cls] = null
        p.append(float(np.mean(null >= f[cls])))
    rej = holm(np.array(p))
    late = times[N_WIN:] >= 4.0 - 1e-9
    xb = (e20["pred_Xb15uf"] @ CN)[:, N_WIN:]
    cells = [float(np.mean(central[s][late][:, SLICE[c]].var(axis=0, ddof=1))
                   / np.mean(xb[s][late][:, SLICE[c]].var(axis=0, ddof=1))) for c in NEUTRAL for s in (0, 1)]
    JT, JL, JS = a["x_hat"][:45], a["x_ls"][:45], a["x_s"][:45]
    e25j = json.loads((RESULTS / "e25_structured_eiv.json").read_text(encoding="utf-8"))["step 1 (data)"]
    e27j = json.loads((RESULTS / "e27_marginal_band.json").read_text(encoding="utf-8"))["step A (data)"]
    num = sum(np.mean((data[..., SLICE[c]] - central[..., SLICE[c]]) ** 2, axis=(0, 2)) for c in NEUTRAL)
    den = sum(np.mean(var[..., SLICE[c]] + sem[..., SLICE[c]] ** 2, axis=(0, 2)) for c in NEUTRAL)
    hxy = a["hxy"] * 1e3
    out = {"factors of T with its band (t > 1.5 ms)": f,
           "factors of S with the E25 band": {c: e25j["factors of S with the new band"][c] for c in NEUTRAL},
           "factors of S_D with the E27 band": {c: e27j["factors of S_D with its band"][c] for c in NEUTRAL},
           "null of the factor: 90%, 95%": {c: np.quantile(nulls[c], [0.9, 0.95]).tolist() for c in NEUTRAL},
           "rule B'': p-values, Holm rejections": {c: [p[i], bool(rej[i])] for i, c in enumerate(NEUTRAL)},
           "rule B'' passes": bool(not rej.any()),
           "guard: late variance T / Xb15uf as learned, mean of six cells": float(np.mean(cells)),
           "guard holds": bool(np.mean(cells) >= 0.8),
           "lambda_T = sum J_T J_LS / sum J_LS^2 (S 1.045, S_D 1.054; late data 1.069, 1.104)": float(JT @ JL / (JL @ JL)),
           "sum J_T J_S / sum J_S^2": float(JT @ JS / (JS @ JS)),
           "J_nn mean (rad/s): least squares, S, T": [float(JL[NN].mean() * 1e3), float(JS[NN].mean() * 1e3),
                                                     float(JT[NN].mean() * 1e3)],
           "currents of T: rms per distance group (rad/s)": [
               float(np.sqrt(np.mean(a["x_hat"][45:90][GROUP == g] ** 2)) * 1e3) for g in range(3)],
           "tau (least-squares rms, rad/s)": (a["tau"] * 1e3).tolist(),
           "chi^2 / nu, Birge factor, iterations": [float(a["chi2"] / a["nu"]), float(a["birge"]), int(a["iterations"])],
           "uniform transverse field of the resampled fits (rad/s): mean, sd of h_x, h_y": [
               float(hxy[:, 0].mean()), float(hxy[:, 0].std(ddof=1)), float(hxy[:, 1].mean()),
               float(hxy[:, 1].std(ddof=1))],
           "band rms per class (t > 1.5 ms): T": {c: float(np.sqrt(np.mean(var[..., SLICE[c]]))) for c in NEUTRAL},
           "pooled neutral z-score per time (data, T)": {"times": times[N_WIN:].tolist(), "z": np.sqrt(num / den).tolist()}}
    out["data consistent with T (rule B'' and guard)"] = bool(out["rule B'' passes"] and out["guard holds"])
    return out, nulls


def analyse():
    red = [dict(np.load(PARTS / f"redrawn_{r}.npz")) for r in range(R_RED)]
    y15 = [dict(np.load(PARTS / f"Y15_{r}.npz")) for r in range(R_Y15)]
    allp = red + y15
    np.savez_compressed(RESULTS / "e29_stepB.npz", arm=np.array(["redrawn"] * R_RED + ["Y15"] * R_Y15),
                        **{k: np.array([p[k] for p in allp]).astype(np.float32 if k in ("central", "truth") else float)
                           for k in ("central", "truth", "x_hat", "x_true", "x_ls", "tau", "chi2", "nu", "birge")},
                        band_var=np.array([p["draws"].astype(float).var(axis=0, ddof=1) for p in allp]).astype(np.float32))
    out = {"route checks": route_checks(),
           "step B, arm redrawn (confirmatory)": step_b_summary(red, "redrawn"),
           "step B, arm Y15 (reported only)": step_b_summary(y15, "Y15")}
    if (RESULTS / "e29_stepA.npz").exists():
        out["step A (data)"] = step_a_summary()[0]
    out["verdict"] = {"procedure validated (rule S3)": out["step B, arm redrawn (confirmatory)"]["rule S3 passes"]}
    if "step A (data)" in out:
        out["verdict"]["data consistent with T (rule B'' and guard)"] = out["step A (data)"][
            "data consistent with T (rule B'' and guard)"]
    return out, red


def plot(out, red):
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.0))
    ax = axes[0]
    rb = out["step B, arm redrawn (confirmatory)"]["per neutral class"]
    yb = out["step B, arm Y15 (reported only)"]["per neutral class"]
    ax.axvspan(*MARGIN, color=CAT[2], alpha=0.10, lw=0)
    for y, n in enumerate(NEUTRAL):
        lo, _, hi = rb[n]["null 2.5%, 50%, 97.5%"]
        ax.plot([lo, hi], [y, y], "-", color=MUTED, lw=4, alpha=0.45, solid_capstyle="butt")
        ax.plot(rb[n]["pooled factor"], y, "o", color=CAT[0], ms=6, label="redrawn currents (48)" if y == 0 else None)
        ax.plot(yb[n]["pooled factor"], y + 0.22, "o", mfc="none", color=CAT[1], ms=5,
                label="Y15 currents (8, shared truth)" if y == 0 else None)
    ax.axvline(1.0, color=MUTED, lw=0.6, ls=":")
    ax.set_yticks(range(len(NEUTRAL)), NEUTRAL)
    ax.set_ylim(len(NEUTRAL) - 0.4, -0.9)
    ax.set_xlabel("pooled factor against the exact truth, t > 1.5 ms")
    ax.legend(fontsize=6.5, loc="upper right", title="green: rule S3; grey: null 2.5-97.5%", title_fontsize=6.5)
    ax.set_title("(a) step B: synthetic drift worlds", fontsize=9)
    ax = axes[1]
    if "step A (data)" in out:
        st = out["step A (data)"]
        for u, (key, label, color, mk) in enumerate((("factors of S with the E25 band", "S, E25 band", MUTED, "s"),
                                                     ("factors of S_D with the E27 band", "S_D, E27 band", INK, "D"),
                                                     ("factors of T with its band (t > 1.5 ms)", "T, E29 band", CAT[0],
                                                      "o"))):
            ax.plot([st[key][n] for n in NEUTRAL], np.arange(len(NEUTRAL)) + 0.15 * (u - 1), mk, color=color, ms=5,
                    label=label)
        q95 = [st["null of the factor: 90%, 95%"][n][1] for n in NEUTRAL]
        ax.plot(q95, np.arange(len(NEUTRAL)) + 0.15, "|", color=CAT[0], ms=12, mew=1.2, alpha=0.6,
                label="T: 95% null (rule B'')")
        ax.axvline(1.0, color=MUTED, lw=0.6, ls=":")
        ax.set_yticks(range(len(NEUTRAL)), NEUTRAL)
        ax.invert_yaxis()
        ax.set_xlabel("calibration factor on the data, t > 1.5 ms")
        ax.set_xlim(0.35, 2.85)
        ax.legend(fontsize=6.5, loc="center right")
    ax.set_title("(b) step A: data", fontsize=9)
    ax = axes[2]
    Dt = np.array([p["x_true"][45:90] for p in red]).ravel() * 1e3
    for key, label, color, ms in (("x_ls", "least squares", MUTED, 2), ("x_hat", "T", CAT[0], 3)):
        Dh = np.array([p[key][45:90] for p in red]).ravel() * 1e3
        ax.plot(Dt, Dh, ".", color=color, ms=ms, alpha=0.5, label=f"{label}: slope {Dh @ Dt / (Dt @ Dt):.2f}")
    lim = float(np.abs(Dt).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "-", color=INK, lw=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-1.6 * lim, 1.6 * lim)
    ax.set_xlabel("true current D_ij (rad/s)")
    ax.set_ylabel("learned current (rad/s)")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("(c) currents, 48 redrawn truths", fontsize=9)
    fig.suptitle("E29: the symmetric Hamiltonian from the neutral trajectories of the window (data: Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e29_neutral_trajectory")


def smoke():
    t0 = time.time()
    print(route_checks(), flush=True)
    print(f"route checks: {time.time() - t0:.0f} s", flush=True)
    p = realisation("smoke", 0, log=lambda m: print(m, flush=True))
    for cls in NEUTRAL:
        s = class_sums(p, cls)
        print(cls, "factor", round(float(np.sqrt(s[0] / s[1])), 3), "coverage", round(s[3], 3), flush=True)
    print("parameter level", parameter_level([p, p]), flush=True)
    print("tau (rad/s)", np.round(p["tau"] * 1e3, 1), "birge", p["birge"], "seconds", round(float(p["seconds"])),
          flush=True)


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        smoke()
    if "--synthetic" in sys.argv:
        step_b()
    if "--data" in sys.argv:
        step_a()
    if "--analyse" in sys.argv:
        out, red = analyse()
        save_json(RESULTS / "e29_neutral_trajectory.json", out)
        print(plot(out, red))
        for key, val in out.items():
            print(key, val, flush=True)
