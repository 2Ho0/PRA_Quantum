"""E27: a drift-aware structured estimator with the currents marginalised in its band (preregistered)

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0), analysed
here from the public raw shots.
Background (Q29, Q30). The band of estimator S (E25) measures the spread of S but not its bias. In the synthetic
control world the late neutral bias is 1.5-2.2 band widths, mostly from setting the currents (XY - YX), which the
window does not identify, to zero (E25b). On the data a slow detuning drift across the measurement settings adds a
coupling shrinkage that shot resampling cannot reproduce (E26).
Estimator S_D (fixed here). S of E25 (common SIMEX scale lambda for the flip-flop couplings, currents zero, SIMEX for
  gamma_c and beta, fields as learned), followed by one step of parametric-bootstrap bias correction in working worlds
  W_k built from S itself:
  W_k the U(1)-symmetric part of S's Hamiltonian (its uniform transverse field dropped) plus currents D_k drawn for
      every pair from the current prior below (XY = D/2, YX = -D/2), with S's collective noise, sampled like the data
      (200 settings x 200 shots, both measured initial states). A share sigma_d^2 / 2 of beta is a global detuning that
      is constant within a setting and drifts across the settings as an AR(1) sequence (lag-1 correlation 0.8, as
      measured in E26); the rest of beta and gamma_c act within every setting. Flips and emission are left out of W_k
      and of its fits.
  Why W_k carries currents (decided before this commit, on synthetic data only): a 2 x 2 diagnostic (the control
      truth with or without its currents, drift 0 or 276 rad/s, 3 realisations each, S with 10 chains) gave coupling
      scales of S over the truth of 1.010 +- 0.004 and 0.967 +- 0.018 with the currents and 0.990 +- 0.007 and
      1.001 +- 0.007 without them (no drift, drift). The drift's shrinkage needs the currents, so a world without
      currents would not see it.
  sigma_d is estimated from the data set itself: sigma_d^2 = sigma_ref^2 x mean_t [E1(t) - 1] / mean_t [E1_ref(t) - 1],
      t = 1.0 and 1.5 ms, where E1 is the mean between-setting over binomial variance of the weight-1 strings (E26
      part A) and E1_ref, sigma_ref = 276 rad/s belong to the E26 drift world. A negative value is set to 0, and
      sigma_d^2 / 2 is capped at beta. W has no other between-setting noise, so the whole excess at the times that
      drive the coupling estimate is assigned to the drift: the shrinkage comes from the excess variance of the
      regressors, whatever its source.
  Choice made before this commit, on synthetic data only: in a smoke test (3 realisations per drift level 0, 194, 250
      and 276 rad/s of the data-matched world) this estimator and one from the spread of the per-setting rotation
      angles (E26 part A, t = 0 subtracted in quadrature) both read 5-9% low with about 10% scatter. On the data they
      give 217 rad/s (194 with the t = 0 excess subtracted) and 246 rad/s. Reported as a sensitivity at parameter
      level (12 parametric fits each, no band): lambda_D and beta_D with sigma_d = 194 and 246 rad/s.
  One data set is drawn from every W_k and put through S (20 chains, the oracle defect of W_k), R in all. With c_k =
      sum J_k J_W / sum J_W^2 over the 45 pairs (J = XX + YY; the currents leave J unchanged), lambda_D = lambda_S /
      mean_k c_k; gamma_c and beta are corrected by their mean bias, x_D = x_S - (mean_k x_k - x_W). Currents stay
      zero, fields as learned.
Current prior: D ~ N(0, tau_g^2) independently per pair, tau_g the rms of the least-squares currents (XY - YX) of the
  same data set in the distance group of the pair (d = 1, d = 2, d >= 3); on the data the least-squares fit is
  Xb15uf (E24 step 2).
Band of S_D (fixed here): R = 24 replicates. Replicate k is S_D plus the deviation of the k-th parametric fit from
  the mean of the 24 (every parameter and beta), with the currents D_k of the world W_k that produced the fit. Every
  replicate is predicted for both states as in E25 (initial state at the measured t = 0 values, flips and emission
  as in E20).
Step A (data): S_D and its band.
Step B (synthetic, end to end; the confirmatory part, blind to the data's late record): a data-matched drift world.
  The E11f control truth (Y15 couplings and currents, z fields, quasi-static collective noise beta_s = 38.2 rad/s
  per ms; no transverse fields, flips or emission) with a share sigma_d^2 / 2 of beta_s as the drifting detuning,
  sigma_d the data's estimate of step A, and the rest within each setting, so that the setting average is the control
  truth. 8 realisations; S with 30 chains, then the S_D correction with R = 8 parametric data sets per realisation,
  the realisation's own sigma_d and current prior; predicted with flips and emission zero and compared with the exact
  truth.
Step C (synthetic, parameter level, reported only): the same procedure in the shot-only control world (no drift; 8
  realisations), where S_D should stay close to S.
Fixed before the run:
  rule S'' (step B): with the step-A band (band spread only, since the truth is exact), the pooled calibration factor
      of each neutral class over the 8 realisations stays below the (1 - 0.05/3) quantile of its null distribution
      (pooled factors of 8 independent draws from the band covariance, 4000 draws).
  rule B' (step A): Holm-Bonferroni at familywise 5% over the three neutral classes, with p-values of S_D's factors
      (step-A band plus shot noise) from the step-A null; it passes if no class is rejected.
  guard: the late temporal variance (t >= 4 ms) of the neutral classes of S_D's prediction, averaged over the six
      class x state cells, is >= 0.8 times that of Xb15uf as learned.
  Reported with them: the coupling scale (sum J J_T / sum J_T^2 over all pairs, J_T of the truth) and the J_nn error
  of least squares, S and S_D in steps B and C (mean, se); sigma_d per realisation against its set value; the bias
  and the spread over the band in step B; band widths against the E25 and Xb15 bands; lambda_D against the late-data
  values of E23d (1.069, 1.104); rule A.
  Verdict: the procedure is validated if rule S'' passes; the data are consistent with it if rule B' passes and the
  guard holds. Step 3 of E25 (an initial slip, run parameters with priors) runs only if rule S'' passes and rule B'
  rejects a neutral class.
Limits stated before the run: the truth's currents were learned from the data, so the current prior of the band has
  about the right size in step B by construction. Step B therefore tests the mechanics (that marginalising currents of
  the window's size covers the bias they leave, and that the drift correction removes the drift's share of the
  shrinkage), not the size of the device's real currents. The t = 0 excess of the data (preparation or readout) is
  not modelled.
Status: S was shaped by E24b on the same late data, so step A is a calibration check, not a blind test. Steps B and C
  are blind to the data's late record.
Run: `--data` (step A), `--synthetic` (steps B and C), `--sensitivity`, `--analyse`. Workers: E27_WORKERS (default 10).
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
from e6_ion_learning import N, PAIRS, ansatz, observables, per_setting_tables, product_state, series_from  # noqa: E402
from e6c_validation import load  # noqa: E402
from e11_model_selection import EMISSION_D52  # noqa: E402
from e11f_nonmarkov_truths import Propagator, hamiltonian, load_truth, sectors  # noqa: E402
from e14_shot_calibration import truth_theta  # noqa: E402
from e20_flip_models import S_51, factors  # noqa: E402
from e24_eiv_correction import N_WIN, NEUTRAL, T_LEARN, holm, strings_list  # noqa: E402
from e25_structured_eiv import (chain_mean, class_matrices, indices, null_pooled, predict, solve,  # noqa: E402
                                structured, to_theta)
from e26_setting_drift import excess_ratios, weights  # noqa: E402
from qerr import spins  # noqa: E402
from qerr.learning import refine, trajectory_defect  # noqa: E402
from qerr.pauli import expectation_tables, expectations  # noqa: E402
from qerr.shots import pauli_tensor, rotated_setting_probabilities, sample_spins  # noqa: E402

R_DATA, R_SENS, CHAINS_W, N_SYN, R_SYN, CHAINS_SYN, N_NULL, SHOTS, RHO_AR = 24, 12, 20, 8, 8, 30, 4000, 200, 0.8
SEEDS = {"a": 270000, "sub": 278000, "angle": 279000}
WORKERS = int(os.environ.get("E27_WORKERS", "10"))
PARTS = RESULTS / "e27_parts"
K0 = len(ansatz().h_terms)                                  # the 210 Hamiltonian coefficients come first
BETA_TRUTH = 2.0 * float(truth_theta()[K0]) / 1.5          # rad/ms^2: the control truth's quasi-static slope
SIGMA_REF = float(np.sqrt(2.0 * BETA_TRUTH))                # rad/ms: the drift of the E26 world (276 rad/s)
GROUPS = ("d = 1", "d = 2", "d >= 3")
_W = {}


# ---------------- worlds ----------------
class World:
    """A U(1)-covariant truth sampled like the data.

    The U(1)-symmetric Hamiltonian is propagated exactly in the magnetisation sectors from the measured product states.
    The collective noise multiplies the charge-q blocks: within every setting by exp(-q^2 (2 gamma_w t + beta_w t^2)),
    and across the settings it is a detuning delta_k S^z that is constant within setting k, Gaussian with sd sigma_d and
    AR(1) over the setting order. The setting average carries exp(-q^2 (2 gamma_w t + (beta_w + sigma_d^2 / 2) t^2)),
    the Xb15 form with beta = beta_w + sigma_d^2 / 2.
    """

    def __init__(self, theta_h, gamma_w, beta_w, sigma_d, times):
        _, s1, s2, bases = load()
        singles = [((i, a),) for i in range(N) for a in "XYZ"]
        series = series_from(per_setting_tables((s1, s2), bases, singles))
        th = np.zeros(ansatz().n_params)
        th[:K0] = np.asarray(theta_h, float)[:K0]
        H = hamiltonian(th).toarray()
        m = spins.total_sz(N)
        comm = float(np.abs(H * (m[None, :] - m[:, None])).max())
        assert comm < 1e-9, f"the Hamiltonian is not U(1)-symmetric ({comm:.2e})"
        self.args = (th[:K0].copy(), float(gamma_w), float(beta_w), float(sigma_d), np.asarray(times, float))
        _, self.gamma_w, self.beta_w, self.sigma_d, self.times = self.args
        self.q2 = (m[:, None] - m[None, :]) ** 2
        self.props = [Propagator(H, product_state(series, s), sectors()) for s in (0, 1)]
        self.E = [[pauli_tensor(P(t) * np.exp(-self.q2 * self.x_within(t)), N) for t in self.times]
                  for P in self.props]

    def x_within(self, t):
        return 2.0 * self.gamma_w * t + self.beta_w * t * t

    def x_mean(self, t):
        return self.x_within(t) + 0.5 * self.sigma_d**2 * t * t

    def trajectory(self, strings, grid):
        """Setting-averaged expectations of `strings` on `grid`, per initial state."""
        idx, ph = expectation_tables(strings, N)
        return [np.array([expectations(P(t) * np.exp(-self.q2 * self.x_mean(t)), idx, ph) for t in grid])
                for P in self.props]

    def defects(self, strings, t_end, weighted=True):
        """Oracle quadrature defects of the integral method on [0, t_end] (unweighted and, if asked, t-weighted)."""
        tw = self.times[self.times <= t_end + 1e-9]
        fine = refine(tw, 4)
        traj = self.trajectory(strings, fine)
        d = trajectory_defect(fine, traj, strings, tw, t_end)
        if not weighted:
            return d
        return d, trajectory_defect(fine, traj, strings, tw, t_end, weight=lambda s: s)

    def shots(self, rng, bases):
        """(2 states) int8 spins (times, settings, SHOTS, N); every setting rotated by its own drifting detuning."""
        delta = self.sigma_d * weights(rng, len(bases), RHO_AR)  # (2, settings), rad/ms
        out = []
        for s in (0, 1):
            sp = np.empty((len(self.times), len(bases), SHOTS, N), np.int8)
            for ti, t in enumerate(self.times):
                for k in range(len(bases)):
                    p = rotated_setting_probabilities(self.E[s][ti], bases[k], delta[s, k] * t)
                    sp[ti, k] = sample_spins(p, SHOTS, rng, N)
            out.append(sp)
        return tuple(out)


def control_world(sigma_d, times):
    """The E11f control truth with a share sigma_d^2 / 2 of its quasi-static slope drifting across the settings."""
    sigma_d = min(float(sigma_d), SIGMA_REF)
    return World(truth_theta()[:K0], 0.0, BETA_TRUTH - 0.5 * sigma_d**2, sigma_d, times)


def drift_reference():
    """Weight-1 excess ratios (t index 0..3) of the E26 drift world, mean over its realisations."""
    e26 = json.loads((RESULTS / "e26_setting_drift.json").read_text(encoding="utf-8"))
    return np.array(e26["part B (synthetic, parameter level)"]["drift"][
        "between-setting / shot variance, mean over realisations"][:4])


def data_excess():
    e26 = json.loads((RESULTS / "e26_setting_drift.json").read_text(encoding="utf-8"))
    ex = e26["part A (data, learning window)"]["between-setting / shot variance (data, t = 0, 0.5, 1.0, 1.5 ms)"]
    return np.array([ex[f"weight 1, t index {t}"] for t in range(4)])


def drift_sigma(e1, subtract_t0=False):
    """sigma_d (rad/ms) from the weight-1 excess ratios e1[t index 0..3] of one data set (the rule above)."""
    ref = drift_reference()
    base = (e1[0], ref[0]) if subtract_t0 else (1.0, 1.0)
    num = np.mean([e1[2] - base[0], e1[3] - base[0]])
    den = np.mean([ref[2] - base[1], ref[3] - base[1]])
    return float(SIGMA_REF * np.sqrt(max(num, 0.0) / den))


def angle_sigma():
    """sigma_d (rad/ms) of the data from the per-setting rotation angles of E26 part A (t = 0 subtracted in quadrature)."""
    e26 = json.loads((RESULTS / "e26_setting_drift.json").read_text(encoding="utf-8"))["part A (data, learning window)"]
    v = []
    for s in (1, 2):
        sd = e26[f"state {s}: per-setting rotation angle"]["robust sd at t = 0, 0.5, 1.0 ms (rad)"]
        v += [(sd[1] ** 2 - sd[0] ** 2) / 0.5**2, (sd[2] ** 2 - sd[0] ** 2) / 1.0**2]
    return float(np.sqrt(max(np.mean(v), 0.0)))


def excess_w1(shots, bases, strings):
    ex = excess_ratios(shots, bases, strings)
    return np.array([ex[f"weight 1, t index {t}"] for t in range(4)])


# ---------------- estimators ----------------
def fit_s(shots, d, dt, rng, chains, bases, strings, tw):
    """(theta, beta, lambda) of S (flips and emission zero) and the least-squares phi."""
    p0 = solve(shots, bases, strings, tw, d, dt, 0.0, 0.0)
    p1 = chain_mean(shots, bases, strings, tw, d, dt, 0.0, 0.0, rng, chains)
    th, be, lam = structured(p0, p1, 0.0, "zero")
    return th, be, lam, p0


def w_params(theta_s, beta_s, sigma_d):
    """(gamma_c >= 0, beta >= 0, sigma_d capped at sqrt(2 beta)) of a working world built from S."""
    beta = max(float(beta_s), 0.0)
    return max(float(theta_s[K0]), 0.0), beta, min(float(sigma_d), float(np.sqrt(2.0 * beta)))


def working_world(theta_s, beta_s, sigma_d, times, D):
    """W_k: S's U(1)-symmetric Hamiltonian (transverse fields dropped) with currents D (XY - YX per pair, rad/ms), its
    gamma_c and beta, of which sigma_d^2 / 2 drifts across the settings."""
    k = {p: m for m, p in enumerate(ansatz().h_terms)}
    _, cu, _ = indices()
    th = np.array(theta_s[:K0], float)
    for i in range(N):
        th[k[((i, "X"),)]] = th[k[((i, "Y"),)]] = 0.0
    th[cu[:, 0]], th[cu[:, 1]] = 0.5 * D, -0.5 * D
    gamma, beta, sigma = w_params(theta_s, beta_s, sigma_d)
    return World(th, gamma, beta - 0.5 * sigma**2, sigma, times)


def w_fit(W, rng, bases, strings, tw, d, dt):
    th, be, _, _ = fit_s(W.shots(rng, bases), d, dt, rng, CHAINS_W, bases, strings, tw)
    return th, be


def correction(theta_s, beta_s, gamma_w, beta_w, fits):
    """S_D from S and the parametric fits [(theta_k, beta_k)] in W (whose couplings are those of theta_s)."""
    ff, _, coll = indices()
    Jw = theta_s[ff].sum(axis=1)
    c = np.array([t[ff].sum(axis=1) @ Jw / (Jw @ Jw) for t, _ in fits])
    th = theta_s.copy()
    th[ff] = theta_s[ff] / c.mean()
    th[coll] = theta_s[coll] - (np.mean([t[coll] for t, _ in fits], axis=0) - gamma_w)
    beta = float(beta_s - (np.mean([b for _, b in fits]) - beta_w))
    return th, beta, c


def current_scale(theta_ls):
    """tau per pair (rad/ms): rms of the least-squares currents (XY - YX) in the pair's distance group; (tau, tau_g, D)."""
    _, cu, _ = indices()
    D = theta_ls[cu[:, 0]] - theta_ls[cu[:, 1]]
    dist = np.array([j - i for i, j in PAIRS])
    group = np.where(dist == 1, 0, np.where(dist == 2, 1, 2))
    tau_g = np.array([np.sqrt(np.mean(D[group == g] ** 2)) for g in range(3)])
    return tau_g[group], tau_g, D


def current_prior():
    """The data's current prior, from the least-squares Xb15uf fit of E24 step 2."""
    return current_scale(to_theta(np.load(RESULTS / "e24_step2.npz")["phi0"], EMISSION_D52)[0])


def data_draws():
    """The R_DATA current draws D_k of the data's working worlds (rad/ms), shared by the correction and the band."""
    return np.random.default_rng(2727).normal(size=(R_DATA, len(PAIRS))) * current_prior()[0]


def coupling_scale(theta):
    """sum J J_T / sum J_T^2 over the 45 pairs, J_T of the control truth."""
    ff, _, _ = indices()
    JT = truth_theta()[ff].sum(axis=1)
    return float(theta[ff].sum(axis=1) @ JT / (JT @ JT))


def jnn_error(theta):
    ff, _, _ = indices()
    nn = [m for m, (i, j) in enumerate(PAIRS) if j == i + 1]
    JT = truth_theta()[ff].sum(axis=1)
    return float(theta[ff].sum(axis=1)[nn].mean() / JT[nn].mean() - 1.0)


# ---------------- step A: the data ----------------
def _init_a(theta_s, beta_s, sigma_d):
    _W.update(theta_s=theta_s, beta_s=beta_s, sigma_d=sigma_d)


def a_job(args):
    tag, k, D = args
    PARTS.mkdir(exist_ok=True)
    path = PARTS / f"{tag}_{k}.npz"
    if path.exists():
        z = np.load(path)
        return z["theta"], float(z["beta"])
    t0 = time.time()
    times, _, _, bases = load()
    tw, strings = times[:N_WIN], strings_list()
    W = working_world(_W["theta_s"], _W["beta_s"], _W["sigma_d"], tw, D)
    d, dt = W.defects(strings, T_LEARN)
    rng = np.random.default_rng(SEEDS[tag] + 7919 * k)
    th, be = w_fit(W, rng, bases, strings, tw, d, dt)
    np.savez_compressed(path, theta=th, beta=be)
    print(f"parametric fit {tag} {k}: {time.time() - t0:.0f} s", flush=True)
    return th, be


def sensitivity():
    """lambda_D, gamma_c and beta of S_D with the two other drift estimates (parameter level, R_SENS fits each)."""
    s1 = np.load(RESULTS / "e25_step1.npz")
    th_s, b_s = s1["theta"], float(s1["beta"])
    draws = data_draws()
    res = {}
    for tag, sig in (("sub", drift_sigma(data_excess(), subtract_t0=True)), ("angle", angle_sigma())):
        gw, bw, sw = w_params(th_s, b_s, sig)
        with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_a, initargs=(th_s, b_s, sig)) as pool:
            fits = list(pool.map(a_job, [(tag, k, draws[k]) for k in range(R_SENS)]))
        th_d, b_d, c = correction(th_s, b_s, gw, bw, fits)
        res[tag] = np.array([sig, sw, c.mean(), c.std(ddof=1), th_d[K0], b_d])
    np.savez_compressed(RESULTS / "e27_sens.npz", **res)


def step_a():
    s1 = np.load(RESULTS / "e25_step1.npz")
    th_s, b_s, lam_s = s1["theta"], float(s1["beta"]), float(s1["lam"])
    sig = drift_sigma(data_excess())
    gw, bw, sw = w_params(th_s, b_s, sig)
    _, tau_g, D_ls = current_prior()
    draws = data_draws()
    print(f"step A: sigma_d = {sig * 1e3:.1f} rad/s (used {sw * 1e3:.1f}), gamma_w = {gw * 1e3:.1f} rad/s, "
          f"beta = {bw * 1e3:.1f} rad/s per ms, tau_g = {np.round(tau_g * 1e3, 1)} rad/s", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_a, initargs=(th_s, b_s, sig)) as pool:
        fits = list(pool.map(a_job, [("a", k, draws[k]) for k in range(R_DATA)]))
    print(f"step A fits: {time.time() - t0:.0f} s", flush=True)
    th_d, b_d, c = correction(th_s, b_s, gw, bw, fits)
    _, cu, _ = indices()
    mean_th = np.mean([t for t, _ in fits], axis=0)
    mean_b = float(np.mean([b for _, b in fits]))
    reps = []
    for (t, b), D in zip(fits, draws):
        th = th_d + (t - mean_th)
        th[cu[:, 0]], th[cu[:, 1]] = 0.5 * D, -0.5 * D
        reps.append((th, b_d + (b - mean_b)))
    jobs = [("e27_data_central", th_d, b_d, s, S_51, EMISSION_D52) for s in (0, 1)]
    jobs += [(f"e27_band_{k}", t, b, s, S_51, EMISSION_D52) for k, (t, b) in enumerate(reps) for s in (0, 1)]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        preds = list(pool.map(predict, jobs))
    central = np.stack(preds[:2])
    band = np.array(preds[2:]).reshape(R_DATA, 2, *central.shape[1:])
    np.savez_compressed(RESULTS / "e27_stepA.npz", central=central, band=band, theta_d=th_d, beta_d=b_d, c=c,
                        theta_s=th_s, beta_s=b_s, lam_s=lam_s, sigma_hat=sig, sigma_used=sw, gamma_W=gw, beta_W=bw,
                        tau_g=tau_g, D_ls=D_ls, draws=draws, rep_beta=np.array([b for _, b in fits]),
                        rep_scale=np.array([coupling_scale(t) for t, _ in fits]))


# ---------------- steps B and C: synthetic ----------------
def _init_bc(sigma_dm):
    _W.update(sigma_dm=sigma_dm)


def bc_job(args):
    world, r = args
    PARTS.mkdir(exist_ok=True)
    path = PARTS / f"{world}_{r}.npz"
    if path.exists():
        return dict(np.load(path))
    t0 = time.time()
    times, _, _, bases = load()
    tw, strings = times[:N_WIN], strings_list()
    truth = control_world(_W["sigma_dm"] if world == "dm" else 0.0, tw)
    d, dt = truth.defects(strings, T_LEARN)
    rng = np.random.default_rng(271000 + 7907 * r + (0 if world == "dm" else 500))
    shots = truth.shots(rng, bases)
    th_s, b_s, lam_s, p0 = fit_s(shots, d, dt, rng, CHAINS_SYN, bases, strings, tw)
    sig = drift_sigma(excess_w1(shots, bases, strings))
    gw, bw, sw = w_params(th_s, b_s, sig)
    th0, b0 = to_theta(p0, 0.0)
    tau, tau_g, _ = current_scale(th0)
    fits = []
    for _ in range(R_SYN):
        W = working_world(th_s, b_s, sig, tw, rng.normal(size=len(PAIRS)) * tau)
        dW, dtW = W.defects(strings, T_LEARN)
        fits.append(w_fit(W, rng, bases, strings, tw, dW, dtW))
    th_d, b_d, c = correction(th_s, b_s, gw, bw, fits)
    out = dict(theta_s=th_s, beta_s=b_s, lam_s=lam_s, theta_d=th_d, beta_d=b_d, c=c, sigma_hat=sig, sigma_used=sw,
               theta_ls=th0, beta_ls=b0, tau_g=tau_g)
    np.savez_compressed(path, **out)
    print(f"step {'B' if world == 'dm' else 'C'} realisation {r}: {time.time() - t0:.0f} s", flush=True)
    return out


def route_check():
    """The worlds against the E11f control truth: setting-averaged trajectory and drift-free outcome distributions."""
    times = load()[0]
    tw, strings = times[:N_WIN], strings_list()
    ctrl = load_truth("control")
    fine = refine(tw, 4)
    dm = control_world(0.1, tw)  # any drift share: the setting average must not change
    traj = dm.trajectory(strings, fine)
    w0 = control_world(0.0, tw)
    from qerr.shots import setting_probabilities
    bases = load()[3]
    prob = max(float(np.abs(setting_probabilities(w0.E[s][t], bases[k]) - ctrl["prob"][s][t, k]).max())
               for s in (0, 1) for t in range(N_WIN) for k in range(0, len(bases), 37))
    return {"setting-averaged trajectory, max |difference| to the control truth": float(
        max(np.abs(traj[s] - ctrl["fine"][s]).max() for s in (0, 1))),
        "drift-free outcome distributions, max |difference| (every 37th setting)": prob}


def step_bc():
    sig = drift_sigma(data_excess())
    times = load()[0]
    jobs = [("dm", r) for r in range(N_SYN)] + [("shot", r) for r in range(N_SYN)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_bc, initargs=(sig,)) as pool:
        res = list(pool.map(bc_job, jobs))
    print(f"steps B, C fits: {time.time() - t0:.0f} s", flush=True)
    pj = [(f"e27_dm{r}", res[r]["theta_d"], float(res[r]["beta_d"]), s, 0.0, 0.0) for r in range(N_SYN) for s in (0, 1)]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        preds = np.array(list(pool.map(predict, pj))).reshape(N_SYN, 2, len(times), -1)
    ctrl = load_truth("control")
    k = {p: m for m, p in enumerate(strings_list())}
    cols = [k[O] for O in observables()]
    truth = np.stack([ctrl["coarse"][s][:, cols] for s in (0, 1)])
    keys = ("theta_s", "beta_s", "lam_s", "theta_d", "beta_d", "sigma_hat", "sigma_used", "theta_ls", "beta_ls", "tau_g")
    np.savez_compressed(RESULTS / "e27_stepB.npz", pred=preds, truth=truth, sigma_dm=sig,
                        **{f"{w}_{key}": np.array([x[key] for (ww, _), x in zip(jobs, res) if ww == w])
                           for w in ("dm", "shot") for key in keys},
                        **{f"{w}_c": np.array([x["c"] for (ww, _), x in zip(jobs, res) if ww == w])
                           for w in ("dm", "shot")})


# ---------------- analysis ----------------
def analyse():
    z = np.load(RESULTS / "e11_model_selection.npz")
    e20 = np.load(RESULTS / "e20_flip_models.npz")
    xb15 = np.load(RESULTS / "e15_coverage.npz")["band_Xb15"]
    e25 = np.load(RESULTS / "e25_step1.npz")
    a, b = np.load(RESULTS / "e27_stepA.npz"), np.load(RESULTS / "e27_stepB.npz")
    times, sem = z["times"], z["sem"]
    oos = times > T_LEARN + 1e-9
    late = times >= 4.0 - 1e-9
    Cs = class_matrices()
    rng = np.random.default_rng(2713)
    band, central = a["band"], a["central"]
    _, cu, coll = indices()
    k0 = K0
    out = {"route check": route_check()}
    # step A
    f_new = factors(central, z, band, oos)
    nulls = {n: null_pooled(band, Cs[n], oos, 1, sem=sem, rng=rng) for n in NEUTRAL}
    p = np.array([float(np.mean(nulls[n] >= f_new[n])) for n in NEUTRAL])
    rej = holm(p)
    cells = [float(np.mean((central @ Cs[c])[s][late].var(axis=0, ddof=1))
                   / np.mean((e20["pred_Xb15uf"] @ Cs[c])[s][late].var(axis=0, ddof=1)))
             for c in NEUTRAL for s in (0, 1)]
    width = lambda bb, n: float(np.mean(((bb @ Cs[n]).std(axis=0, ddof=1) ** 2)[:, oos]))
    out["step A (data)"] = {
        "drift: sigma_d estimated, used (rad/s); its share of beta_S": [
            float(a["sigma_hat"]) * 1e3, float(a["sigma_used"]) * 1e3,
            float(0.5 * a["sigma_used"] ** 2 / a["beta_W"]) if float(a["beta_W"]) > 0 else None],
        "current prior tau (rad/s) for d = 1, d = 2, d >= 3": (a["tau_g"] * 1e3).tolist(),
        "parametric correction: mean coupling ratio c (S over W), sd": [float(a["c"].mean()), float(a["c"].std(ddof=1))],
        "lambda: S, S_D (late-data values 1.069 and 1.104, E23d)": [float(a["lam_s"]),
                                                                   float(a["lam_s"] / a["c"].mean())],
        "gamma_c (rad/s): S, S_D": [float(a["theta_s"][k0]) * 1e3, float(a["theta_d"][k0]) * 1e3],
        "beta (rad/s per ms): S, S_D; replicates mean, sd": [
            float(a["beta_s"]) * 1e3, float(a["beta_d"]) * 1e3, float(a["rep_beta"].mean()) * 1e3,
            float(a["rep_beta"].std(ddof=1)) * 1e3],
        "band spread over the E25 band and over the Xb15 band (rms, t > 1.5 ms)": {
            n: [float(np.sqrt(width(band, n) / width(e25["band"], n))), float(np.sqrt(width(band, n) / width(xb15, n)))]
            for n in Cs},
        "factors of S_D with its band": f_new,
        "factors of S_D with the E25 band": factors(central, z, e25["band"], oos),
        "null of the factor: 90%, 95% per neutral class": {
            n: [float(np.quantile(nulls[n], 0.9)), float(np.quantile(nulls[n], 0.95))] for n in NEUTRAL},
        "rule B': p-values, Holm rejections": {n: [float(p[i]), bool(rej[i])] for i, n in enumerate(NEUTRAL)},
        "rule B' passes": bool(not rej.any()),
        "guard: late variance S_D / Xb15uf as learned, mean of six cells": float(np.mean(cells)),
        "guard holds": bool(np.mean(cells) >= 0.8),
        "rule A (E20 rule, own band)": bool(all(f_new[n] <= 1.2 for n in NEUTRAL)
                                            and f_new["all 435 observables"] <= 1.15)}
    out["step A (data)"]["data consistent with S_D (rule B' and guard)"] = bool(
        out["step A (data)"]["rule B' passes"] and out["step A (data)"]["guard holds"])
    # step B
    truth, P = b["truth"], b["pred"]
    row = {}
    for n in list(NEUTRAL) + ["all 435 observables"]:
        C = Cs[n]
        sd2 = width(band, n)
        dev = (P - truth[None]) @ C
        bias, spread = dev.mean(axis=0), dev.std(axis=0, ddof=1)
        row[n] = {"pooled factor": float(np.sqrt(np.mean(dev[:, :, oos] ** 2) / sd2)),
                  "bias rms / band rms": float(np.sqrt(np.mean(bias[:, oos] ** 2) / sd2)),
                  "spread across realisations / band rms": float(np.sqrt(np.mean(spread[:, oos] ** 2) / sd2))}
    nullp = {n: null_pooled(band, Cs[n], oos, N_SYN, rng=rng) for n in NEUTRAL}
    q = 1 - 0.05 / 3
    passes = {n: bool(row[n]["pooled factor"] <= np.quantile(nullp[n], q)) for n in NEUTRAL}
    out["step B (synthetic, data-matched drift world, end to end)"] = {
        "sigma_d (rad/s): set; estimated per realisation": [float(b["sigma_dm"]) * 1e3,
                                                            (b["dm_sigma_hat"] * 1e3).tolist()],
        "per neutral class (S_D, step-A band)": row,
        "null of the pooled factor: 2.5%, 50%, 98.3% quantiles": {
            n: [float(np.quantile(nullp[n], 0.025)), float(np.median(nullp[n])), float(np.quantile(nullp[n], q))]
            for n in NEUTRAL},
        "rule S'' per neutral class": passes, "rule S'' passes": bool(all(passes.values()))}
    # parameter level, steps B and C
    par = {}
    for w, label in (("dm", "step B: data-matched drift world"), ("shot", "step C: shot-only world")):
        rr = {}
        for est, key in (("least squares", "theta_ls"), ("S", "theta_s"), ("S_D", "theta_d")):
            sc = np.array([coupling_scale(t) for t in b[f"{w}_{key}"]])
            je = np.array([jnn_error(t) for t in b[f"{w}_{key}"]])
            rr[est] = {"coupling scale over the truth: mean, se": [float(sc.mean()), float(sc.std(ddof=1) / np.sqrt(len(sc)))],
                       "J_nn relative error: mean, se": [float(je.mean()), float(je.std(ddof=1) / np.sqrt(len(je)))]}
        rr["beta (rad/s per ms, truth 38.2): least squares, S, S_D"] = [
            float(b[f"{w}_beta_ls"].mean()) * 1e3, float(b[f"{w}_beta_s"].mean()) * 1e3,
            float(b[f"{w}_beta_d"].mean()) * 1e3]
        rr["sigma_d estimated (rad/s): mean, sd"] = [float(b[f"{w}_sigma_hat"].mean()) * 1e3,
                                                    float(b[f"{w}_sigma_hat"].std(ddof=1)) * 1e3]
        par[label] = rr
    out["parameter level (reported)"] = par
    sens_path = RESULTS / "e27_sens.npz"
    if sens_path.exists():
        sz = np.load(sens_path)
        out["sensitivity to the drift estimate (data, parameter level)"] = {
            label: {"sigma_d estimated, used (rad/s)": [float(sz[tag][0]) * 1e3, float(sz[tag][1]) * 1e3],
                    "c mean, sd": [float(sz[tag][2]), float(sz[tag][3])],
                    "lambda_D": float(a["lam_s"] / sz[tag][2]),
                    "gamma_c of S_D (rad/s)": float(sz[tag][4]) * 1e3, "beta of S_D (rad/s per ms)": float(sz[tag][5]) * 1e3}
            for tag, label in (("sub", "excess with the t = 0 excess subtracted"),
                               ("angle", "spread of the per-setting rotation angles"))}
    out["verdict"] = {"procedure validated (rule S'')": out["step B (synthetic, data-matched drift world, end to end)"][
        "rule S'' passes"], "data consistent with S_D (rule B' and guard)": out["step A (data)"][
        "data consistent with S_D (rule B' and guard)"]}

    def z_time(q_, bnd):
        num = sum(np.mean(((z["data"] - q_) @ Cs[c]) ** 2, axis=(0, 2)) for c in NEUTRAL)
        den = sum(np.mean((bnd @ Cs[c]).std(axis=0, ddof=1) ** 2 + (sem**2) @ (Cs[c] ** 2), axis=(0, 2)) for c in NEUTRAL)
        return np.sqrt(num / den)

    out["pooled neutral z-score per time (data)"] = {
        "times": times.tolist(), "S, E25 band": z_time(e25["central"], e25["band"]).tolist(),
        "S_D, E27 band": z_time(central, band).tolist()}
    return out, nullp


def plot(out, nullp):
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.0))
    ax = axes[0]
    st = out["step B (synthetic, data-matched drift world, end to end)"]
    for y, n in enumerate(NEUTRAL):
        lo, hi = np.quantile(nullp[n], [0.025, 1 - 0.05 / 3])
        ax.plot([lo, hi], [y, y], "-", color=MUTED, lw=4, alpha=0.45, solid_capstyle="butt")
        ax.plot(st["per neutral class (S_D, step-A band)"][n]["pooled factor"], y, "o", color=CAT[0], ms=6,
                label="S_D, E27 band" if y == 0 else None)
    ax.axvline(1.0, color=MUTED, lw=0.6, ls=":")
    ax.set_yticks(range(len(NEUTRAL)), NEUTRAL)
    ax.set_ylim(len(NEUTRAL) - 0.4, -0.9)
    ax.set_xlabel("pooled factor against the exact truth, 8 realisations")
    ax.legend(fontsize=6.5, loc="upper right", title="grey: 2.5-98.3% if the band were exact", title_fontsize=6.5)
    ax.set_title("(a) step B: data-matched drift world", fontsize=9)
    ax = axes[1]
    st = out["step A (data)"]
    classes = list(st["factors of S_D with its band"])
    e25j = json.loads((RESULTS / "e25_structured_eiv.json").read_text(encoding="utf-8"))["step 1 (data)"]
    ax.plot([e25j["factors of S with the new band"][n] for n in classes], np.arange(len(classes)) - 0.12, "s",
            color=MUTED, ms=5, label="S, E25 band")
    ax.plot([st["factors of S_D with its band"][n] for n in classes], np.arange(len(classes)) + 0.12, "o",
            color=CAT[0], ms=5, label="S_D, E27 band")
    ax.axvline(1.0, color=MUTED, lw=0.6, ls=":")
    ax.set_yticks(range(len(classes)), classes)
    ax.invert_yaxis()
    ax.set_xlabel("calibration factor, t > 1.5 ms")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("(b) step A: data", fontsize=9)
    ax = axes[2]
    par = out["parameter level (reported)"]
    for x0, (label, rr) in enumerate(par.items()):
        for u, (est, color, mk) in enumerate((("least squares", INK, "D"), ("S", MUTED, "s"), ("S_D", CAT[0], "o"))):
            m, se = rr[est]["coupling scale over the truth: mean, se"]
            ax.errorbar(x0 + 0.18 * (u - 1), m, yerr=2 * se, fmt=mk, color=color, ms=5, capsize=2,
                        label=est if x0 == 0 else None)
    ax.axhline(1.0, color=MUTED, lw=0.6, ls=":")
    ax.set_xticks([0, 1], ["data-matched drift", "shot only"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("coupling scale over the truth, mean +- 2 se")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("(c) steps B, C: parameter level", fontsize=9)
    fig.suptitle("E27: drift-aware estimator with the currents in its band (data: Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e27_marginal_band")


if __name__ == "__main__":
    if "--data" in sys.argv:
        step_a()
    if "--synthetic" in sys.argv:
        step_bc()
    if "--sensitivity" in sys.argv:
        sensitivity()
    if "--analyse" in sys.argv:
        out, nullp = analyse()
        save_json(RESULTS / "e27_marginal_band.json", out)
        print(plot(out, nullp))
        for key in ("route check", "step A (data)", "step B (synthetic, data-matched drift world, end to end)",
                    "parameter level (reported)", "verdict"):
            print(key, out[key], flush=True)
