"""E25: the structured errors-in-variables estimator as the learning method (preregistered)

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0), analysed
here from the public raw shots.
Background (Q28). Shot-resampling SIMEX (E24) removes the shrinkage of the couplings on synthetic data, but a
per-parameter correction inflates the current terms, which the window does not identify. A structured correction
(E24b, V3) was best on synthetic and on real data, but it was chosen after E24 on the same late record.
Estimator S (structured SIMEX), from the level-1 shot-resampling chains (linear extrapolation):
  couplings   one common scale lambda, the slope of the SIMEX-corrected over the learned flip-flop couplings, applied
              to all flip-flop terms;
  currents    (XY - YX) chosen in step 0: (a) scaled with lambda (V3 of E24b) or (b) set to zero;
  collective  gamma_c and beta from SIMEX;
  fields      as learned.
Step 0 (synthetic, parameter level, the 12 realisations of E24 step 1): the current treatment with the smaller rms
  error against the truth's currents is used. Reported for information: the best linear shrinkage (an oracle).
Step 1 (data): S from the 100 chains of E24. Its band: 12 shot-level bootstrap replicates of the learning window,
  each put through the whole procedure (20 chains) and predicted for both states (initial state fixed at the measured
  t = 0 values, as in E15). The null distribution of the factor for this band as in E23d (4000 draws).
Step 2 (synthetic, end to end; the confirmatory part, it does not use the data's late record): 8 new realisations
  of the E11f control truth, which lies inside the model class (Y15 couplings and currents, quasi-static collective
  noise; no transverse fields, flips or emission, so these fixed rates are zero in the fit and the prediction),
  learned by least squares and by S (40 chains), predicted on 1.5 < t <= 10 ms and compared with the exact truth.
Fixed before the run:
  rule S (step 2): with the step-1 band (band spread only, since the truth is exact), the pooled calibration factor
      of each neutral class over the 8 realisations stays below the (1 - 0.05/3) quantile of its null distribution
      (pooled factors of 8 independent draws from the band covariance, 4000 draws). Reported with it: least squares
      with the Xb15 band, the bias of the mean prediction, and the spread across realisations over the band.
  rule B' (step 1): Holm-Bonferroni at familywise 5% over the three neutral classes, with p-values of S's factors
      (step-1 band plus shot noise) from the step-1 null; it passes if no class is rejected.
  guard: the late temporal variance (t >= 4 ms) of the neutral classes, averaged over the six class x state cells, is
      >= 0.8 times that of Xb15uf as learned.
  Verdict: the procedure is validated if rule S passes; the data are consistent with it if rule B' passes and the
      guard holds. Step 3 (an initial slip, run parameters with priors) runs only if rule B' rejects a neutral class.
Status: S was shaped by E24b on the same late data, so step 1 is a calibration check, not a blind test. Step 2 is
  blind to the data's late record.
"""

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import gc  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

from common import CAT, INK, MUTED, RESULTS, save_figure, save_json, style  # noqa: E402
from e6_ion_learning import N, PAIRS, observables, per_setting_tables, product_state, series_from  # noqa: E402
from e6c_validation import charge_classes, load  # noqa: E402
from e9_symmetry_data import extended_ansatz  # noqa: E402
from e11_model_selection import EMISSION_D52  # noqa: E402
from e11f_nonmarkov_truths import grids, load_truth  # noqa: E402
from e14_shot_calibration import truth_theta  # noqa: E402
from e20_flip_models import S_51, factors, structure  # noqa: E402
from e24_eiv_correction import N_WIN, NEUTRAL, T_LEARN, data_inputs, holm, strings_list  # noqa: E402
from qerr.learning import gamma_matrix, integral_system, simex_extrapolate, trajectory_defect  # noqa: E402
from qerr.lindblad import OpenModel, lindblad_rhs  # noqa: E402
from qerr.pauli import expectation_tables, expectations, psparse  # noqa: E402
from qerr.shots import resample_shots, sample_spins  # noqa: E402

N_REP, CHAINS_REP, N_SYN, CHAINS_SYN, N_NULL, WORKERS, SHOTS = 12, 20, 8, 40, 4000, 6, 200
PARTS = RESULTS / "e25_parts"
_W = {}


# ---------------- the estimator ----------------
def indices():
    ext = extended_ansatz()
    k = {p: m for m, p in enumerate(ext.h_terms)}
    ff = np.array([[k[((i, "X"), (j, "X"))], k[((i, "Y"), (j, "Y"))]] for i, j in PAIRS])
    cu = np.array([[k[((i, "X"), (j, "Y"))], k[((i, "Y"), (j, "X"))]] for i, j in PAIRS])
    k0 = len(ext.h_terms)
    return ff, cu, np.arange(k0, k0 + len(ext.deph_pairs))


def to_theta(phi, emission):
    """theta of the Xb15uf structure with the given fixed emission rate (1/ms), and beta."""
    T, fix, _, _ = structure(extended_ansatz(), uniform=True)
    return T @ phi[:-1] + fix * (emission / EMISSION_D52), float(phi[-1])


def solve(shots, bases, strings, times, d, dt, flip, emission):
    """phi of the Xb15uf least-squares fit with fixed flip and emission rates (1/ms)."""
    obs, ext = observables(), extended_ansatz()
    series = series_from(per_setting_tables(shots, bases, strings))
    T, fix, fl, coll = structure(ext, uniform=True)
    A, y, _ = integral_system(obs, ext, series, times, T_LEARN, defect=d)
    At, _, _ = integral_system(obs, ext, series, times, T_LEARN, defect=dt, weight=lambda s: s)
    M = np.column_stack([A @ T, At @ coll])
    return np.linalg.lstsq(M, y - (emission / EMISSION_D52) * (A @ fix) - flip * (A @ fl), rcond=None)[0]


def chain_mean(shots, bases, strings, times, d, dt, flip, emission, rng, n):
    """Mean phi over n level-1 chains (shots of every setting and time resampled once)."""
    return sum(solve(tuple(resample_shots(s, rng) for s in shots), bases, strings, times, d, dt, flip, emission)
               for _ in range(n)) / n


def structured(phi0, phi1, emission, currents):
    """(theta, beta, lambda) of estimator S from the plain phi and the mean level-1 phi."""
    ff, cu, coll = indices()
    th0, _ = to_theta(phi0, emission)
    thl, bl = to_theta(simex_extrapolate([phi0, phi1]), emission)
    J0, Jl = th0[ff].sum(axis=1), thl[ff].sum(axis=1)
    lam = float(Jl @ J0 / (J0 @ J0))
    th = th0.copy()
    th[ff] = lam * th0[ff]
    th[cu] = lam * th0[cu] if currents == "scaled" else 0.0
    th[coll] = thl[coll]
    return th, bl, lam


# ---------------- prediction ----------------
def predict(args):
    """(21, 435) expectations of the Xb15uf-structured model from the measured product state of state s."""
    tag, theta, beta, s, flip, emission = args
    PARTS.mkdir(exist_ok=True)
    path = PARTS / f"{tag}_s{s}.npy"
    if path.exists():
        return np.load(path)
    t0 = time.time()
    times, s1, s2, bases = load()
    obs, ext = observables(), extended_ansatz()
    rho = product_state(series_from(per_setting_tables((s1, s2), bases, [((i, a),) for i in range(N) for a in "XYZ"])), s)
    H = sum(theta[m] * psparse(P, N) for m, P in enumerate(ext.h_terms) if theta[m] != 0.0).tocsr()
    model = OpenModel(H=H, Gamma=gamma_matrix(theta, ext), emission=np.full(N, emission + flip),
                      absorption=np.full(N, flip))
    D = 2**N
    gamma_c = float(theta[len(ext.h_terms)])
    Wc = OpenModel(H=sp.csr_matrix((D, D)), Gamma=np.ones((N, N)), emission=np.zeros(N))._elementwise
    idx, ph = expectation_tables(obs, N)
    out = [expectations(rho, idx, ph)]
    for a, b in zip(times[:-1], times[1:]):
        def rhs(u, y, a=a):
            X = y.reshape(D, D)
            return (lindblad_rhs(model, X) + max(beta * (a + u), -gamma_c) * Wc * X).ravel()
        sol = solve_ivp(rhs, (0.0, b - a), rho.ravel(), t_eval=[b - a], method="DOP853", rtol=1e-7, atol=1e-9)
        rho = sol.y[:, -1].reshape(D, D)
        del sol
        gc.collect()  # the solver's stage arrays sit in reference cycles (see qerr.lindblad.evolve)
        out.append(expectations(rho, idx, ph))
    out = np.array(out)
    np.save(path, out)
    print(f"{tag} state {s + 1}: {time.time() - t0:.0f} s", flush=True)
    return out


# ---------------- step 0 ----------------
def step0():
    z = np.load(RESULTS / "e24_step1.npz")
    ff, cu, _ = indices()
    tr = truth_theta()
    Dt = tr[cu[:, 0]] - tr[cu[:, 1]]
    out, errs = {}, {}
    for name in ("scaled", "zero"):
        D = []
        for p0, lv in zip(z["phi0"], z["levels"]):
            th, _, _ = structured(p0, lv[0], EMISSION_D52, name)
            D.append(th[cu[:, 0]] - th[cu[:, 1]])
        D = np.array(D)
        errs[name] = float(np.sqrt(np.mean((D - Dt) ** 2)) * 1e3)
    Ds = []
    for p0, lv in zip(z["phi0"], z["levels"]):
        th, _, _ = structured(p0, lv[0], EMISSION_D52, "scaled")
        Ds.append(th[cu[:, 0]] - th[cu[:, 1]])
    Ds = np.array(Ds)
    c = float(np.sum(Ds * Dt) / np.sum(Ds * Ds))
    out["rms error of the currents against the truth (rad/s): (a) scaled, (b) zero"] = [errs["scaled"], errs["zero"]]
    out["truth current rms (rad/s)"] = float(np.sqrt(np.mean(Dt**2)) * 1e3)
    out["oracle linear shrinkage of the scaled currents: factor, rms error"] = [
        c, float(np.sqrt(np.mean((c * Ds - Dt) ** 2)) * 1e3)]
    choice = "scaled" if errs["scaled"] < errs["zero"] else "zero"
    out["choice"] = choice
    return out, choice


# ---------------- step 1: the data ----------------
def _init_band(inputs, choice):
    _W.update(inputs=inputs, choice=choice)


def band_fit(r):
    shots, bases, strings, tw, d, dt = _W["inputs"]
    rng = np.random.default_rng(250000 + 7907 * r)
    rep = tuple(resample_shots(s, rng) for s in shots)
    p0 = solve(rep, bases, strings, tw, d, dt, S_51, EMISSION_D52)
    p1 = chain_mean(rep, bases, strings, tw, d, dt, S_51, EMISSION_D52, rng, CHAINS_REP)
    return structured(p0, p1, EMISSION_D52, _W["choice"])


def step1(choice):
    s2 = np.load(RESULTS / "e24_step2.npz")
    th, be, lam = structured(s2["phi0"], s2["levels"][0], EMISSION_D52, choice)
    inputs = data_inputs()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_band, initargs=(inputs, choice)) as pool:
        reps = list(pool.map(band_fit, range(N_REP)))
    print(f"step 1 fits: {time.time() - t0:.0f} s", flush=True)
    jobs = [(f"data_S_{choice}", th, be, s, S_51, EMISSION_D52) for s in (0, 1)]
    jobs += [(f"band_{choice}_{r}", t, b, s, S_51, EMISSION_D52) for r, (t, b, _) in enumerate(reps) for s in (0, 1)]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        preds = list(pool.map(predict, jobs))
    central = np.stack(preds[:2])
    band = np.array(preds[2:]).reshape(N_REP, 2, *central.shape[1:])
    np.savez_compressed(RESULTS / "e25_step1.npz", central=central, band=band, theta=th, beta=be, lam=lam,
                        rep_lam=np.array([x[2] for x in reps]), rep_beta=np.array([x[1] for x in reps]))


# ---------------- step 2: synthetic, end to end ----------------
def _init_syn(prob, d, dt, choice):
    _W.update(prob=prob, d=d, dt=dt, choice=choice)


def synthetic_fit(r):
    times, _, _, bases = load()
    rng = np.random.default_rng(104729 * r + 25)
    shots = []
    for p in _W["prob"]:
        sp_ = np.empty((p.shape[0], p.shape[1], SHOTS, N), np.int8)
        for t in range(p.shape[0]):
            for k in range(p.shape[1]):
                sp_[t, k] = sample_spins(p[t, k], SHOTS, rng, N)
        shots.append(sp_)
    shots, strings, tw = tuple(shots), strings_list(), times[:N_WIN]
    p0 = solve(shots, bases, strings, tw, _W["d"], _W["dt"], 0.0, 0.0)
    p1 = chain_mean(shots, bases, strings, tw, _W["d"], _W["dt"], 0.0, 0.0, rng, CHAINS_SYN)
    th0, b0 = to_theta(p0, 0.0)
    th, be, lam = structured(p0, p1, 0.0, _W["choice"])
    return (th0, b0), (th, be, lam)


def step2(choice):
    times = load()[0]
    ctrl = load_truth("control")
    fine, _ = grids(times)
    strings = strings_list()
    tw = times[:N_WIN]
    d = trajectory_defect(fine, ctrl["fine"], strings, tw, T_LEARN)
    dt = trajectory_defect(fine, ctrl["fine"], strings, tw, T_LEARN, weight=lambda s: s)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_syn,
                             initargs=([p[:N_WIN] for p in ctrl["prob"]], d, dt, choice)) as pool:
        fits = list(pool.map(synthetic_fit, range(N_SYN)))
    print(f"step 2 fits: {time.time() - t0:.0f} s", flush=True)
    jobs = []
    for r, ((th0, b0), (th, be, _)) in enumerate(fits):
        jobs += [(f"syn{r}_plain", th0, b0, s, 0.0, 0.0) for s in (0, 1)]
        jobs += [(f"syn{r}_S_{choice}", th, be, s, 0.0, 0.0) for s in (0, 1)]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        preds = np.array(list(pool.map(predict, jobs))).reshape(N_SYN, 2, 2, len(times), -1)  # (r, est, s, t, obs)
    k = {p: m for m, p in enumerate(strings)}
    cols = [k[O] for O in observables()]
    truth = np.stack([ctrl["coarse"][s][:, cols] for s in (0, 1)])
    np.savez_compressed(RESULTS / "e25_step2.npz", plain=preds[:, 0], S=preds[:, 1], truth=truth,
                        lam=np.array([f[1][2] for f in fits]), beta_S=np.array([f[1][1] for f in fits]),
                        beta_plain=np.array([f[0][1] for f in fits]))


# ---------------- analysis ----------------
def class_matrices():
    obs = observables()
    cls = dict(charge_classes(obs), **{"all 435 observables": [{m: 1.0} for m in range(len(obs))]})
    out = {}
    for name, coeffs in cls.items():
        C = np.zeros((len(obs), len(coeffs)))
        for c, co in enumerate(coeffs):
            for m, v in co.items():
                C[m, c] = v
        out[name] = C
    return out


def null_pooled(band, C, oos, n_pool, sem=None, rng=None, n=N_NULL):
    """Null draws of the (pooled) factor when the deviations follow the band covariance (+ shot noise if sem)."""
    dev_b = band - band.mean(axis=0)
    B = band.shape[0]
    sd2 = (band @ C).std(axis=0, ddof=1) ** 2 + (0.0 if sem is None else (sem**2) @ (C**2))
    den = np.mean(sd2[:, oos])
    out = np.empty(n)
    for i in range(n):
        num = 0.0
        for _ in range(n_pool):
            d = np.tensordot(rng.normal(size=B) / np.sqrt(B - 1), dev_b, axes=1)
            if sem is not None:
                d = d + rng.normal(size=sem.shape) * sem
            num += np.mean(((d @ C)[:, oos]) ** 2)
        out[i] = np.sqrt(num / n_pool / den)
    return out


def analyse(choice, out0):
    z = np.load(RESULTS / "e11_model_selection.npz")
    e20 = np.load(RESULTS / "e20_flip_models.npz")
    xb15 = np.load(RESULTS / "e15_coverage.npz")["band_Xb15"]
    s1 = np.load(RESULTS / "e25_step1.npz")
    s2 = np.load(RESULTS / "e25_step2.npz")
    times, data, sem = z["times"], z["data"], z["sem"]
    oos = times > T_LEARN + 1e-9
    late = times >= 4.0 - 1e-9
    Cs = class_matrices()
    rng = np.random.default_rng(2511)
    band, central = s1["band"], s1["central"]
    out = {"step 0 (synthetic, parameter level)": out0}
    # step 1
    f_new = factors(central, z, band, oos)
    f_old = factors(central, z, xb15, oos)
    nulls = {n: null_pooled(band, Cs[n], oos, 1, sem=sem, rng=rng) for n in NEUTRAL}
    p = np.array([float(np.mean(nulls[n] >= f_new[n])) for n in NEUTRAL])
    rej = holm(p)
    cells = [float(np.mean((central @ Cs[c])[s][late].var(axis=0, ddof=1))
                   / np.mean((e20["pred_Xb15uf"] @ Cs[c])[s][late].var(axis=0, ddof=1)))
             for c in NEUTRAL for s in (0, 1)]
    sd_ratio = {n: float(np.sqrt(np.mean(((band @ Cs[n]).std(axis=0, ddof=1) ** 2)[:, oos])
                                 / np.mean(((xb15 @ Cs[n]).std(axis=0, ddof=1) ** 2)[:, oos]))) for n in Cs}
    out["step 1 (data)"] = {
        "estimator S: lambda, beta (rad/s per ms)": [float(s1["lam"]), float(s1["beta"]) * 1e3],
        "bootstrap replicates: lambda mean, sd; beta mean, sd": [float(s1["rep_lam"].mean()), float(s1["rep_lam"].std(ddof=1)),
                                                                  float(s1["rep_beta"].mean() * 1e3),
                                                                  float(s1["rep_beta"].std(ddof=1) * 1e3)],
        "band spread, new over Xb15 (rms over t > 1.5 ms)": sd_ratio,
        "factors of S with the new band": f_new, "factors of S with the Xb15 band": f_old,
        "null of the factor (new band): 90%, 95% per neutral class": {
            n: [float(np.quantile(nulls[n], 0.9)), float(np.quantile(nulls[n], 0.95))] for n in NEUTRAL},
        "rule B': p-values, Holm rejections": {n: [float(p[i]), bool(rej[i])] for i, n in enumerate(NEUTRAL)},
        "rule B' passes": bool(not rej.any()),
        "guard: late variance S / Xb15uf as learned, mean of six cells": float(np.mean(cells)),
        "guard holds": bool(np.mean(cells) >= 0.8),
        "rule A (E20 rule, new band)": bool(all(f_new[n] <= 1.2 for n in NEUTRAL) and f_new["all 435 observables"] <= 1.15)}
    out["step 1 (data)"]["data consistent with S (rule B' and guard)"] = bool(
        out["step 1 (data)"]["rule B' passes"] and out["step 1 (data)"]["guard holds"])
    # step 2
    truth = s2["truth"]
    res = {}
    for est, P, bnd in (("least squares, Xb15 band", s2["plain"], xb15), ("S, step-1 band", s2["S"], band)):
        row = {}
        for n in list(NEUTRAL) + ["all 435 observables"]:
            C = Cs[n]
            sd2 = np.mean(((bnd @ C).std(axis=0, ddof=1) ** 2)[:, oos])
            dev = (P - truth[None]) @ C  # (r, s, t, combos)
            pooled = float(np.sqrt(np.mean(dev[:, :, oos] ** 2) / sd2))
            bias = dev.mean(axis=0)
            spread = dev.std(axis=0, ddof=1)
            row[n] = {"pooled factor": pooled,
                      "bias rms / band rms": float(np.sqrt(np.mean(bias[:, oos] ** 2) / sd2)),
                      "spread across realisations / band rms": float(np.sqrt(np.mean(spread[:, oos] ** 2) / sd2))}
        res[est] = row
    nullp = {n: null_pooled(band, Cs[n], oos, N_SYN, rng=rng) for n in NEUTRAL}
    q = 1 - 0.05 / 3
    passes = {n: bool(res["S, step-1 band"][n]["pooled factor"] <= np.quantile(nullp[n], q)) for n in NEUTRAL}
    out["step 2 (synthetic, end to end)"] = {
        "per estimator and class": res,
        "null of the pooled factor (step-1 band): 2.5%, 50%, 98.3% quantiles": {
            n: [float(np.quantile(nullp[n], 0.025)), float(np.median(nullp[n])), float(np.quantile(nullp[n], q))]
            for n in NEUTRAL},
        "lambda of S per realisation: mean, sd": [float(s2["lam"].mean()), float(s2["lam"].std(ddof=1))],
        "beta (rad/s per ms, truth 38.2): least squares mean, S mean": [float(s2["beta_plain"].mean() * 1e3),
                                                                       float(s2["beta_S"].mean() * 1e3)],
        "rule S per neutral class": passes, "rule S passes": bool(all(passes.values()))}

    def z_time(q_, bnd):
        num = sum(np.mean(((data - q_) @ Cs[c]) ** 2, axis=(0, 2)) for c in NEUTRAL)
        den = sum(np.mean((bnd @ Cs[c]).std(axis=0, ddof=1) ** 2 + (sem**2) @ (Cs[c] ** 2), axis=(0, 2)) for c in NEUTRAL)
        return np.sqrt(num / den)

    out["pooled neutral z-score per time (data)"] = {
        "times": times.tolist(), "Xb15uf as learned, Xb15 band": z_time(e20["pred_Xb15uf"], xb15).tolist(),
        "S, new band": z_time(central, band).tolist()}
    return out, nullp


def plot(out, nullp):
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.0))
    ax = axes[0]
    st2 = out["step 2 (synthetic, end to end)"]
    for y, n in enumerate(NEUTRAL):
        lo, hi = np.quantile(nullp[n], [0.025, 1 - 0.05 / 3])
        ax.plot([lo, hi], [y, y], "-", color=MUTED, lw=4, alpha=0.45, solid_capstyle="butt")
        ax.plot(st2["per estimator and class"]["least squares, Xb15 band"][n]["pooled factor"], y - 0.12, "D",
                color=INK, ms=5, label="least squares (Xb15 band)" if y == 0 else None)
        ax.plot(st2["per estimator and class"]["S, step-1 band"][n]["pooled factor"], y + 0.12, "o", color=CAT[0],
                ms=5, label="estimator S (its own band)" if y == 0 else None)
    ax.axvline(1.0, color=MUTED, lw=0.6, ls=":")
    ax.set_yticks(range(len(NEUTRAL)), NEUTRAL)
    ax.set_ylim(len(NEUTRAL) + 0.25, -0.4)  # inverted, with room for the legend below the last row
    ax.set_xlabel("pooled factor against the exact truth, 8 realisations")
    ax.legend(fontsize=6.5, loc="lower right", title="grey: 2.5-98.3% if the band were exact", title_fontsize=6.5)
    ax.set_title("(a) step 2: synthetic data, truth known", fontsize=9)
    ax = axes[1]
    st1 = out["step 1 (data)"]
    classes = list(st1["factors of S with the new band"])
    ax.plot([st1["factors of S with the Xb15 band"][n] for n in classes], np.arange(len(classes)) - 0.12, "s",
            color=MUTED, ms=5, label="S, Xb15 band")
    ax.plot([st1["factors of S with the new band"][n] for n in classes], np.arange(len(classes)) + 0.12, "o",
            color=CAT[0], ms=5, label="S, its own band")
    ax.axvline(1.0, color=MUTED, lw=0.6, ls=":")
    ax.set_yticks(range(len(classes)), classes)
    ax.invert_yaxis()
    ax.set_xlabel("calibration factor, t > 1.5 ms")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("(b) step 1: data", fontsize=9)
    ax = axes[2]
    zt = out["pooled neutral z-score per time (data)"]
    t = np.array(zt["times"])
    ax.plot(t, zt["Xb15uf as learned, Xb15 band"], "-o", color=INK, lw=1.3, ms=3, label="Xb15uf as learned, Xb15 band")
    ax.plot(t, zt["S, new band"], "-o", color=CAT[0], lw=1.3, ms=3, label="S, its own band")
    ax.axvspan(0.0, T_LEARN, color=MUTED, alpha=0.08, lw=0)
    ax.axhline(1.0, color=MUTED, lw=0.6, ls=":")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("neutral z-score (rms deviation / rms spread)")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("(c) step 1: neutral error against time", fontsize=9)
    fig.suptitle("E25: the structured errors-in-variables estimator (data: Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e25_structured_eiv")


if __name__ == "__main__":
    out0, choice = step0()
    print("step 0:", out0, flush=True)
    if "--step0" in sys.argv:
        sys.exit(0)
    if "--band" in sys.argv:
        step1(choice)
    if "--synthetic" in sys.argv:
        step2(choice)
    if "--analyse" in sys.argv:
        out, nullp = analyse(choice, out0)
        save_json(RESULTS / "e25_structured_eiv.json", out)
        print(plot(out, nullp))
        for key in ("step 1 (data)", "step 2 (synthetic, end to end)"):
            print(key, out[key], flush=True)
