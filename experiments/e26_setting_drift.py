"""E26: a slow detuning drift across the measurement settings, in the noise model of the learning (preregistered)

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0), analysed
here from the public raw shots.
Finding that motivates this run (diagnostics on the learning window only, part A below). The per-setting means of a
Pauli string scatter more than shot noise allows: in the data the ratio of the between-setting variance to the
binomial shot variance is 1.2-2.2 for strings with X or Y letters on t = 0.5-1.5 ms, about 1.0 for ZZ, and close to 1
at t = 0; in shot-level synthetic data it is 1.00. Fitting one global rotation angle per setting to the transverse
single-qubit data gives an angle whose spread grows linearly in time (robust sd 0.06, 0.15, 0.29 rad at 0, 0.5,
1.0 ms for state 1), whose values at 0.5 and 1.0 ms agree setting by setting (Spearman 0.87 and 0.76), and which is
correlated between consecutive settings (lag-1 Spearman about 0.8). This is a global detuning that drifts slowly over
the acquisition, with an rms of about 220-270 rad/s. The quasi-static collective noise that E11b/E14 inferred from the
data has the same size (rms 276 rad/s for beta_s = 38.2 rad/s per ms). Shot resampling (E24, E25) adds only shot
noise, so it under-represents the noise of the string averages: the shrinkage of the couplings is underestimated and
the band of E25 is too narrow where the drift acts.
Noise models for SIMEX and for the band (level 1 adds one copy of the noise):
  shot    resample the shots of every (setting, time) (E24);
  wild    setting-level wild bootstrap: every per-setting mean m_k(p, t) becomes m_k + c_p w_k (m_k - mean_k m_k),
          c_p = sqrt(n_p / (n_p - 1)), with one weight w_k per setting (all its strings and times) and initial state;
          wild-iid draws w_k independently, wild-ar from an AR(1) sequence over the setting order with lag-1
          correlation 0.8 (the measured drift correlation). Strings measured in one setting get no added noise.
Estimator: S of E25 (one common SIMEX scale for the flip-flop couplings, currents set to zero by E25 step 0, SIMEX for
gamma_c and beta, fields as learned) with the noise model chosen in part B.
Part B (synthetic, parameter level): two worlds built from the E11f control truth (Y15 couplings and currents, no
transverse fields, flips or emission; these fixed rates are zero in the fits). Drift world: the Hamiltonian evolution
of the truth with one global detuning delta_k per setting and initial state, an AR(1) sequence over the setting order
with lag-1 correlation 0.8 and rms sqrt(2 beta_s) = 276 rad/s, so that the setting average reproduces the truth's
quasi-static collective decay; 8 realisations. Shot-only world: the truth itself; 6 realisations. Each realisation:
least squares and S under the three noise models (30 level-1 chains each), oracle defect of the averaged truth.
Fixed before part B: a noise model passes if the mean relative error of J_nn of S lies within 1% in both worlds. The
primary noise model for part C is wild-ar if it passes, else wild-iid if it passes; if neither passes, part C is not
run.
Part C (data): S with the primary noise model (100 chains), predicted as in E20 for both states; its band from 12
replicates of the same noise model (each replicate put through the whole procedure with 20 chains). Fixed before
part C: rule B' and the guard of E25 with this band; rule A reported; the coupling scale reported against the
late-data values of E23d (1.069 and 1.104).
Status: the drift was found in the learning window after E24 and E25 had been designed; part B is blind to the
data's late record, part C reuses it (a calibration check, not a blind test).
"""

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from common import CAT, MUTED, ORDINAL, RESULTS, save_figure, save_json, style  # noqa: E402
from e6_ion_learning import N, PAIRS, ansatz, observables, per_setting_tables, product_state, series_from  # noqa: E402
from e6c_validation import charge_classes, load  # noqa: E402
from e9_symmetry_data import extended_ansatz  # noqa: E402
from e11_model_selection import EMISSION_D52  # noqa: E402
from e11f_nonmarkov_truths import Propagator, grids, hamiltonian, load_truth, sectors  # noqa: E402
from e14_shot_calibration import truth_theta  # noqa: E402
from e20_flip_models import S_51, factors, structure  # noqa: E402
from e24_eiv_correction import N_WIN, NEUTRAL, T_LEARN, data_inputs, holm, strings_list  # noqa: E402
from e25_structured_eiv import indices, null_pooled, predict, to_theta  # noqa: E402
from qerr.learning import integral_system, simex_extrapolate, trajectory_defect  # noqa: E402
from qerr.shots import pauli_tensor, resample_shots, rotated_setting_probabilities, sample_spins  # noqa: E402

RHO_AR, N_DRIFT, N_SHOT, CHAINS_B, CHAINS_C, N_REP, CHAINS_REP, WORKERS, SHOTS = 0.8, 8, 6, 30, 100, 12, 20, 6, 200
MODELS = ("shot", "wild-iid", "wild-ar")
PARTS = RESULTS / "e26_parts"
_W = {}


# ---------------- noise models ----------------
def weights(rng, n, rho):
    """(2, n) setting weights with unit variance: i.i.d. (rho = 0) or AR(1) over the setting order."""
    w = rng.normal(size=(2, n))
    if rho:
        for k in range(1, n):
            w[:, k] = rho * w[:, k - 1] + np.sqrt(1.0 - rho**2) * w[:, k]
    return w


def wild(tables, w):
    """Setting-level wild perturbation of per-setting tables (one weight per setting and initial state)."""
    out = {}
    for p, (idx, per) in tables.items():
        n = len(idx)
        if n < 2:
            out[p] = (idx, per)
            continue
        c = np.sqrt(n / (n - 1.0))
        out[p] = (idx, tuple(m + c * w[s][idx][:, None] * (m - m.mean(axis=0)) for s, m in enumerate(per)))
    return out


def perturb(model, shots, tables, bases, strings, rng):
    """One level-1 copy of the data under a noise model: (shots, tables)."""
    if model == "shot":
        new = tuple(resample_shots(s, rng) for s in shots)
        return new, per_setting_tables(new, bases, strings)
    w = weights(rng, len(bases), RHO_AR if model == "wild-ar" else 0.0)
    return shots, wild(tables, w)


def fit_tables(tables, times, d, dt, flip, emission):
    obs, ext = observables(), extended_ansatz()
    series = series_from(tables)
    T, fix, fl, coll = structure(ext, uniform=True)
    A, y, _ = integral_system(obs, ext, series, times, T_LEARN, defect=d)
    At, _, _ = integral_system(obs, ext, series, times, T_LEARN, defect=dt, weight=lambda s: s)
    M = np.column_stack([A @ T, At @ coll])
    return np.linalg.lstsq(M, y - (emission / EMISSION_D52) * (A @ fix) - flip * (A @ fl), rcond=None)[0]


def level1(model, shots, tables, bases, strings, times, d, dt, flip, emission, rng, n):
    return sum(fit_tables(perturb(model, shots, tables, bases, strings, rng)[1], times, d, dt, flip, emission)
               for _ in range(n)) / n


def estimator_s(phi0, phi1, emission):
    """(theta, beta, lambda) of S with the currents set to zero (E25 step 0)."""
    ff, cu, coll = indices()
    th0, _ = to_theta(phi0, emission)
    thl, bl = to_theta(simex_extrapolate([phi0, phi1]), emission)
    J0, Jl = th0[ff].sum(axis=1), thl[ff].sum(axis=1)
    lam = float(Jl @ J0 / (J0 @ J0))
    th = th0.copy()
    th[ff] = lam * th0[ff]
    th[cu] = 0.0
    th[coll] = thl[coll]
    return th, bl, lam


def jnn(theta):
    ff, _, _ = indices()
    J = theta[ff].sum(axis=1)
    return float(np.mean([J[m] for m, (i, j) in enumerate(PAIRS) if j == i + 1]))


# ---------------- part A: diagnostics of the data ----------------
def rotation_angles(sp, bases, ti, thr=0.5):
    m = sp.astype(float).mean(axis=2)
    X = np.array([m[ti, bases[:, i] == 0, i].mean() for i in range(N)])
    Y = np.array([m[ti, bases[:, i] == 1, i].mean() for i in range(N)])
    out = np.full(len(bases), np.nan)
    for k in range(len(bases)):
        g, r = [], []
        for i in range(N):
            if bases[k, i] == 0:
                g.append(-Y[i])
                r.append(m[ti, k, i] - X[i])
            elif bases[k, i] == 1:
                g.append(X[i])
                r.append(m[ti, k, i] - Y[i])
        g, r = np.array(g), np.array(r)
        if g @ g > thr:
            out[k] = (g @ r) / (g @ g)
    return out


def excess_ratios(shots, bases, strings):
    """Mean between-setting variance / binomial shot variance of the per-setting means, by weight and time."""
    tab = per_setting_tables(shots, bases, strings)
    acc = {}
    for p, (idx, per) in tab.items():
        if len(idx) < 3:
            continue
        for m in per:
            vb = m.var(axis=0, ddof=1)
            vs = np.clip(1 - m**2, 0, None).mean(axis=0) / SHOTS
            for t in range(m.shape[1]):
                if vs[t] > 1e-4:
                    acc.setdefault((len(p), t), []).append(vb[t] / vs[t])
    return {f"weight {w}, t index {t}": float(np.mean(v)) for (w, t), v in sorted(acc.items())}


def part_a():
    times, s1, s2, bases = load()
    rob = lambda q: float(1.4826 * np.median(np.abs(q - np.median(q))))
    out = {"between-setting / shot variance (data, t = 0, 0.5, 1.0, 1.5 ms)": excess_ratios(
        (s1[:N_WIN], s2[:N_WIN]), bases, strings_list())}
    for s, sp in ((0, s1), (1, s2)):
        P = {ti: rotation_angles(sp, bases, ti) for ti in (0, 1, 2)}
        ok = np.isfinite(P[1]) & np.isfinite(P[2])
        q1 = P[1][np.isfinite(P[1])]
        out[f"state {s + 1}: per-setting rotation angle"] = {
            "robust sd at t = 0, 0.5, 1.0 ms (rad)": [rob(P[t][np.isfinite(P[t])]) for t in (0, 1, 2)],
            "Spearman between t = 0.5 and 1.0 ms": float(spearmanr(P[1][ok], P[2][ok]).correlation),
            "lag-1 Spearman over the setting order at t = 0.5 ms": float(spearmanr(q1[:-1], q1[1:]).correlation)}
    return out


# ---------------- part B: synthetic worlds ----------------
def _init_b(E_t, d, dt):
    _W.update(E_t=E_t, d=d, dt=dt)


def hamiltonian_tensors():
    """Pauli tensors of the control truth's Hamiltonian evolution (no collective factor) at the window times."""
    times, s1, s2, bases = load()
    singles = [((i, a),) for i in range(N) for a in "XYZ"]
    series = series_from(per_setting_tables((s1, s2), bases, singles))
    H = hamiltonian(truth_theta()).toarray()
    return [[pauli_tensor(Propagator(H, product_state(series, s), sectors())(t), N) for t in times[:N_WIN]]
            for s in (0, 1)]


def world_shots(world, r, rng, bases):
    """Shots (2 states) of one realisation; the drift world rotates every setting by its own detuning."""
    truth = truth_theta()
    sigma = np.sqrt(2.0 * (2.0 * truth[len(ansatz().h_terms)] / 1.5)) if world == "drift" else 0.0  # rad/ms
    times = load()[0][:N_WIN]
    delta = sigma * weights(rng, len(bases), RHO_AR)  # (2, settings) in rad/ms; H_delta = delta S^z
    shots = []
    for s in (0, 1):
        sp = np.empty((N_WIN, len(bases), SHOTS, N), np.int8)
        for t in range(N_WIN):
            E = _W["E_t"][s][t]  # Pauli tensor of the Hamiltonian evolution; the rotation acts on it per setting
            for k in range(len(bases)):
                p = rotated_setting_probabilities(E, bases[k], delta[s, k] * times[t])
                sp[t, k] = sample_spins(p, SHOTS, rng, N)
        shots.append(sp)
    return tuple(shots)


def synthetic_job(args):
    world, r = args
    PARTS.mkdir(exist_ok=True)
    path = PARTS / f"b_{world}_{r}.npz"
    if path.exists():
        z = np.load(path)
        return z["phi0"], z["phi1"]
    t0 = time.time()
    times, _, _, bases = load()
    rng = np.random.default_rng(26000 + 1009 * r + (0 if world == "drift" else 500))
    if world == "drift":
        shots = world_shots(world, r, rng, bases)
    else:  # the control truth itself, with its quasi-static decay, sampled shot by shot
        ctrl = load_truth("control")
        shots = []
        for p in ctrl["prob"]:
            sp = np.empty((N_WIN, p.shape[1], SHOTS, N), np.int8)
            for t in range(N_WIN):
                for k in range(p.shape[1]):
                    sp[t, k] = sample_spins(p[t, k], SHOTS, rng, N)
            shots.append(sp)
        shots = tuple(shots)
    strings, tw = strings_list(), times[:N_WIN]
    tables = per_setting_tables(shots, bases, strings)
    phi0 = fit_tables(tables, tw, _W["d"], _W["dt"], 0.0, 0.0)
    phi1 = np.array([level1(mdl, shots, tables, bases, strings, tw, _W["d"], _W["dt"], 0.0, 0.0, rng, CHAINS_B)
                     for mdl in MODELS])
    ex = excess_ratios(shots, bases, strings)
    np.savez_compressed(path, phi0=phi0, phi1=phi1, excess=np.array(list(ex.values())))
    print(f"part B {world} {r}: {time.time() - t0:.0f} s", flush=True)
    return phi0, phi1


def part_b():
    times = load()[0]
    ctrl = load_truth("control")
    fine, _ = grids(times)
    strings, tw = strings_list(), times[:N_WIN]
    d = trajectory_defect(fine, ctrl["fine"], strings, tw, T_LEARN)
    dt = trajectory_defect(fine, ctrl["fine"], strings, tw, T_LEARN, weight=lambda s: s)
    E_t = hamiltonian_tensors()
    jobs = [("drift", r) for r in range(N_DRIFT)] + [("shot-only", r) for r in range(N_SHOT)]
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_b, initargs=(E_t, d, dt)) as pool:
        res = list(pool.map(synthetic_job, jobs))
    J0 = jnn(truth_theta())
    out = {"truth J_nn (rad/s)": J0 * 1e3}
    for world in ("drift", "shot-only"):
        sel = [m for m, j in enumerate(jobs) if j[0] == world]
        ex = np.array([np.load(PARTS / f"b_{world}_{jobs[m][1]}.npz")["excess"] for m in sel]).mean(axis=0)
        row = {"between-setting / shot variance, mean over realisations": ex.tolist(),
               "least squares: J_nn relative error mean, se": None}
        rel0 = np.array([jnn(to_theta(res[m][0], 0.0)[0]) / J0 - 1 for m in sel])
        row["least squares: J_nn relative error mean, se"] = [float(rel0.mean()), float(rel0.std(ddof=1) / np.sqrt(len(sel)))]
        for i, mdl in enumerate(MODELS):
            relS = np.array([jnn(estimator_s(res[m][0], res[m][1][i], 0.0)[0]) / J0 - 1 for m in sel])
            lam = np.array([estimator_s(res[m][0], res[m][1][i], 0.0)[2] for m in sel])
            row[f"S with {mdl}: J_nn relative error mean, se; lambda mean"] = [
                float(relS.mean()), float(relS.std(ddof=1) / np.sqrt(len(sel))), float(lam.mean())]
        out[world] = row
    passes = {mdl: bool(all(abs(out[w][f"S with {mdl}: J_nn relative error mean, se; lambda mean"][0]) <= 0.01
                            for w in ("drift", "shot-only"))) for mdl in MODELS}
    out["passes (|mean relative error of J_nn| <= 1% in both worlds)"] = passes
    out["primary noise model for part C"] = ("wild-ar" if passes["wild-ar"] else
                                             "wild-iid" if passes["wild-iid"] else None)
    return out


# ---------------- part C: the data ----------------
def _init_c(inputs, model, tables):
    _W.update(inputs=inputs, model=model, tables=tables)


def data_chunk(c):
    shots, bases, strings, tw, d, dt = _W["inputs"]
    rng = np.random.default_rng(2626 + 7919 * c)
    return level1(_W["model"], shots, _W["tables"], bases, strings, tw, d, dt, S_51, EMISSION_D52, rng, 10)


def band_fit(r):
    shots, bases, strings, tw, d, dt = _W["inputs"]
    rng = np.random.default_rng(262600 + 7907 * r)
    sh, tab = perturb(_W["model"], shots, _W["tables"], bases, strings, rng)
    p0 = fit_tables(tab, tw, d, dt, S_51, EMISSION_D52)
    p1 = level1(_W["model"], sh, tab, bases, strings, tw, d, dt, S_51, EMISSION_D52, rng, CHAINS_REP)
    return estimator_s(p0, p1, EMISSION_D52)


def part_c(model):
    inputs = data_inputs()
    shots, bases, strings, tw, d, dt = inputs
    tables = per_setting_tables(shots, bases, strings)
    phi0 = fit_tables(tables, tw, d, dt, S_51, EMISSION_D52)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_c, initargs=(inputs, model, tables)) as pool:
        phi1 = sum(pool.map(data_chunk, range(CHAINS_C // 10))) / (CHAINS_C // 10)
        reps = list(pool.map(band_fit, range(N_REP)))
    print(f"part C fits: {time.time() - t0:.0f} s", flush=True)
    th, be, lam = estimator_s(phi0, phi1, EMISSION_D52)
    jobs = [(f"c_{model}_central", th, be, s, S_51, EMISSION_D52) for s in (0, 1)]
    jobs += [(f"c_{model}_band{r}", t, b, s, S_51, EMISSION_D52) for r, (t, b, _) in enumerate(reps) for s in (0, 1)]
    with ProcessPoolExecutor(max_workers=2 * WORKERS) as pool:
        preds = list(pool.map(predict, jobs))
    central = np.stack(preds[:2])
    band = np.array(preds[2:]).reshape(N_REP, 2, *central.shape[1:])
    np.savez_compressed(RESULTS / "e26_partc.npz", central=central, band=band, theta=th, beta=be, lam=lam,
                        phi0=phi0, phi1=phi1, rep_lam=np.array([x[2] for x in reps]))


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


def analyse_c(model):
    z = np.load(RESULTS / "e11_model_selection.npz")
    e20 = np.load(RESULTS / "e20_flip_models.npz")
    xb15 = np.load(RESULTS / "e15_coverage.npz")["band_Xb15"]
    c = np.load(RESULTS / "e26_partc.npz")
    times, sem = z["times"], z["sem"]
    oos = times > T_LEARN + 1e-9
    late = times >= 4.0 - 1e-9
    Cs = class_matrices()
    rng = np.random.default_rng(2612)
    band, central = c["band"], c["central"]
    f_new = factors(central, z, band, oos)
    nulls = {n: null_pooled(band, Cs[n], oos, 1, sem=sem, rng=rng) for n in NEUTRAL}
    p = np.array([float(np.mean(nulls[n] >= f_new[n])) for n in NEUTRAL])
    rej = holm(p)
    cells = [float(np.mean((central @ Cs[k])[s][late].var(axis=0, ddof=1))
                   / np.mean((e20["pred_Xb15uf"] @ Cs[k])[s][late].var(axis=0, ddof=1))) for k in NEUTRAL for s in (0, 1)]
    out = {"noise model": model,
           "coupling scale lambda (late-data values of E23d: 1.069, 1.104)": float(c["lam"]),
           "replicates: lambda mean, sd": [float(c["rep_lam"].mean()), float(c["rep_lam"].std(ddof=1))],
           "beta (rad/s per ms)": float(c["beta"]) * 1e3,
           "band spread over the Xb15 band (rms, t > 1.5 ms)": {
               n: float(np.sqrt(np.mean(((band @ Cs[n]).std(axis=0, ddof=1) ** 2)[:, oos])
                                / np.mean(((xb15 @ Cs[n]).std(axis=0, ddof=1) ** 2)[:, oos]))) for n in Cs},
           "factors with its own band": f_new, "factors with the Xb15 band": factors(central, z, xb15, oos),
           "rule B': p-values, Holm rejections": {n: [float(p[i]), bool(rej[i])] for i, n in enumerate(NEUTRAL)},
           "rule B' passes": bool(not rej.any()),
           "guard: late variance / Xb15uf as learned, mean of six cells": float(np.mean(cells)),
           "rule A (E20 rule, own band)": bool(all(f_new[n] <= 1.2 for n in NEUTRAL)
                                               and f_new["all 435 observables"] <= 1.15)}
    out["guard holds"] = bool(out["guard: late variance / Xb15uf as learned, mean of six cells"] >= 0.8)
    out["data consistent with S (rule B' and guard)"] = bool(out["rule B' passes"] and out["guard holds"])
    return out


def plot(out):
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    ax = axes[0]
    b = out["part B (synthetic, parameter level)"]
    labels = ["least squares"] + [f"S, {m}" for m in MODELS]
    for j, (world, c) in enumerate((("drift", CAT[1]), ("shot-only", CAT[0]))):
        row = b[world]
        vals = [row["least squares: J_nn relative error mean, se"]] + [
            row[f"S with {m}: J_nn relative error mean, se; lambda mean"][:2] for m in MODELS]
        y = np.arange(len(labels)) + 0.15 * (j - 0.5)
        ax.errorbar([100 * v[0] for v in vals], y, xerr=[200 * v[1] for v in vals], fmt="o", color=c, ms=5,
                    capsize=2, label=f"{world} world")
    ax.axvspan(-1, 1, color=CAT[0], alpha=0.08, lw=0)
    ax.axvline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("relative error of $J_{nn}$ (%), mean $\\pm$ 2 se")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("(a) part B: synthetic data, truth known", fontsize=9)
    ax = axes[1]
    c = out.get("part C (data)")
    if c:
        classes = list(c["factors with its own band"])
        ax.plot([c["factors with the Xb15 band"][n] for n in classes], np.arange(len(classes)) - 0.12, "s",
                color=MUTED, ms=5, label="S (drift-aware), Xb15 band")
        ax.plot([c["factors with its own band"][n] for n in classes], np.arange(len(classes)) + 0.12, "o",
                color=ORDINAL[2], ms=5, label="S (drift-aware), its own band")
        ax.axvline(1.0, color=MUTED, lw=0.6, ls=":")
        ax.set_yticks(range(len(classes)), classes)
        ax.invert_yaxis()
        ax.set_xlabel("calibration factor, t > 1.5 ms")
        ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("(b) part C: data", fontsize=9)
    fig.suptitle("E26: detuning drift across the settings in the noise model (data: Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e26_setting_drift")


if __name__ == "__main__":
    out = {"part A (data, learning window)": part_a()}
    print("part A:", out["part A (data, learning window)"], flush=True)
    if "--part-a" in sys.argv:
        sys.exit(0)
    out["part B (synthetic, parameter level)"] = part_b()
    print("part B:", out["part B (synthetic, parameter level)"], flush=True)
    model = out["part B (synthetic, parameter level)"]["primary noise model for part C"]
    if model is not None and "--no-c" not in sys.argv:
        if not (RESULTS / "e26_partc.npz").exists():
            part_c(model)
        out["part C (data)"] = analyse_c(model)
        print("part C:", out["part C (data)"], flush=True)
    save_json(RESULTS / "e26_setting_drift.json", out)
    print(plot(out))
