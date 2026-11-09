"""E24: does an errors-in-variables correction of the learning remove the neutral error? (preregistered)

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0), analysed
here from the public raw shots.
Background (Q27, E23d). The integral method regresses on measured correlators, so least squares shrinks the
couplings. E11b, E14 and the setting bootstrap put the shrinkage of J_nn at 3-4% on t <= 1.5 ms, and one common scale
of 1.040 lowered the neutral factors of Xb15uf to 1.16, 1.23 and 1.06 without damping. This run replaces the common
scale by a per-parameter correction of the Xb15uf fit of E20 (same structure, window, defect and fixed flips).
Estimator: SIMEX with shot resampling. The setting bootstrap of E11 is not a clean noise doubling here: a resample of
the 200 settings leaves some weight-3 strings unmeasured, and the rows that need them are dropped (2514 -> 1704 rows
in a test). Resampling the 200 shots of every (setting, time) keeps the design and adds shot noise of the measured
size to every Pauli estimate, with its correlations within a setting (qerr.shots.resample_shots). Level zeta = m
resamples m times in a chain (noise variance (1 + m) sigma^2). The extrapolation to zeta = -1 is linear,
2 phi_0 - mean phi_1, or quadratic, 3 phi_0 - 3 mean phi_1 + mean phi_2 (qerr.learning.simex_extrapolate).
Step 1 (synthetic, known truth): the E11f control truth (Y15 couplings, J_nn = 607.1 rad/s, quasi-static collective
noise with gamma_c = 0 and beta = 38.2 rad/s per ms), sampled like the data (200 settings, 200 shots, t <= 1.5 ms),
12 realisations, 40 chains per realisation, the truth's own (oracle) defect.
Fixed before step 1: an extrapolation passes if the mean relative error of J_nn over the realisations is within 1%.
The primary estimator is the linear one if it passes, else the quadratic one if it passes; if neither passes, step 2
is not run.
Step 2 (data): the primary estimator with 100 chains (seeds fixed), predicted as in E20 for both initial states.
Fixed before step 2 (calibration factors as in E20, with the Xb15 bootstrap band of E15, on 1.5 < t <= 10 ms):
  rule A (the E20 rule): all three neutral classes <= 1.2 and all 435 observables <= 1.15;
  rule B (familywise): Holm-Bonferroni at familywise 5% over the three neutral classes, with p-values from the E23d
          null distribution of the factor (a perfectly calibrated band, 4000 draws); it passes if no class is
          rejected. Xb15uf as learned fails it (smallest p = 0.0165 < 0.05/3).
  guard: the late temporal variance (t >= 4 ms) of the neutral classes, averaged over the six class x state cells,
          stays at >= 0.8 times that of Xb15uf as learned (no smoothing).
Verdict: the errors-in-variables correction explains the neutral excess if rule B passes and the guard holds. Rule A
is reported with it.
"""

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from common import CAT, INK, MUTED, RESULTS, save_figure, save_json, style  # noqa: E402
from e6_ion_learning import N, PAIRS, RAD_S, ansatz, observables, per_setting_tables, series_from  # noqa: E402
from e6c_validation import charge_classes, combine, combine_sem, load  # noqa: E402
from e9_symmetry_data import extended_ansatz  # noqa: E402
from e11_model_selection import N_SUB  # noqa: E402
from e11f_nonmarkov_truths import grids, load_truth  # noqa: E402
from e14_shot_calibration import truth_theta  # noqa: E402
from e20_flip_models import S_51, factors, predict_job, structure  # noqa: E402
from qerr.learning import integral_system, needed_strings, refine, simex_extrapolate, trajectory_defect  # noqa: E402
from qerr.shots import resample_shots, sample_spins  # noqa: E402

T_LEARN, N_WIN = 1.5, 4  # the learning window holds the first four times: 0, 0.5, 1.0, 1.5 ms
N_REAL, CHAINS_SYN, CHAINS_DATA, CHUNK, WORKERS = 12, 40, 100, 10, 6
SHOTS, LEVELS = 200, 2
NEUTRAL = ("q=0  Z_i", "q=0  Z_iZ_j", "q=0  flip-flop")
STEP1 = RESULTS / "e24_step1.npz"
STEP2 = RESULTS / "e24_step2.npz"
_W = {}


def strings_list():
    obs = observables()
    return sorted(set(obs) | set(needed_strings(obs, ansatz())))


def solve(shots, bases, strings, times, d, dt):
    """phi of the Xb15uf fit (E20: unbounded least squares) from the shots of both initial states."""
    obs, ext = observables(), extended_ansatz()
    series = series_from(per_setting_tables(shots, bases, strings))
    T, fix, flip, coll = structure(ext, uniform=True)
    A, y, _ = integral_system(obs, ext, series, times, T_LEARN, defect=d)
    At, _, _ = integral_system(obs, ext, series, times, T_LEARN, defect=dt, weight=lambda s: s)
    M = np.column_stack([A @ T, At @ coll])
    return np.linalg.lstsq(M, y - A @ fix - S_51 * (A @ flip), rcond=None)[0]


def to_theta(phi):
    T, fix, _, _ = structure(extended_ansatz(), uniform=True)
    return T @ phi[:-1] + fix, float(phi[-1])


def chains(shots, bases, strings, times, d, dt, rng, n):
    """Sum over n chains of phi at SIMEX levels 1..LEVELS (level m: shots resampled m times in a chain)."""
    total = None
    for _ in range(n):
        cur, row = shots, []
        for _m in range(LEVELS):
            cur = tuple(resample_shots(s, rng) for s in cur)
            row.append(solve(cur, bases, strings, times, d, dt))
        total = np.array(row) if total is None else total + np.array(row)
    return total


# ---------------- step 1: synthetic data with a known truth ----------------
def _init_syn(prob, d, dt):
    _W.update(prob=prob, d=d, dt=dt)


def synthetic_job(r):
    times, _, _, bases = load()
    rng = np.random.default_rng(104729 * r + 24)
    shots = []
    for p in _W["prob"]:  # (N_WIN, settings, 1024) per initial state
        sp = np.empty((p.shape[0], p.shape[1], SHOTS, N), np.int8)
        for t in range(p.shape[0]):
            for k in range(p.shape[1]):
                sp[t, k] = sample_spins(p[t, k], SHOTS, rng, N)
        shots.append(sp)
    shots, strings, tw = tuple(shots), strings_list(), times[:N_WIN]
    phi0 = solve(shots, bases, strings, tw, _W["d"], _W["dt"])
    sums = chains(shots, bases, strings, tw, _W["d"], _W["dt"], rng, CHAINS_SYN)
    return phi0, sums / CHAINS_SYN


def step1():
    if STEP1.exists():
        z = np.load(STEP1)
        return z["phi0"], z["levels"]
    times = load()[0]
    ctrl = load_truth("control")
    fine, _ = grids(times)
    strings = strings_list()
    tw = times[:N_WIN]
    d = trajectory_defect(fine, ctrl["fine"], strings, tw, T_LEARN)
    dt = trajectory_defect(fine, ctrl["fine"], strings, tw, T_LEARN, weight=lambda s: s)
    prob = [p[:N_WIN] for p in ctrl["prob"]]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_syn, initargs=(prob, d, dt)) as pool:
        res = list(pool.map(synthetic_job, range(N_REAL)))
    print(f"step 1: {time.time() - t0:.0f} s", flush=True)
    phi0, levels = np.array([r[0] for r in res]), np.array([r[1] for r in res])
    np.savez_compressed(STEP1, phi0=phi0, levels=levels)
    return phi0, levels


def couplings(theta):
    t = np.asarray(theta)[..., :180].reshape(*np.shape(theta)[:-1], 45, 4)
    return t[..., 0] + t[..., 1]


def jnn(theta):
    J = couplings(theta)
    nn = [m for m, (i, j) in enumerate(PAIRS) if j == i + 1]
    return J[..., nn].mean(axis=-1)


def analyse_step1(phi0, levels):
    truth = truth_theta()
    J0 = float(jnn(truth))
    k0 = len(ansatz().h_terms)
    gamma_truth, beta_truth = 0.0, float(2.0 * truth[k0] / 1.5)
    est = {"plain": phi0, "linear": simex_extrapolate([phi0, levels[:, 0]]),
           "quadratic": simex_extrapolate([phi0, levels[:, 0], levels[:, 1]])}
    out = {"truth: J_nn, gamma_c, beta (rad/s, rad/s per ms)": [J0 * RAD_S, gamma_truth, beta_truth * RAD_S]}
    Jt = couplings(truth)
    for name, ph in est.items():
        th = np.array([to_theta(p)[0] for p in ph])
        be = np.array([to_theta(p)[1] for p in ph])
        rel = jnn(th) / J0 - 1.0
        scale = couplings(th) @ Jt / (Jt @ Jt)
        out[name] = {"J_nn relative error: mean, sd, se": [float(rel.mean()), float(rel.std(ddof=1)),
                                                            float(rel.std(ddof=1) / np.sqrt(len(rel)))],
                     "J_nn rms relative error": float(np.sqrt(np.mean(rel**2))),
                     "coupling scale (45 pairs) against the truth: mean, sd": [float(scale.mean()),
                                                                              float(scale.std(ddof=1))],
                     "gamma_c (rad/s): mean, sd": [float(th[:, k0].mean() * RAD_S), float(th[:, k0].std(ddof=1) * RAD_S)],
                     "beta (rad/s per ms): mean, sd": [float(be.mean() * RAD_S), float(be.std(ddof=1) * RAD_S)],
                     "passes (|mean relative error of J_nn| <= 1%)": bool(abs(rel.mean()) <= 0.01)}
    primary = "linear" if out["linear"]["passes (|mean relative error of J_nn| <= 1%)"] else (
        "quadratic" if out["quadratic"]["passes (|mean relative error of J_nn| <= 1%)"] else None)
    out["primary estimator for step 2"] = primary
    return out, primary


# ---------------- step 2: the data ----------------
def _init_data(shots, bases, strings, times, d, dt):
    _W.update(shots=shots, bases=bases, strings=strings, times=times, d=d, dt=dt)


def data_chunk(c):
    rng = np.random.default_rng(20261109 + 7919 * c)
    return chains(_W["shots"], _W["bases"], _W["strings"], _W["times"], _W["d"], _W["dt"], rng, CHUNK)


def data_inputs():
    times, s1, s2, bases = load()
    obs, ans = observables(), ansatz()
    z = np.load(RESULTS / "e11_stage1.npz")
    fine = refine(times, N_SUB)
    traj_strings = needed_strings(obs, ans)
    tw = times[:N_WIN]
    d = trajectory_defect(fine, [z["traj0"], z["traj1"]], traj_strings, tw, T_LEARN)
    dt = trajectory_defect(fine, [z["traj0"], z["traj1"]], traj_strings, tw, T_LEARN, weight=lambda s: s)
    return (s1[:N_WIN], s2[:N_WIN]), bases, strings_list(), tw, d, dt


def step2(primary):
    if STEP2.exists():
        z = np.load(STEP2)
        return z["phi0"], z["levels"], z["pred"]
    shots, bases, strings, tw, d, dt = data_inputs()
    phi0 = solve(shots, bases, strings, tw, d, dt)
    e20 = np.load(RESULTS / "e20_flip_models.npz")
    assert np.abs(to_theta(phi0)[0] - e20["theta_Xb15uf"]).max() < 1e-12, "the window-only fit must reproduce E20"
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_data,
                             initargs=(shots, bases, strings, tw, d, dt)) as pool:
        sums = sum(pool.map(data_chunk, range(CHAINS_DATA // CHUNK)))
    levels = sums / CHAINS_DATA
    print(f"step 2 chains: {time.time() - t0:.0f} s", flush=True)
    ext = [levels[0]] if primary == "linear" else [levels[0], levels[1]]
    theta_c, beta_c = to_theta(simex_extrapolate([phi0] + ext))
    bp = (0.0, 0.0, 1.0)  # no flip burst (Xb15uf)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=2) as pool:
        pred = np.stack(list(pool.map(predict_job, [("Xb15uf", theta_c, beta_c, s, bp) for s in (0, 1)])))
    print(f"step 2 predictions: {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(STEP2, phi0=phi0, levels=levels, pred=pred, theta_c=theta_c, beta_c=beta_c)
    return phi0, levels, pred


def holm(pvals, alpha=0.05):
    """Rejections of Holm-Bonferroni at familywise level alpha."""
    order = np.argsort(pvals)
    rej = np.zeros(len(pvals), bool)
    for r, i in enumerate(order):
        if pvals[i] > alpha / (len(pvals) - r):
            break
        rej[i] = True
    return rej


def analyse_step2(phi0, levels, pred, primary):
    z = np.load(RESULTS / "e11_model_selection.npz")
    e20 = np.load(RESULTS / "e20_flip_models.npz")
    band = np.load(RESULTS / "e15_coverage.npz")["band_Xb15"]
    nulls = np.load(RESULTS / "e23d_scale_bias.npz")
    times = z["times"]
    oos = times > T_LEARN + 1e-9
    late = times >= 4.0 - 1e-9
    ext = [levels[0]] if primary == "linear" else [levels[0], levels[1]]
    phi_c = simex_extrapolate([phi0] + ext)
    th0, b0 = to_theta(phi0)
    thc, bc = to_theta(phi_c)
    k0 = len(extended_ansatz().h_terms)
    k = {p: m for m, p in enumerate(extended_ansatz().h_terms)}
    J0, Jc = couplings(th0), couplings(thc)
    hz = lambda th: np.array([th[k[((i, "Z"),)]] for i in range(N)])
    out = {"primary estimator": primary,
           "parameters (rad/s; beta in rad/s per ms): as learned / corrected": {
               "J_nn": [float(jnn(th0) * RAD_S), float(jnn(thc) * RAD_S)],
               "coupling scale corrected / as learned (45 pairs)": float(Jc @ J0 / (J0 @ J0)),
               "gamma_c": [float(th0[k0] * RAD_S), float(thc[k0] * RAD_S)],
               "beta": [b0 * RAD_S, bc * RAD_S],
               "uniform h_x, h_y": [[float(th0[k[((0, a),)]] * RAD_S) for a in "XY"],
                                    [float(thc[k[((0, a),)]] * RAD_S) for a in "XY"]],
               "mean and spread of h_z": [[float(hz(th0).mean() * RAD_S), float(np.ptp(hz(th0)) * RAD_S)],
                                          [float(hz(thc).mean() * RAD_S), float(np.ptp(hz(thc)) * RAD_S)]]}}
    fac = {"Xb15uf as learned (E20)": factors(e20["pred_Xb15uf"], z, band, oos),
           f"Xb15uf, SIMEX-corrected ({primary})": factors(pred, z, band, oos)}
    out["calibration factors, t > 1.5 ms (Xb15 bootstrap band)"] = fac
    f = fac[f"Xb15uf, SIMEX-corrected ({primary})"]
    p = np.array([float(np.mean(nulls["null_" + n.replace(" ", "_")] >= f[n])) for n in NEUTRAL])
    rej = holm(p)
    cls = charge_classes(observables())
    cells = []
    for n in NEUTRAL:
        C = cls[n]
        for s in (0, 1):
            var = lambda q: float(np.mean(combine(q, C)[s][late].var(axis=0, ddof=1)))
            cells.append(var(pred) / var(e20["pred_Xb15uf"]))
    out["rule A (E20): all neutral <= 1.2 and all 435 <= 1.15"] = bool(
        all(f[n] <= 1.2 for n in NEUTRAL) and f["all 435 observables"] <= 1.15)
    out["rule B: null p-values of the neutral classes, Holm rejections at familywise 5%"] = {
        n: [float(p[i]), bool(rej[i])] for i, n in enumerate(NEUTRAL)}
    out["rule B passes (no rejection)"] = bool(not rej.any())
    out["guard: late variance corrected / as learned, six cells and mean"] = [cells, float(np.mean(cells))]
    out["guard holds (mean >= 0.8)"] = bool(np.mean(cells) >= 0.8)
    out["verdict: the errors-in-variables correction explains the neutral excess"] = bool(
        out["rule B passes (no rejection)"] and out["guard holds (mean >= 0.8)"])

    def z_time(q):
        num = sum(np.mean((combine(z["data"], cls[n]) - combine(q, cls[n])) ** 2, axis=(0, 2)) for n in NEUTRAL)
        den = sum(np.mean(combine(band, cls[n]).std(axis=0, ddof=1) ** 2 + combine_sem(z["sem"], cls[n]) ** 2,
                          axis=(0, 2)) for n in NEUTRAL)
        return np.sqrt(num / den)

    out["pooled neutral z-score per time: as learned, corrected"] = {
        f"{t:.1f}": [float(a), float(b)] for t, a, b in zip(times, z_time(e20["pred_Xb15uf"]), z_time(pred))}
    return out


def plot(out1, phi0_syn, levels_syn, out2):
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ax = axes[0]
    J0 = float(jnn(truth_theta()))
    est = {"plain": phi0_syn, "linear": simex_extrapolate([phi0_syn, levels_syn[:, 0]]),
           "quadratic": simex_extrapolate([phi0_syn, levels_syn[:, 0], levels_syn[:, 1]])}
    for y, (name, ph) in enumerate(est.items()):
        rel = 100 * (np.array([jnn(to_theta(p)[0]) for p in ph]) / J0 - 1.0)
        ax.plot(rel, np.full(rel.size, y) + np.linspace(-0.12, 0.12, rel.size), "o", color=[MUTED, CAT[0], CAT[1]][y],
                ms=4, alpha=0.8)
        ax.plot(rel.mean(), y, "D", color=INK, ms=6)
    ax.axvspan(-1, 1, color=CAT[0], alpha=0.08, lw=0)
    ax.axvline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_yticks(range(3), ["least squares", "SIMEX linear", "SIMEX quadratic"])
    ax.invert_yaxis()
    ax.set_xlabel("relative error of $J_{nn}$ (%), synthetic, truth known")
    ax.set_title("(a) step 1: shrinkage on synthetic data", fontsize=9)
    if out2 is not None:
        ax = axes[1]
        fac = out2["calibration factors, t > 1.5 ms (Xb15 bootstrap band)"]
        names = list(fac)
        classes = list(fac[names[0]])
        for j, (v, c) in enumerate(zip(names, [MUTED, CAT[0]])):
            ax.plot([fac[v][n] for n in classes], np.arange(len(classes)) + 0.14 * (j - 0.5), "D", color=c, ms=5,
                    label=v)
        ax.axvline(1.0, color=MUTED, ls=":", lw=1)
        ax.axvline(1.2, color=MUTED, ls="--", lw=0.8)
        ax.set_yticks(range(len(classes)), classes)
        ax.invert_yaxis()
        ax.set_xlabel("calibration factor, t > 1.5 ms")
        ax.legend(fontsize=6.5, loc="upper left")
        ax.set_title("(b) step 2: calibration by charge class", fontsize=9)
        ax = axes[2]
        zt = out2["pooled neutral z-score per time: as learned, corrected"]
        tt = np.array([float(t) for t in zt])
        v = np.array(list(zt.values()))
        ax.plot(tt, v[:, 0], "-o", color=INK, lw=1.4, ms=3.5, label="Xb15uf as learned")
        ax.plot(tt, v[:, 1], "-o", color=CAT[0], lw=1.4, ms=3.5, label="SIMEX-corrected")
        ax.axvspan(0.0, T_LEARN, color=MUTED, alpha=0.08, lw=0)
        ax.axhline(1.0, color=MUTED, lw=0.6, ls=":")
        ax.set_xlabel("t (ms)")
        ax.set_ylabel("neutral z-score (rms deviation / rms spread)")
        ax.legend(fontsize=6.5, loc="upper left")
        ax.set_title("(c) neutral error against time", fontsize=9)
    else:
        for ax in axes[1:]:
            ax.axis("off")
    fig.suptitle("E24: errors-in-variables correction of the learning (data: Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e24_eiv_correction")


if __name__ == "__main__":
    phi0_syn, levels_syn = step1()
    out1, primary = analyse_step1(phi0_syn, levels_syn)
    print("step 1:", out1, flush=True)
    out = {"step 1 (synthetic, E11f control truth)": out1}
    out2 = None
    if primary is not None and "--step1-only" not in sys.argv:
        phi0, levels, pred = step2(primary)
        out2 = analyse_step2(phi0, levels, pred, primary)
        out["step 2 (data)"] = out2
    save_json(RESULTS / "e24_eiv_correction.json", out)
    print(plot(out1, phi0_syn, levels_syn, out2))
    for key, v in (out2 or {}).items():
        print(f"{key}: {v}")
