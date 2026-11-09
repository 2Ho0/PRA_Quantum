"""E23s: does a generator of its own for each initial state remove the neutral-class error? (preregistered)

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0), analysed
here from the public raw shots. E23k found that Xb15uf learned on each initial state alone gives two generators that
disagree (mean z field 191 against 68 rad/s, 5.6 sigma; gamma_c 2.9 sigma). The post hoc check of E23c showed that the
data's late oscillations have the amplitude of the undamped model, so the neutral error of the joint model is a phase
error of coherent dynamics. A compromise generator for two data sets that one generator cannot describe would produce
exactly such phase errors.
Model Xb15uf-s: the Xb15uf structure (E20: U(1)-symmetric couplings, z fields, uniform transverse fields, collective
dephasing gamma_c + beta t, D5/2 emission and 51-ion flips fixed) learned on t <= 1.5 ms of one initial state alone, and
that state predicted with its own generator on 1.5 < t <= 10 ms (out of sample in time, as in E20).
Fixed before the run (the rule of E20): generators of their own per initial state explain the neutral error if all
three neutral classes reach a calibration factor <= 1.2 without raising the factor of all 435 observables above 1.15.
Factors use the Xb15 bootstrap spread, which understates the larger estimation spread of a single-state fit; this
works against the model. Reported, not judged: factors per state and the late oscillation amplitudes.
Second variant (`--run-params`, added after the first run failed: the separate generators made the neutral classes
worse, 1.6-1.7): Xb15uf-r keeps one joint generator and adds two run-specific nuisance parameters for state 2, a global
detuning sum_i Z_i and a collective dephasing rate (as E23k (D)); each state is predicted with its own run parameters.
Fixed before that run: the same rule of E20.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import lsq_linear  # noqa: E402

from common import CAT, INK, MUTED, RESULTS, save_figure, save_json, style  # noqa: E402
from e6_ion_learning import RAD_S, ansatz, observables, per_setting_tables, series_from  # noqa: E402
from e6c_validation import charge_classes, combine, combine_sem, load  # noqa: E402
from e9_symmetry_data import extended_ansatz  # noqa: E402
from e11_model_selection import N_SUB  # noqa: E402
from e20_flip_models import S_51, factors, predict_job, structure  # noqa: E402
from qerr.learning import integral_system, needed_strings, refine, trajectory_defect  # noqa: E402

T_LEARN = 1.5
NEUTRAL = ("q=0  Z_i", "q=0  Z_iZ_j", "q=0  flip-flop")


def fit_state(s):
    """Xb15uf learned on t <= T_LEARN of initial state s alone; returns (theta, beta)."""
    times, s1, s2, bases = load()
    obs, ext = observables(), extended_ansatz()
    strings = sorted(set(obs) | set(needed_strings(obs, ansatz())))
    series = series_from(per_setting_tables((s1, s2), bases, strings))
    z = np.load(RESULTS / "e11_stage1.npz")
    fine = refine(times, N_SUB)
    traj = [z["traj0"], z["traj1"]]
    d = trajectory_defect(fine, traj, strings, times, T_LEARN)
    dt = trajectory_defect(fine, traj, strings, times, T_LEARN, weight=lambda u: u)
    A, y, _ = integral_system(obs, ext, series, times, T_LEARN, states=(s,), defect=d)
    At, _, _ = integral_system(obs, ext, series, times, T_LEARN, states=(s,), defect=dt, weight=lambda u: u)
    T, fix, flip, coll = structure(ext, uniform=True)
    phi = lsq_linear(np.column_stack([A @ T, At @ coll]), y - A @ fix - S_51 * (A @ flip), method="bvls", tol=1e-12).x
    return T @ phi[:-1] + fix, float(phi[-1])


def fit_run_parameters():
    """One joint Xb15uf generator plus a global detuning and a collective rate for state 2; (theta_1, theta_2, beta)."""
    times, s1, s2, bases = load()
    obs, ext = observables(), extended_ansatz()
    strings = sorted(set(obs) | set(needed_strings(obs, ansatz())))
    series = series_from(per_setting_tables((s1, s2), bases, strings))
    z = np.load(RESULTS / "e11_stage1.npz")
    fine = refine(times, N_SUB)
    traj = [z["traj0"], z["traj1"]]
    d = trajectory_defect(fine, traj, strings, times, T_LEARN)
    dt = trajectory_defect(fine, traj, strings, times, T_LEARN, weight=lambda u: u)
    T, fix, flip, coll = structure(ext, uniform=True)
    k = {p: m for m, p in enumerate(ext.h_terms)}
    hz = np.zeros(ext.n_params)
    for i in range(10):
        hz[k[((i, "Z"),)]] = 1.0
    blocks, rhs = [], []
    for s in (0, 1):
        A, y, _ = integral_system(obs, ext, series, times, T_LEARN, states=(s,), defect=d)
        At, _, _ = integral_system(obs, ext, series, times, T_LEARN, states=(s,), defect=dt, weight=lambda u: u)
        run = np.column_stack([A @ hz, A @ coll]) * (1.0 if s == 1 else 0.0)
        blocks.append(np.column_stack([A @ T, At @ coll, run]))
        rhs.append(y - A @ fix - S_51 * (A @ flip))
    phi = lsq_linear(np.vstack(blocks), np.concatenate(rhs), method="bvls", tol=1e-12).x
    nT = T.shape[1]
    theta1 = T @ phi[:nT] + fix
    theta2 = theta1 + phi[nT + 1] * hz + phi[nT + 2] * coll
    return theta1, theta2, float(phi[nT])


def run_parameters():
    th1, th2, beta = fit_run_parameters()
    ext = extended_ansatz()
    k = {p: m for m, p in enumerate(ext.h_terms)}
    print(f"run parameters of state 2: detuning {(th2 - th1)[k[((0, 'Z'),)]] * RAD_S:.1f} rad/s, collective rate "
          f"{(th2 - th1)[len(ext.h_terms)] * RAD_S:.1f} rad/s; beta {beta * RAD_S:.1f}", flush=True)
    jobs = [("Xb15uf", th, beta, s, (0.0, 0.0, 1.0)) for s, th in ((0, th1), (1, th2))]
    with ProcessPoolExecutor(max_workers=2) as pool:
        return th1, th2, beta, np.stack(list(pool.map(predict_job, jobs)))


def run():
    fits = {s: fit_state(s) for s in (0, 1)}
    ext = extended_ansatz()
    k = {p: m for m, p in enumerate(ext.h_terms)}
    for s, (th, b) in fits.items():
        print(f"state {s + 1}: gamma_c {th[len(ext.h_terms)] * RAD_S:.1f}, beta {b * RAD_S:.1f}, "
              f"h_x {th[k[((0, 'X'),)]] * RAD_S:.1f}, h_y {th[k[((0, 'Y'),)]] * RAD_S:.1f}", flush=True)
    jobs = [("Xb15uf", fits[s][0], fits[s][1], s, (0.0, 0.0, 1.0)) for s in (0, 1)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        preds = list(pool.map(predict_job, jobs))
    return fits, np.stack(preds)  # (state, time, observable), each state from its own generator


def analyse(pred, pred_r=None):
    z = np.load(RESULTS / "e11_model_selection.npz")
    e20 = np.load(RESULTS / "e20_flip_models.npz")
    bands = np.load(RESULTS / "e15_coverage.npz")["band_Xb15"]
    times = z["times"]
    oos = times > T_LEARN + 1e-9
    preds = {"Xb15uf (one generator)": e20["pred_Xb15uf"], "Xb15uf-s (one generator per state)": pred}
    if pred_r is not None:
        preds["Xb15uf-r (joint generator, run parameters for state 2)"] = pred_r
    out = {"calibration factors, t > 1.5 ms (Xb15 bootstrap spread)": {v: factors(p, z, bands, oos) for v, p in preds.items()}}
    cls = charge_classes(observables())
    per_state = {}
    for v, p in preds.items():
        for s in (0, 1):
            row = {}
            for n in NEUTRAL:
                co = cls[n]
                d, pc, se = combine(z["data"], co)[s], combine(p, co)[s], combine_sem(z["sem"], co)[s]
                sd = combine(bands, co).std(axis=0, ddof=1)[s]
                row[n] = float(np.sqrt(np.mean((d - pc)[oos] ** 2) / np.mean((sd**2 + se**2)[oos])))
            per_state[f"{v}, state {s + 1}"] = row
    out["neutral factors per state"] = per_state
    late = times >= 4.0 - 1e-9
    amp = {}
    for n in NEUTRAL:
        co = cls[n]
        for s in (0, 1):
            d, se = combine(z["data"], co)[s][late], combine_sem(z["sem"], co)[s][late]
            amp[f"{n}, state {s + 1}"] = {
                "data (shot noise removed)": float(np.mean(d.var(axis=0, ddof=1) - np.mean(se**2, axis=0))),
                **{v: float(np.mean(combine(p, co)[s][late].var(axis=0, ddof=1))) for v, p in preds.items()}}
    out["temporal variance on t >= 4 ms"] = amp
    f = out["calibration factors, t > 1.5 ms (Xb15 bootstrap spread)"]["Xb15uf-s (one generator per state)"]
    out["verdict"] = {"all neutral classes <= 1.2": bool(all(f[n] <= 1.2 for n in NEUTRAL)),
                      "all 435 observables <= 1.15": bool(f["all 435 observables"] <= 1.15)}
    out["verdict"]["one generator per state explains the neutral error"] = bool(all(out["verdict"].values()))
    if pred_r is not None:
        f = out["calibration factors, t > 1.5 ms (Xb15 bootstrap spread)"][
            "Xb15uf-r (joint generator, run parameters for state 2)"]
        ok = [all(f[n] <= 1.2 for n in NEUTRAL), f["all 435 observables"] <= 1.15]
        out["verdict (run parameters)"] = {"all neutral classes <= 1.2": bool(ok[0]),
                                           "all 435 observables <= 1.15": bool(ok[1]),
                                           "run parameters explain the neutral error": bool(all(ok))}
    return out


def plot(out, pred):
    plt = style()
    z = np.load(RESULTS / "e11_model_selection.npz")
    e20 = np.load(RESULTS / "e20_flip_models.npz")
    k = {O: m for m, O in enumerate(observables())}
    times = z["times"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
    ax = axes[0]
    fac = out["calibration factors, t > 1.5 ms (Xb15 bootstrap spread)"]
    classes = list(next(iter(fac.values())))
    for j, (v, c) in enumerate(zip(fac, [MUTED, CAT[1], CAT[2]])):
        ax.plot([fac[v][n] for n in classes], np.arange(len(classes)) + 0.15 * (j - 1), "D", color=c, ms=5, label=v)
    ax.axvline(1.0, color=MUTED, ls=":", lw=1)
    ax.axvline(1.2, color=MUTED, ls="--", lw=0.8)
    ax.set_yticks(range(len(classes)), classes)
    ax.invert_yaxis()
    ax.set_xlabel("calibration factor, t > 1.5 ms")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("(a) calibration by charge class", fontsize=9)
    ax = axes[1]
    m = k[((0, "Z"), (9, "Z"))]
    ax.plot(times, z["data"][1][:, m], "o", color=INK, ms=3.5, label="data")
    ax.plot(times, e20["pred_Xb15uf"][1][:, m], "-", color=MUTED, lw=1.2, label="Xb15uf (one generator)")
    ax.plot(times, pred[1][:, m], "-", color=CAT[1], lw=1.4, label="Xb15uf-s (own generator)")
    ax.axvspan(0.0, T_LEARN, color=MUTED, alpha=0.08, lw=0)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("$\\langle Z_0 Z_9 \\rangle$, state 2")
    ax.legend(fontsize=7)
    ax.set_title("(b) an edge correlation", fontsize=9)
    fig.suptitle("E23s: a generator of its own for each initial state (data: Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e23s_state_generators")


if __name__ == "__main__":
    path = RESULTS / "e23s_state_generators.npz"
    path_r = RESULTS / "e23s_run_parameters.npz"
    if "--run-params" in sys.argv:
        th1, th2, beta, pred_r = run_parameters()
        np.savez_compressed(path_r, pred=pred_r, theta_0=th1, theta_1=th2, beta=beta)
    elif "--replot" not in sys.argv:
        fits, pred = run()
        np.savez_compressed(path, pred=pred, **{f"theta_{s}": fits[s][0] for s in (0, 1)},
                            **{f"beta_{s}": fits[s][1] for s in (0, 1)})
    pred = np.load(path)["pred"]
    out = analyse(pred, np.load(path_r)["pred"] if path_r.exists() else None)
    save_json(RESULTS / "e23s_state_generators.json", out)
    print(plot(out, pred))
    for key, v in out.items():
        print(f"{key}: {v}")
