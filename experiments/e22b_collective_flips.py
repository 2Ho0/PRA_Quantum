"""E22b: do collective flips, calibrated on the other initial state, remove the neutral-class error of Xb15uf?

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0), analysed
here from the public raw shots. E22(B) found that the mean ZZ over all pairs, (M^2 - N) / (N (N - 1)), grows at late
times in both initial states, although every U(1)-covariant generator conserves it and the independent flips of Xb15uf
pull it towards zero. Flips common to all ions do the opposite. With the jump operators sqrt(g) sum_k X_k and
sqrt(g) sum_k Y_k (collective flips: covariant, no charge-2 part, rate g each),
  d<Z_i Z_j>/dt = 4 g <X_i X_j + Y_i Y_j> - 8 g <Z_i Z_j>,   d<Z_i>/dt = -4 g <Z_i>,
and the D5/2 emission adds -e (<Z_i> + <Z_j> + 2 <Z_i Z_j>). The late M^2 slope of one state and its own late
correlations therefore fix g (t >= 2 ms).
Variant Xb15uc: Xb15uf (E20) with the independent 51-ion flips replaced by collective flips. For a strict out-of-sample
test each initial state is predicted with the rate calibrated on the other state; the remaining parameters are learned
on t <= 1.5 ms as in E20 (collective columns fixed, qerr.learning.correlated_adjoint) and predicted on 1.5 < t <= 10 ms.
Fixed before the run (the rule of E20): the collective flips explain the neutral error if all three neutral classes
reach a calibration factor <= 1.2 without raising the factor of all 435 observables above 1.15 (factors as in E20,
with the Xb15 bootstrap spread). Reported, not judged: the late M^2 and mean-Z slopes of the predictions.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import gc  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.optimize import lsq_linear  # noqa: E402

from common import CAT, INK, MUTED, RESULTS, save_figure, save_json, style  # noqa: E402
from e6_ion_learning import N, PAIRS, RAD_S, ansatz, observables, per_setting_tables, product_state, series_from  # noqa: E402
from e6c_validation import load  # noqa: E402
from e9_symmetry_data import extended_ansatz  # noqa: E402
from e11_model_selection import EMISSION_D52, N_SUB  # noqa: E402
from e17_neutral_diagnosis import conserved, energy_weights, indices  # noqa: E402
from e20_flip_models import factors, structure  # noqa: E402
from e22_first_milliseconds import T_LATE, cross_columns  # noqa: E402
from qerr.learning import gamma_matrix, integral_system, needed_strings, refine, trajectory_defect  # noqa: E402
from qerr.lindblad import OpenModel, lindblad_rhs  # noqa: E402
from qerr.pauli import expectation_tables, expectations, psparse  # noqa: E402

T_LEARN = 1.5
VARIANT = "Xb15uc"


def calibrate(series, times):
    """Collective flip rate g (1/ms) of each state from its late M^2 slope and correlations."""
    late = times >= T_LATE - 1e-9
    out = []
    for s in (0, 1):
        zz = np.mean([series[(s, ((i, "Z"), (j, "Z")))] for i, j in PAIRS], axis=0)
        ff = np.mean([series[(s, ((i, "X"), (j, "X")))] + series[(s, ((i, "Y"), (j, "Y")))] for i, j in PAIRS], axis=0)
        zm = np.mean([series[(s, ((i, "Z"),))] for i in range(N)], axis=0)
        slope = np.polyfit(times[late], zz[late], 1)[0]
        emission = -EMISSION_D52 * (2 * zm[late].mean() + 2 * zz[late].mean())
        out.append(float((slope - emission) / (4 * ff[late].mean() - 8 * zz[late].mean())))
    return out


def fit(series, times, traj, traj_strings, g):
    """Xb15uf structure with collective flips at rate g (fixed); returns theta and the collective slope beta."""
    ext, obs = extended_ansatz(), observables()
    fine = refine(times, N_SUB)
    d = trajectory_defect(fine, traj, traj_strings, times, T_LEARN)
    dt = trajectory_defect(fine, traj, traj_strings, times, T_LEARN, weight=lambda s: s)
    A, y, rows = integral_system(obs, ext, series, times, T_LEARN, defect=d)
    At, _, _ = integral_system(obs, ext, series, times, T_LEARN, defect=dt, weight=lambda s: s)
    Ac = cross_columns(series, rows, times, d)
    keep = np.all(np.isfinite(Ac), axis=1)
    T, fix, flip, coll = structure(ext, uniform=True)
    # single-ion part S_XX = S_YY = g is 2 g * flip; cross-ion part cXX = cYY = g
    target = (y - A @ fix - g * (2 * (A @ flip) + Ac[:, 0] + Ac[:, 1]))[keep]
    M = np.column_stack([(A @ T)[keep], (At @ coll)[keep]])
    phi = lsq_linear(M, target, method="bvls", tol=1e-12).x
    return T @ phi[:-1] + fix, float(phi[-1])


def model_of(theta, g):
    ext = extended_ansatz()
    H = sum(theta[m] * psparse(P, N) for m, P in enumerate(ext.h_terms) if theta[m] != 0.0).tocsr()
    jx = sum(psparse(((k, "X"),), N) for k in range(N)).tocsr()
    jy = sum(psparse(((k, "Y"),), N) for k in range(N)).tocsr()
    return OpenModel(H=H, Gamma=gamma_matrix(theta, ext), emission=np.full(N, EMISSION_D52),
                     jumps=(np.sqrt(g) * jx, np.sqrt(g) * jy))


def predict_job(args):
    theta, beta, g, s = args
    t0 = time.time()
    times, s1, s2, bases = load()
    obs = observables()
    rho = product_state(series_from(per_setting_tables((s1, s2), bases, [((i, a),) for i in range(N) for a in "XYZ"])), s)
    model = model_of(theta, g)
    D = 2**N
    gamma_c = float(theta[len(extended_ansatz().h_terms)])
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
    print(f"{VARIANT} state {s + 1}: {time.time() - t0:.0f} s", flush=True)
    return np.array(out)


def run():
    times, s1, s2, bases = load()
    obs, ans = observables(), ansatz()
    strings = sorted(set(obs) | set(needed_strings(obs, ans)))
    series = series_from(per_setting_tables((s1, s2), bases, strings))
    z = np.load(RESULTS / "e11_stage1.npz")
    traj = [z["traj0"], z["traj1"]]
    g = calibrate(series, times)
    fits = {s: fit(series, times, traj, strings, g[1 - s]) for s in (0, 1)}  # state s uses the other state's rate
    print("collective rates (rad/s):", [round(v * RAD_S, 3) for v in g], flush=True)
    jobs = [(fits[s][0], fits[s][1], g[1 - s], s) for s in (0, 1)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        preds = list(pool.map(predict_job, jobs))
    return g, fits, np.stack(preds)


def slopes(p, times):
    obs = observables()
    k = {O: m for m, O in enumerate(obs)}
    late = times >= T_LATE - 1e-9
    out = []
    for s in (0, 1):
        zz = np.mean([p[s][:, k[((i, "Z"), (j, "Z"))]] for i, j in PAIRS], axis=0)
        zm = np.mean([p[s][:, k[((i, "Z"),)]] for i in range(N)], axis=0)
        out.append([float(np.polyfit(times[late], zz[late], 1)[0]), float(np.polyfit(times[late], zm[late], 1)[0])])
    return out


def analyse(g, fits, pred):
    z = np.load(RESULTS / "e11_model_selection.npz")
    e20 = np.load(RESULTS / "e20_flip_models.npz")
    bands = np.load(RESULTS / "e15_coverage.npz")["band_Xb15"]
    times = z["times"]
    oos = times > T_LEARN + 1e-9
    preds = {"Xb15": z["pred_Xb15"], "Xb15uf": e20["pred_Xb15uf"], VARIANT: pred}
    fac = {v: factors(p, z, bands, oos) for v, p in preds.items()}
    out = {"collective flip rate g calibrated on each state (rad/s)": [v * RAD_S for v in g],
           "rate used to predict state 1, state 2 (rad/s)": [g[1] * RAD_S, g[0] * RAD_S],
           "calibration factors, t > 1.5 ms (Xb15 bootstrap spread)": fac}
    ext = extended_ansatz()
    k = {p: m for m, p in enumerate(ext.h_terms)}
    out["learned parameters (rad/s), fit for state 1 and state 2"] = {
        f"state {s + 1}": {"gamma_c": float(th[len(ext.h_terms)] * RAD_S), "beta (rad/s per ms)": float(b * RAD_S),
                           "uniform h_x, h_y": [float(th[k[((0, "X"),)]] * RAD_S), float(th[k[((0, "Y"),)]] * RAD_S)]}
        for s, (th, b) in fits.items()}
    out["late slopes per ms: mean ZZ (all pairs), mean Z; state 1, state 2"] = {
        "data": slopes(z["data"], times), **{v: slopes(p, times) for v, p in preds.items()}}
    ix = indices()
    ecols, ew = energy_weights(z["theta_Y15"])
    late = times >= 2.0 - 1e-9
    cq = {v: np.stack([conserved(p[s], ix, ecols, ew) for s in (0, 1)]) for v, p in preds.items()}
    cq["data"] = np.stack([conserved(z["data"][s], ix, ecols, ew) for s in (0, 1)])
    out["conserved quantities, mean over t >= 2 ms minus t = 0: mean Z (state 1), mean ZZ (state 2), energy (state 2)"] = {
        v: [float(c[0, late, 0].mean() - c[0, 0, 0]), float(c[1, late, 1].mean() - c[1, 0, 1]),
            float(c[1, late, 2].mean() - c[1, 0, 2])] for v, c in cq.items()}
    f = fac[VARIANT]
    neutral = [n for n in f if n.startswith("q=0")]
    out["verdict"] = {"all neutral classes <= 1.2": bool(all(f[n] <= 1.2 for n in neutral)),
                      "all 435 observables <= 1.15": bool(f["all 435 observables"] <= 1.15)}
    out["verdict"]["collective flips explain the neutral error"] = bool(all(out["verdict"].values()))
    return out


def plot(out, pred, times):
    plt = style()
    z = np.load(RESULTS / "e11_model_selection.npz")
    e20 = np.load(RESULTS / "e20_flip_models.npz")
    obs = observables()
    k = {O: m for m, O in enumerate(obs)}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
    ax = axes[0]
    fac = out["calibration factors, t > 1.5 ms (Xb15 bootstrap spread)"]
    classes = list(next(iter(fac.values())))
    for j, (v, c) in enumerate(zip(fac, [MUTED, CAT[0], CAT[1]])):
        ax.plot([fac[v][n] for n in classes], np.arange(len(classes)) + 0.15 * (j - 1), "D", color=c, ms=5, label=v)
    ax.axvline(1.0, color=MUTED, ls=":", lw=1)
    ax.axvline(1.2, color=MUTED, ls="--", lw=0.8)
    ax.set_yticks(range(len(classes)), classes)
    ax.invert_yaxis()
    ax.set_xlabel("calibration factor, t > 1.5 ms")
    ax.legend(fontsize=7)
    ax.set_title("(a) calibration by charge class", fontsize=9)
    ax = axes[1]
    zz = lambda p, s: np.mean([p[s][:, k[((i, "Z"), (j, "Z"))]] for i, j in PAIRS], axis=0)
    for s, mk, ls in ((0, "o", "-"), (1, "s", "--")):
        ax.plot(times, zz(z["data"], s) - zz(z["data"], s)[0], mk, color=INK, ms=3.5, label=f"data, state {s + 1}")
        for p, c, name in ((e20["pred_Xb15uf"], CAT[0], "Xb15uf"), (pred, CAT[1], VARIANT)):
            ax.plot(times, zz(p, s) - zz(p, s)[0], ls, color=c, lw=1.3, label=f"{name}, state {s + 1}")
    ax.axvspan(0.0, T_LEARN, color=MUTED, alpha=0.08, lw=0)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("mean ZZ over all pairs, change since t = 0")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("(b) $M^2$: independent versus collective flips", fontsize=9)
    fig.suptitle("E22b: collective flips calibrated on the other initial state (data: Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e22b_collective_flips")


if __name__ == "__main__":
    path = RESULTS / "e22b_collective_flips.npz"
    if "--replot" not in sys.argv:
        g, fits, pred = run()
        np.savez_compressed(path, g=np.array(g), pred=pred, **{f"theta_{s}": fits[s][0] for s in (0, 1)},
                            **{f"beta_{s}": fits[s][1] for s in (0, 1)})
    zz = np.load(path)
    fits = {s: (zz[f"theta_{s}"], float(zz[f"beta_{s}"])) for s in (0, 1)}
    out = analyse(list(zz["g"]), fits, zz["pred"])
    save_json(RESULTS / "e22b_collective_flips.json", out)
    print(plot(out, zz["pred"], np.load(RESULTS / "e11_model_selection.npz")["times"]))
    for key, v in out.items():
        print(f"{key}: {v}")
