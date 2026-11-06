"""E20: do incoherent flips and a uniform transverse field remove the 1.5x neutral-class error of the memory model?

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0), analysed
here from the public raw shots. E17 traced the neutral-class error of Xb15 (calibration factor about 1.5, E15) to
processes that break the conservation laws of the learned models; E19 found an incoherent flip rate of 2-3 /s that
agrees with the 51-ion data, and E11h(A) a flip burst in the first millisecond.
Variants, all learned on t <= 1.5 ms with the defect of E11 and predicted on 1.5 < t <= 10 ms. They share the Xb15
structure (U(1)-symmetric couplings, z fields per ion, collective dephasing gamma_c + beta t, D5/2 emission fixed):
  Xb15     per-ion transverse fields (E11, the reference)
  Xb15u    uniform transverse fields (two parameters instead of twenty)
  Xb15uf   uniform transverse fields and isotropic flips at the 51-ion rate S = 2.10 /s, fixed from the independent
           51-ion data (primary: a strict out-of-sample test)
  Xb15ub   Xb15uf plus the flip burst S(t) = S_inf + S_1 e^{-t/tau} of E11h(A) in place of the constant S. The burst
           was calibrated on the whole record of mean Z, so this variant is an adequacy check, not out of sample.
Flips are the Kossakowski entries S_XX = S_YY = S/2 on every ion, i.e. emission S and absorption S (qerr.lindblad).
Calibration factors as in E15, rms(data - prediction) / rms(sqrt(sd_boot^2 + SEM^2)), with sd_boot the bootstrap
spread of Xb15 from E15 for every variant (the variants have fewer or equal free parameters).
Fixed before the run: the flips and the uniform field explain the neutral error if the primary variant brings all three
neutral classes to a calibration factor <= 1.2 without raising the factor of all 435 observables above 1.15.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import gc  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.optimize import lsq_linear  # noqa: E402

from common import CAT, INK, MUTED, ORDINAL, RESULTS, save_figure, save_json, style  # noqa: E402
from e6_ion_learning import N, PAIRS, RAD_S, ansatz, observables, per_setting_tables, product_state, series_from  # noqa: E402
from e6c_validation import charge_classes, combine, combine_sem, load  # noqa: E402
from e9_symmetry_data import extended_ansatz  # noqa: E402
from e11_model_selection import EMISSION_D52, N_SUB  # noqa: E402
from e17_neutral_diagnosis import conserved, energy_weights, indices  # noqa: E402
from qerr.learning import NOISE_KEYS, gamma_matrix, integral_system, needed_strings, refine, trajectory_defect  # noqa: E402
from qerr.lindblad import OpenModel, lindblad_rhs  # noqa: E402
from qerr.pauli import expectation_tables, expectations, psparse  # noqa: E402

S_51 = 2.0992815708627806 / RAD_S  # 1/ms, E19 (51-ion balanced rate)
VARIANTS = ("Xb15u", "Xb15uf", "Xb15ub")
PRIMARY = "Xb15uf"
T_LEARN, T_LEARN_END = 1.5, 1.5
_W = {}


def burst():
    """(S_inf, S_1, tau) of the flip burst from E11h(A), in 1/ms and ms."""
    from e11h_transient_truths import calibrate
    cal = calibrate()["flip burst"]
    p = cal["params (rad/s, 1/s)"]
    return p[2] / RAD_S, p[3] / RAD_S, cal["tau (ms)"]


def structure(ext, uniform):
    """theta_ext = T phi + fix: Xb15 structure with per-ion or uniform transverse fields; emission in fix."""
    k = {p: m for m, p in enumerate(ext.h_terms)}
    k0, nd = len(ext.h_terms), len(ext.deph_pairs)
    kn = k0 + nd
    cols = []

    def col(entries):
        c = np.zeros(ext.n_params)
        for m, v in entries:
            c[m] += v
        cols.append(c)

    for i, j in PAIRS:
        xx, yy, xy, yx = (k[((i, a), (j, b))] for a, b in ("XX", "YY", "XY", "YX"))
        col([(xx, 1.0), (yy, 1.0)])
        col([(xy, 1.0), (yx, -1.0)])
    for i in range(N):
        col([(k[((i, "Z"),)], 1.0)])
    if uniform:
        for a in "XY":
            col([(k[((i, a),)], 1.0) for i in range(N)])
    else:
        for i in range(N):
            for a in "XY":
                col([(k[((i, a),)], 1.0)])
    col([(k0 + m, 1.0) for m in range(nd)])
    fix = np.zeros(ext.n_params)
    u = {key: n for n, key in enumerate(NOISE_KEYS)}
    for i in range(N):
        for key in ("SXX", "SYY", "TXY"):
            fix[kn + len(NOISE_KEYS) * i + u[key]] += EMISSION_D52 / 4
    flip = np.zeros(ext.n_params)  # unit flip rate: S_XX = S_YY = 1/2 on every ion
    for i in range(N):
        flip[kn + len(NOISE_KEYS) * i + u["SXX"]] = flip[kn + len(NOISE_KEYS) * i + u["SYY"]] = 0.5
    collective = np.zeros(ext.n_params)
    collective[k0:k0 + nd] = 1.0
    return np.array(cols).T, fix, flip, collective


def fit(variant, series, times, traj, traj_strings, bp):
    ext, obs = extended_ansatz(), observables()
    fine = refine(times, N_SUB)
    d = trajectory_defect(fine, traj, traj_strings, times, T_LEARN)
    dt = trajectory_defect(fine, traj, traj_strings, times, T_LEARN, weight=lambda s: s)
    A, y, _ = integral_system(obs, ext, series, times, T_LEARN, defect=d)
    At, _, _ = integral_system(obs, ext, series, times, T_LEARN, defect=dt, weight=lambda s: s)
    T, fix, flip, coll = structure(ext, uniform=variant != "Xb15")
    target = y - A @ fix
    if variant == "Xb15uf":
        target = target - S_51 * (A @ flip)
    if variant == "Xb15ub":
        S_inf, S_1, tau = bp
        wb = lambda s: np.exp(-s / tau)
        db = trajectory_defect(fine, traj, traj_strings, times, T_LEARN, weight=wb)
        Ab, _, _ = integral_system(obs, ext, series, times, T_LEARN, defect=db, weight=wb)
        target = target - S_inf * (A @ flip) - S_1 * (Ab @ flip)
    M = np.column_stack([A @ T, At @ coll])
    phi = lsq_linear(M, target, method="bvls", tol=1e-12).x
    theta = T @ phi[:-1] + fix
    return theta, float(phi[-1])


def model_of(theta, variant, bp):
    ext = extended_ansatz()
    H = sum(theta[m] * psparse(P, N) for m, P in enumerate(ext.h_terms) if theta[m] != 0.0).tocsr()
    S = {"Xb15uf": S_51, "Xb15ub": bp[0]}.get(variant, 0.0)
    return OpenModel(H=H, Gamma=gamma_matrix(theta, ext), emission=np.full(N, EMISSION_D52 + S),
                     absorption=np.full(N, S))


def predict_job(args):
    variant, theta, beta, s, bp = args
    t0 = time.time()
    times, s1, s2, bases = load()
    obs = observables()
    rho = product_state(series_from(per_setting_tables((s1, s2), bases, [((i, a),) for i in range(N) for a in "XYZ"])), s)
    model = model_of(theta, variant, bp)
    D = 2**N
    gamma_c = float(theta[len(extended_ansatz().h_terms)])
    Wc = OpenModel(H=sp.csr_matrix((D, D)), Gamma=np.ones((N, N)), emission=np.zeros(N))._elementwise
    fm = OpenModel(H=sp.csr_matrix((D, D)), Gamma=np.zeros((N, N)), emission=np.ones(N), absorption=np.ones(N))
    S_1, tau = (bp[1], bp[2]) if variant == "Xb15ub" else (0.0, 1.0)
    idx, ph = expectation_tables(obs, N)
    out = [expectations(rho, idx, ph)]
    for a, b in zip(times[:-1], times[1:]):
        def rhs(u, y, a=a):
            X = y.reshape(D, D)
            r = lindblad_rhs(model, X) + max(beta * (a + u), -gamma_c) * Wc * X
            if S_1 != 0.0:
                r = r + S_1 * np.exp(-(a + u) / tau) * lindblad_rhs(fm, X)
            return r.ravel()
        sol = solve_ivp(rhs, (0.0, b - a), rho.ravel(), t_eval=[b - a], method="DOP853", rtol=1e-7, atol=1e-9)
        rho = sol.y[:, -1].reshape(D, D)
        del sol
        gc.collect()  # the solver's stage arrays sit in reference cycles (see qerr.lindblad.evolve)
        out.append(expectations(rho, idx, ph))
    print(f"{variant} state {s + 1}: {time.time() - t0:.0f} s", flush=True)
    return np.array(out)


def run():
    times, s1, s2, bases = load()
    obs, ans = observables(), ansatz()
    strings = sorted(set(obs) | set(needed_strings(obs, ans)))
    series = series_from(per_setting_tables((s1, s2), bases, strings))
    z = np.load(RESULTS / "e11_stage1.npz")
    traj_strings = needed_strings(obs, ans)
    bp = burst()
    fits = {v: fit(v, series, times, [z["traj0"], z["traj1"]], traj_strings, bp) for v in VARIANTS}
    print("fits:", {v: (round(f[1] * RAD_S, 2), round(f[0][len(extended_ansatz().h_terms)] * RAD_S, 2))
                    for v, f in fits.items()}, flush=True)
    jobs = [(v, fits[v][0], fits[v][1], s, bp) for v in VARIANTS for s in (0, 1)]
    with ProcessPoolExecutor(max_workers=6) as pool:
        preds = list(pool.map(predict_job, jobs))
    pred = {v: np.stack(preds[2 * i:2 * i + 2]) for i, v in enumerate(VARIANTS)}
    return fits, pred


def factors(pred, z, sd_boot, oos):
    cls = dict(charge_classes(observables()), **{"all 435 observables": [{m: 1.0} for m in range(z["data"].shape[-1])]})
    out = {}
    for name, coeffs in cls.items():
        d, p, s = combine(z["data"], coeffs), combine(pred, coeffs), combine_sem(z["sem"], coeffs)
        sd = combine(sd_boot, coeffs).std(axis=0, ddof=1)
        tot, dev = np.sqrt(sd**2 + s**2), d - p
        out[name] = float(np.sqrt(np.mean(dev[:, oos] ** 2) / np.mean(tot[:, oos] ** 2)))
    return out


def analyse(fits, pred):
    z = np.load(RESULTS / "e11_model_selection.npz")
    bands = np.load(RESULTS / "e15_coverage.npz")["band_Xb15"]
    times = z["times"]
    oos = times > T_LEARN + 1e-9
    preds = {"Xb15": z["pred_Xb15"], **pred}
    out = {"calibration factors, t > 1.5 ms (Xb15 bootstrap spread for every variant)":
           {v: factors(p, z, bands, oos) for v, p in preds.items()}}
    ext = extended_ansatz()
    k = {p: m for m, p in enumerate(ext.h_terms)}
    out["learned parameters (rad/s)"] = {
        v: {"gamma_c": float(th[len(ext.h_terms)] * RAD_S), "beta (rad/s per ms)": float(b * RAD_S),
            "uniform h_x, h_y": [float(th[k[((0, "X"),)]] * RAD_S), float(th[k[((0, "Y"),)]] * RAD_S)]}
        for v, (th, b) in fits.items()}
    ix = indices()
    ecols, ew = energy_weights(z["theta_Y15"])
    late = times >= 2.0 - 1e-9
    cq = {v: np.stack([conserved(p[s], ix, ecols, ew) for s in (0, 1)]) for v, p in preds.items()}
    cq["data"] = np.stack([conserved(z["data"][s], ix, ecols, ew) for s in (0, 1)])
    out["conserved quantities, mean over t >= 2 ms minus t = 0: mean Z (state 1), mean ZZ (state 2), energy (state 2)"] = {
        v: [float(c[0, late, 0].mean() - c[0, 0, 0]), float(c[1, late, 1].mean() - c[1, 0, 1]),
            float(c[1, late, 2].mean() - c[1, 0, 2])] for v, c in cq.items()}
    f = out["calibration factors, t > 1.5 ms (Xb15 bootstrap spread for every variant)"][PRIMARY]
    neutral = [n for n in f if n.startswith("q=0")]
    out["verdict"] = {"primary": PRIMARY,
                      "all neutral classes <= 1.2": bool(all(f[n] <= 1.2 for n in neutral)),
                      "all 435 observables <= 1.15": bool(f["all 435 observables"] <= 1.15)}
    out["verdict"]["flips and a uniform field explain the neutral error"] = bool(
        out["verdict"]["all neutral classes <= 1.2"] and out["verdict"]["all 435 observables <= 1.15"])
    return out, cq


def plot(out, cq, times):
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.8))
    ax = axes[0]
    fac = out["calibration factors, t > 1.5 ms (Xb15 bootstrap spread for every variant)"]
    classes = list(next(iter(fac.values())))
    for j, (v, c) in enumerate(zip(fac, [INK, ORDINAL[1], CAT[1], CAT[2]])):
        ax.plot([fac[v][n] for n in classes], np.arange(len(classes)) + 0.12 * (j - 1.5), "D", color=c, ms=5, label=v)
    ax.axvline(1.0, color=MUTED, ls=":", lw=1)
    ax.axvline(1.2, color=MUTED, ls="--", lw=0.8)
    ax.set_yticks(range(len(classes)), classes)
    ax.invert_yaxis()
    ax.set_xlabel("calibration factor, t > 1.5 ms")
    ax.legend(fontsize=7)
    ax.set_title("(a) calibration by charge class", fontsize=9)
    ax = axes[1]
    for v, c in zip(["data", "Xb15", "Xb15uf", "Xb15ub"], [INK, MUTED, CAT[1], CAT[2]]):
        y = cq[v][0, :, 0] - cq[v][0, 0, 0]
        ax.plot(times, y, "o" if v == "data" else "-", color=c, ms=3.5, lw=1.4, label=v)
    ax.axvspan(0.0, T_LEARN, color=MUTED, alpha=0.08, lw=0)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("mean Z change since t = 0, initial state 1")
    ax.legend(fontsize=7)
    ax.set_title("(b) the conserved magnetisation", fontsize=9)
    fig.suptitle("E20: flips and a uniform field in the memory model (data: Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e20_flip_models")


if __name__ == "__main__":
    if "--replot" not in sys.argv:
        fits, pred = run()
        np.savez_compressed(RESULTS / "e20_flip_models.npz", **{f"pred_{v}": p for v, p in pred.items()},
                            **{f"theta_{v}": f[0] for v, f in fits.items()}, **{f"beta_{v}": f[1] for v, f in fits.items()})
    zz = np.load(RESULTS / "e20_flip_models.npz")
    fits = {v: (zz[f"theta_{v}"], float(zz[f"beta_{v}"])) for v in VARIANTS}
    pred = {v: zz[f"pred_{v}"] for v in VARIANTS}
    out, cq = analyse(fits, pred)
    save_json(RESULTS / "e20_flip_models.json", out)
    print(plot(out, cq, np.load(RESULTS / "e11_model_selection.npz")["times"]))
    for key, v in out.items():
        print(f"{key}: {v}")
