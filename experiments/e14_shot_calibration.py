"""E14: shot-level calibration of the memory test and a noise-robust re-certification of the collective protection.

E11b calibrated the integral method with independent Gaussian noise. Here synthetic data are generated exactly like the
BEQS 10-ion data (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0): the
same 200 inferred random Pauli settings, 200 shots per setting and time, 21 times, both initial states, sampled from
the exact outcome distributions (qerr.shots). Shot noise and its correlations between strings measured in the same
setting are therefore those of the experiment.
Truths: the Y15 generator of E11 (U(1)-symmetric couplings and z fields, D5/2 emission) with a collective rate that
mixes white and quasi-static noise, gamma_c(t) = (1 - f) gamma_c + f beta_s t, beta_s = 2 gamma_c / 1.5 ms, for a
quasi-static fraction f in {0, 1/4, 1/2, 3/4, 1}; every truth accumulates the same collective decay over 1.5 ms.
The truths are covariant, so by Lemma G.0 one evolution without collective dephasing gives all of them (charge-q
blocks times exp(-2 q^2 (1 - f) gamma_c t - q^2 f beta_s t^2)).
Each synthetic data set goes through the measured-data pipeline: Xb15 and Xb30 (slope beta of the collective rate)
and the full ansatz on the windows of E11b (J drift). Outputs: realistic null and alternative distributions of
beta, a Neyman-belt confidence interval for f from the measured beta, the J-drift artefact under realistic noise, and
the Zeno factors at 10 ms over the noise models the data allow (re-certification of E9's white-noise statement).
"""

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import CAT, INK, MUTED, ORDINAL, RESULTS, save_figure, save_json, style
from e6_ion_learning import N, RAD_S, ansatz, observables, per_setting_tables, physical, product_state, series_from
from e6c_validation import load
from e11_model_selection import fit_variant
from qerr import spins
from qerr.learning import fit, integral_system, needed_strings, refine, trajectory_defect
from qerr.lindblad import evolve
from qerr.pauli import expectation_tables, expectations
from qerr.shots import pauli_tensor, sample_spins, setting_probabilities

FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
WINDOWS = [(0.0, 1.5), (0.0, 3.0), (0.0, 5.0), (1.0, 3.0), (3.0, 5.0), (5.0, 10.0)]
MEMORY = ["Xb15", "Xb30"]
N_REAL, N_SUB, SHOTS, GAMMA_LB = 20, 4, 200, 20.7 / RAD_S   # E9's certified collective rate, rad/ms
_W = {}


def truth_theta():
    return np.load(RESULTS / "e11_stage1.npz")["theta_Y15"]


def chi_exponent(f, gamma_c, t):
    """w(t)/2 per unit q^2: chi_q(t) = exp(-q^2 x) with x = 2(1-f) gamma_c t + f beta_s t^2."""
    return 2.0 * (1.0 - f) * gamma_c * t + f * (2.0 * gamma_c / 1.5) * t * t


def truth_job(s):
    """Fine-grid string values and coarse-time outcome distributions of every truth, initial state s."""
    times, s1, s2, bases = load()
    ans, obs = ansatz(), observables()
    rho = product_state(series_from(per_setting_tables((s1, s2), bases, [((i, a),) for i in range(N) for a in "XYZ"])), s)
    strings = sorted(set(obs) | set(needed_strings(obs, ans)))
    theta = truth_theta().copy()
    k0 = len(ans.h_terms)
    gamma_c = theta[k0]
    theta[k0:k0 + len(ans.deph_pairs)] = 0.0
    model = physical(theta, ans)
    m = spins.total_sz(N)
    q2 = (m[:, None] - m[None, :]) ** 2
    idx, ph = expectation_tables(strings, N)
    fine = refine(times, N_SUB)
    coarse = set(np.searchsorted(fine, times - 1e-12).tolist())
    traj = {f: [] for f in FRACTIONS}
    prob = {f: [] for f in FRACTIONS}
    for k, t in enumerate(fine):
        if k:
            rho = evolve(model, rho, [0.0, t - fine[k - 1]], rtol=1e-9, atol=1e-11)[-1]
        for f in FRACTIONS:
            rf = rho * np.exp(-q2 * chi_exponent(f, gamma_c, t))
            traj[f].append(expectations(rf, idx, ph))
            if k in coarse:
                E = pauli_tensor(rf, N)
                prob[f].append(np.array([setting_probabilities(E, b) for b in bases], np.float32))
    return {f: (np.array(traj[f]), np.array(prob[f])) for f in FRACTIONS}


def defects_for(traj, strings, times):
    fine = refine(times, N_SUB)
    d = {(w, False): trajectory_defect(fine, traj, strings, times, w[1], t_start=w[0]) for w in WINDOWS}
    for win in (1.5, 3.0):
        d[((0.0, win), True)] = trajectory_defect(fine, traj, strings, times, win, weight=lambda s: s)
    return d


def fit_all(series, times, defects):
    """J-relevant full fits on every window and (beta, gamma_c) of the memory variants (as in E11b)."""
    ans, obs = ansatz(), observables()
    full, systems = [], {}
    for ta, tb in WINDOWS:
        A, y, _ = integral_system(obs, ans, series, times, tb, t_start=ta, defect=defects[((ta, tb), False)])
        full.append(fit(A, y))
        if ta == 0.0 and tb in (1.5, 3.0):
            At, _, _ = integral_system(obs, ans, series, times, tb, defect=defects[((0.0, tb), True)],
                                       weight=lambda s: s)
            systems[tb] = (A, y, At)
    mem = []
    for v in MEMORY:
        theta, beta, _ = fit_variant(v, systems, ans)
        mem.append((beta, theta[len(ans.h_terms)]))
    return np.array(full), np.array(mem)


def _init(prob, defects, bases, strings, times):
    _W.update(prob=prob, defects=defects, bases=bases, strings=strings, times=times)


def realisation(args):
    f, seed = args
    rng = np.random.default_rng(seed)
    data = []
    for s in (0, 1):
        p = _W["prob"][f][s]                                              # (21, settings, 1024)
        sp = np.empty((p.shape[0], p.shape[1], SHOTS, N), np.int8)
        for t in range(p.shape[0]):
            for k in range(p.shape[1]):
                sp[t, k] = sample_spins(p[t, k], SHOTS, rng, N)
        data.append(sp)
    series = series_from(per_setting_tables(tuple(data), _W["bases"], _W["strings"]))
    return fit_all(series, _W["times"], _W["defects"][f])


def jnn(theta, ans):
    k = {p: m for m, p in enumerate(ans.h_terms)}
    return np.mean([theta[k[((i, "X"), (i + 1, "X"))]] + theta[k[((i, "Y"), (i + 1, "Y"))]] for i in range(N - 1)])


def run():
    times, s1, s2, bases = load()
    ans, obs = ansatz(), observables()
    strings = sorted(set(obs) | set(needed_strings(obs, ans)))
    with ProcessPoolExecutor(max_workers=2) as pool:
        per_state = list(pool.map(truth_job, (0, 1)))
    prob = {f: [per_state[s][f][1] for s in (0, 1)] for f in FRACTIONS}
    defects = {f: defects_for([per_state[s][f][0] for s in (0, 1)], strings, times) for f in FRACTIONS}
    del per_state
    jobs = [(f, 104729 * i + int(1000 * f)) for f in FRACTIONS for i in range(N_REAL)]
    with ProcessPoolExecutor(max_workers=8, initializer=_init, initargs=(prob, defects, bases, strings, times)) as pool:
        fits = list(pool.map(realisation, jobs))
    measured = series_from(per_setting_tables((s1, s2), bases, strings))
    data = {f: fit_all(measured, times, defects[f]) for f in (0.0, 0.5)}
    return jobs, fits, data


def belt_interval(fs, means, sds, value):
    """Neyman belt from mean +- 1.96 sd at each f (linear interpolation): estimate and 95% interval for f."""
    grid = np.linspace(fs[0], fs[-1], 401)
    mu, sd = np.interp(grid, fs, means), np.interp(grid, fs, sds)
    inside = np.abs(value - mu) <= 1.96 * sd
    est = float(np.interp(value, means, fs)) if np.all(np.diff(means) > 0) else float("nan")
    return est, ([float(grid[inside].min()), float(grid[inside].max())] if inside.any() else None)


def zeno_factor(q, f, gamma, t=10.0, n=4001):
    b = np.linspace(0.0, t, n)
    return float(np.trapezoid((t - b) * np.exp(-q * q * chi_exponent(f, gamma, b)), b) / (0.5 * t * t))


def analyse(jobs, fits, data):
    ans = ansatz()
    fr = np.array([j[0] for j in jobs])
    J = np.array([[jnn(th, ans) * RAD_S for th in f[0]] for f in fits])
    B = np.array([f[1][:, 0] * RAD_S for f in fits])
    out = {"truth J_nn (rad/s)": float(jnn(truth_theta(), ans) * RAD_S),
           "truth gamma_c (rad/s), beta_s (rad/s per ms)": [float(truth_theta()[len(ans.h_terms)] * RAD_S),
                                                           float(2 * truth_theta()[len(ans.h_terms)] / 1.5 * RAD_S)],
           "windows": [str(w) for w in WINDOWS], "fractions": FRACTIONS}
    means = np.array([B[fr == f].mean(axis=0) for f in FRACTIONS])
    sds = np.array([B[fr == f].std(axis=0, ddof=1) for f in FRACTIONS])
    out["beta mean per f (Xb15, Xb30)"] = means.tolist()
    out["beta sd per f (Xb15, Xb30)"] = sds.tolist()
    out["J_nn mean per f and window"] = [J[fr == f].mean(axis=0).tolist() for f in FRACTIONS]
    out["J_nn sd per f and window"] = [J[fr == f].std(axis=0, ddof=1).tolist() for f in FRACTIONS]
    meas = {f: (np.array([jnn(th, ans) * RAD_S for th in d[0]]), d[1][:, 0] * RAD_S) for f, d in data.items()}
    out["measured J_nn per window (white-truth defect)"] = meas[0.0][0].tolist()
    out["measured beta (Xb15, Xb30) with the white / f=1/2 defect"] = [meas[0.0][1].tolist(), meas[0.5][1].tolist()]
    est = {}
    for i, v in enumerate(MEMORY):
        value = meas[0.0][1][i]
        f_hat, ci = belt_interval(np.array(FRACTIONS), means[:, i], sds[:, i], value)
        est[v] = {"z under white noise": float((value - means[0, i]) / sds[0, i]),
                  "z under pure quasi-static noise": float((value - means[-1, i]) / sds[-1, i]),
                  "quasi-static fraction estimate": f_hat, "95% interval": ci}
    out["memory test with shot-level calibration"] = est
    ci_all = [c for e in est.values() if e["95% interval"] for c in e["95% interval"]]
    f_lo, f_hi = (min(ci_all), max(ci_all)) if ci_all else (0.0, 1.0)
    cert = {}
    for label, gamma in (("certified scale gamma_c >= 20.7 rad/s (E9)", GAMMA_LB),
                         ("point scale gamma_c = Y15", truth_theta()[len(ans.h_terms)])):
        rows = {}
        for q in (1, 2):
            vals = {f"f={f:g}": zeno_factor(q, f, gamma) for f in FRACTIONS}
            span = np.linspace(f_lo, f_hi, 21)
            vals["worst case over the 95% interval of f"] = max(zeno_factor(q, f, gamma) for f in span)
            rows[f"|Q|={q}"] = vals
        cert[label] = rows
    out["Zeno factor at 10 ms (window-matched mixtures)"] = cert
    out["acceptance"] = {
        "noise-free check not run here; E11b covered it": True,
        "white collective noise rejected at > 3 sd in both windows": bool(all(e["z under white noise"] > 3
                                                                              for e in est.values())),
        "beta increases monotonically with f in both windows": bool(np.all(np.diff(means, axis=0) > 0)),
    }
    return out, fr, J, B, meas


def plot(fr, J, B, meas, out):
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))
    for i, (ax, v) in enumerate(zip(axes[:2], MEMORY)):
        for f, c in zip(FRACTIONS, ORDINAL + [CAT[1]]):
            ax.plot(np.full((fr == f).sum(), f), B[fr == f][:, i], "o", color=c, ms=3.5, alpha=0.6)
        mu = [B[fr == f][:, i].mean() for f in FRACTIONS]
        sd = [B[fr == f][:, i].std(ddof=1) for f in FRACTIONS]
        ax.fill_between(FRACTIONS, np.array(mu) - 1.96 * np.array(sd), np.array(mu) + 1.96 * np.array(sd),
                        color=MUTED, alpha=0.15, lw=0, label="95% belt (synthetic)")
        ax.plot(FRACTIONS, mu, color=MUTED, lw=1)
        ax.axhline(meas[0.0][1][i], color=INK, lw=2, label="BEQS data")
        e = out["memory test with shot-level calibration"][v]
        if e["95% interval"]:
            ax.axvspan(*e["95% interval"], color=CAT[2], alpha=0.12, lw=0, label="95% interval for f")
        ax.set_xlabel("quasi-static fraction f of the truth")
        ax.set_ylabel("fitted slope $\\beta$ (rad/s per ms)")
        ax.set_title(f"({'ab'[i]}) {v}: shot-level calibration")
        ax.legend(fontsize=7, loc="upper left")
    ax = axes[2]
    x = np.arange(len(WINDOWS))
    for f, c, mk in ((0.0, CAT[0], "o"), (1.0, CAT[1], "^")):
        ax.errorbar(x + (0.1 if f else -0.1), J[fr == f].mean(axis=0), yerr=J[fr == f].std(axis=0, ddof=1), fmt=mk,
                    color=c, ms=5, capsize=2, label=f"synthetic, f = {f:g}")
    ax.plot(x, meas[0.0][0], "D", mfc="white", mec=INK, ms=6, label="BEQS data")
    ax.axhline(out["truth J_nn (rad/s)"], color=MUTED, ls=":", label="truth")
    ax.set_xticks(x, [f"[{a:g},{b:g}]" for a, b in WINDOWS], fontsize=7)
    ax.set_xlabel("learning window (ms)")
    ax.set_ylabel("learned nearest-neighbour J (rad/s)")
    ax.set_title("(c) J drift under shot-level noise")
    ax.legend(fontsize=7)
    fig.suptitle("E14: shot-level calibration of the memory test (data: Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e14_shot_calibration")


if __name__ == "__main__":
    if "--replot" not in sys.argv:
        jobs, fits, data = run()
        np.savez_compressed(RESULTS / "e14_shot_calibration.npz", jobs=np.array(jobs),
                            full=np.array([f[0] for f in fits]), mem=np.array([f[1] for f in fits]),
                            data_full=np.array([data[f][0] for f in (0.0, 0.5)]),
                            data_mem=np.array([data[f][1] for f in (0.0, 0.5)]))
    z = np.load(RESULTS / "e14_shot_calibration.npz")
    jobs = [tuple(j) for j in z["jobs"]]
    fits = list(zip(z["full"], z["mem"]))
    data = {f: (z["data_full"][i], z["data_mem"][i]) for i, f in enumerate((0.0, 0.5))}
    out, fr, J, B, meas = analyse(jobs, fits, data)
    save_json(RESULTS / "e14_shot_calibration.json", out)
    print(plot(fr, J, B, meas, out))
    for k, v in out.items():
        print(f"{k}: {v}")
