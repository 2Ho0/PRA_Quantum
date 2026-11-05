"""E11b: simulation-based calibration of the integral method on BEQS-like data (window drift and memory test).

Truths, started from the measured product states of the BEQS data (measured by Kraft et al., PRX 16, 031037 (2026);
Zenodo 10.5281/zenodo.17750419, CC BY 4.0):
  white   the Y15 generator of E11: U(1)-symmetric couplings and z fields, constant collective dephasing gamma_c,
          emission at the D5/2 rate;
  static  the same Hamiltonian and emission, with purely quasi-static collective noise: gamma_c(t) = beta_s t, beta_s =
          2 gamma_c / 1.5 ms, so that the collective decay accumulated over the 1.5 ms window is the same.
Both truths are covariant, so by Lemma G.0 they follow from one evolution without collective dephasing: the charge-q
block of the state is multiplied by exp(-2 gamma_c q^2 t) (white) or exp(-beta_s q^2 t^2) (static).
Synthetic data: the exact expectation value of every string the integral method uses, plus independent Gaussian noise
with the empirical standard error of each (state, string, time) of the measured data; 20 realisations per case, a
noise-free control, and a quarter-noise variant for the white truth. Fits, each with the exact quadrature defect of its
truth: the full E6a ansatz on the cumulative windows t <= 1.5, 3, 5 ms and on the late windows [1, 3], [3, 5], [5, 10]
ms; and the Xb15, Xb30 variants of E11 (collective rate gamma_c + beta t). The measured data are refitted with the same
procedure (defect of the white truth).
Questions: (1) is the window dependence of J_nn reproduced by a time-independent truth (errors-in-variables bias of
quasi-stationary late data)? (2) where does the measured beta fall in the null distribution (white truth) and in the
alternative (static truth)?
"""

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import CAT, INK, MUTED, ORDINAL, RESULTS, save_figure, save_json, style
from e6_ion_learning import N, RAD_S, ansatz, observables, per_setting_tables, physical, product_state, series_from
from e6c_validation import load
from e11_model_selection import fit_variant
from qerr import spins
from qerr.iondata import pauli_means
from qerr.learning import fit, integral_system, needed_strings, refine, trajectory_defect
from qerr.lindblad import evolve
from qerr.pauli import expectation_tables, expectations

WINDOWS = [(0.0, 1.5), (0.0, 3.0), (0.0, 5.0), (1.0, 3.0), (3.0, 5.0), (5.0, 10.0)]
MEMORY = ["Xb15", "Xb30"]
N_REAL, N_SUB = 20, 4
CASES = [("white", 0.0), ("white", 0.25), ("white", 1.0), ("static", 0.0), ("static", 1.0)]
_W = {}


def truth_theta():
    return np.load(RESULTS / "e11_stage1.npz")["theta_Y15"]


def static_slope(ans):
    return 2.0 * truth_theta()[len(ans.h_terms)] / 1.5  # rad/ms per ms


def truth_job(s):
    """Fine-grid values of every needed string for both truths and initial state s (one evolution, Lemma G.0)."""
    times, s1, s2, bases = load()
    ans, obs = ansatz(), observables()
    tables = per_setting_tables((s1, s2), bases, [((i, a),) for i in range(N) for a in "XYZ"])
    rho = product_state(series_from(tables), s)
    strings = sorted(set(obs) | set(needed_strings(obs, ans)))
    fine = refine(times, N_SUB)
    theta = truth_theta().copy()
    k0 = len(ans.h_terms)
    gamma_c, beta_s = theta[k0], static_slope(ans)
    theta[k0:k0 + len(ans.deph_pairs)] = 0.0  # the collective noise enters through chi_q below
    model = physical(theta, ans)
    m = spins.total_sz(N)
    q2 = (m[:, None] - m[None, :]) ** 2
    idx, ph = expectation_tables(strings, N)
    out = {"white": [], "static": []}
    for k, t in enumerate(fine):
        if k:
            rho = evolve(model, rho, [0.0, t - fine[k - 1]], rtol=1e-9, atol=1e-11)[-1]
        out["white"].append(expectations(rho * np.exp(-2.0 * gamma_c * q2 * t), idx, ph))
        out["static"].append(expectations(rho * np.exp(-beta_s * q2 * t * t), idx, ph))
    return {kind: np.array(v) for kind, v in out.items()}


def defects_for(traj, strings, times):
    fine = refine(times, N_SUB)
    d = {(w, False): trajectory_defect(fine, traj, strings, times, w[1], t_start=w[0]) for w in WINDOWS}
    for win in (1.5, 3.0):
        d[((0.0, win), True)] = trajectory_defect(fine, traj, strings, times, win, weight=lambda s: s)
    return d


def fit_all(series, times, defects):
    """J-relevant full fits on every window, and (beta, gamma_c) of the memory variants."""
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


def _init(exact, sem, strings, times, defects):
    _W.update(exact=exact, sem=sem, strings=strings, times=times, defects=defects)


def realisation(args):
    kind, level, seed = args
    rng = np.random.default_rng(seed)
    series = {}
    for s in (0, 1):
        ex = _W["exact"][kind][s]
        noisy = ex + level * rng.normal(size=ex.shape) * np.nan_to_num(_W["sem"][s])
        noisy[:, np.isnan(_W["sem"][s][0])] = np.nan  # strings no setting measures stay missing, as in the data
        for k, S in enumerate(_W["strings"]):
            series[(s, S)] = noisy[:, k]
    return fit_all(series, _W["times"], _W["defects"][kind])


def jnn(theta, ans):
    k = {p: m for m, p in enumerate(ans.h_terms)}
    return np.mean([theta[k[((i, "X"), (i + 1, "X"))]] + theta[k[((i, "Y"), (i + 1, "Y"))]] for i in range(N - 1)])


def run():
    times, s1, s2, bases = load()
    ans, obs = ansatz(), observables()
    strings = sorted(set(obs) | set(needed_strings(obs, ans)))
    sem = [pauli_means(sp, bases, strings)[1] for sp in (s1, s2)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        per_state = list(pool.map(truth_job, (0, 1)))
    tr = {(k, s): per_state[s][k] for k in ("white", "static") for s in (0, 1)}
    fine = refine(times, N_SUB)
    at = np.searchsorted(fine, times - 1e-12)
    exact = {k: [tr[(k, s)][at] for s in (0, 1)] for k in ("white", "static")}
    defects = {k: defects_for([tr[(k, s)] for s in (0, 1)], strings, times) for k in ("white", "static")}
    jobs = [(k, lvl, 7919 * i + int(100 * lvl) + (0 if k == "white" else 50000))
            for k, lvl in CASES for i in range(1 if lvl == 0.0 else N_REAL)]
    with ProcessPoolExecutor(max_workers=8, initializer=_init, initargs=(exact, sem, strings, times, defects)) as pool:
        fits = list(pool.map(realisation, jobs))
    data = fit_all(series_from(per_setting_tables((s1, s2), bases, strings)), times, defects["white"])
    return jobs, fits, data


def analyse(jobs, fits, data):
    ans = ansatz()
    theta = truth_theta()
    out = {"truth J_nn (rad/s)": float(jnn(theta, ans) * RAD_S),
           "truth gamma_c (rad/s), static slope beta_s (rad/s per ms)": [float(theta[len(ans.h_terms)] * RAD_S),
                                                                        float(static_slope(ans) * RAD_S)],
           "windows": [str(w) for w in WINDOWS]}
    J = np.array([[jnn(th, ans) * RAD_S for th in f[0]] for f in fits])
    B = np.array([f[1][:, 0] * RAD_S for f in fits])      # (n_jobs, 2): beta of Xb15, Xb30
    G = np.array([f[1][:, 1] * RAD_S for f in fits])
    key = [(j[0], j[1]) for j in jobs]
    for case in CASES:
        sel = np.array([k == case for k in key])
        name = f"{case[0]} truth, noise x{case[1]}"
        out[f"{name}: J_nn mean per window"] = [float(x) for x in J[sel].mean(axis=0)]
        out[f"{name}: beta (Xb15, Xb30) mean"] = [float(x) for x in B[sel].mean(axis=0)]
        out[f"{name}: gamma_c (Xb15, Xb30) mean"] = [float(x) for x in G[sel].mean(axis=0)]
        if sel.sum() > 1:
            out[f"{name}: J_nn sd per window"] = [float(x) for x in J[sel].std(axis=0, ddof=1)]
            out[f"{name}: beta (Xb15, Xb30) sd"] = [float(x) for x in B[sel].std(axis=0, ddof=1)]
    dJ = [float(jnn(th, ans) * RAD_S) for th in data[0]]
    dB = [float(x) for x in data[1][:, 0] * RAD_S]
    out["measured: J_nn per window"] = dJ
    out["measured: beta (Xb15, Xb30)"] = dB
    out["measured: gamma_c (Xb15, Xb30)"] = [float(x) for x in data[1][:, 1] * RAD_S]
    null = np.array([k == ("white", 1.0) for k in key])
    alt = np.array([k == ("static", 1.0) for k in key])
    out["percentile of the measured beta in the white-truth (null) distribution (Xb15, Xb30)"] = [
        float(np.mean(B[null][:, i] <= dB[i]) * 100) for i in range(2)]
    out["percentile of the measured beta in the static-truth distribution (Xb15, Xb30)"] = [
        float(np.mean(B[alt][:, i] <= dB[i]) * 100) for i in range(2)]
    z_null = [(dB[i] - B[null][:, i].mean()) / B[null][:, i].std(ddof=1) for i in range(2)]
    z_alt = [(dB[i] - B[alt][:, i].mean()) / B[alt][:, i].std(ddof=1) for i in range(2)]
    out["z of the measured beta under the white / static truth (Xb15, Xb30)"] = [[float(x) for x in z_null],
                                                                                  [float(x) for x in z_alt]]
    mw = np.array(out["white truth, noise x1.0: J_nn mean per window"])
    control = np.array(out["white truth, noise x0.0: J_nn mean per window"])
    out["acceptance"] = {
        "noise-free control recovers J_nn in every window (within 2%)":
            bool(np.all(np.abs(control - out["truth J_nn (rad/s)"]) < 0.02 * out["truth J_nn (rad/s)"])),
        "a time-independent truth reproduces the late-window collapse (J[5,10] < 0.5 J[0,1.5])": bool(mw[-1] < 0.5 * mw[0]),
        "and the cumulative-window drift (J[0,5] < 0.95 J[0,1.5])": bool(mw[2] < 0.95 * mw[0]),
        "the beta test separates the truths (|mean difference| > 2 sd, Xb30)": bool(
            abs(B[alt][:, 1].mean() - B[null][:, 1].mean()) > 2 * B[null][:, 1].std(ddof=1)),
    }
    return out, J, B, key


def plot(J, B, key, out):
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    x = np.arange(len(WINDOWS))
    for case, c, mk, lab, off in ((("white", 0.0), INK, "s", "white truth, noise-free", -0.15),
                                  (("white", 0.25), ORDINAL[1], "o", "white truth, noise x 1/4", -0.05),
                                  (("white", 1.0), CAT[1], "o", "white truth, measured noise", 0.05),
                                  (("static", 1.0), CAT[2], "^", "static truth, measured noise", 0.15)):
        sel = np.array([k == case for k in key])
        err = J[sel].std(axis=0, ddof=1) if sel.sum() > 1 else None
        ax.errorbar(x + off, J[sel].mean(axis=0), yerr=err, fmt=mk, color=c, ms=5, capsize=2, label=lab)
    ax.plot(x, out["measured: J_nn per window"], "D", mfc="white", mec=CAT[0], ms=6, label="BEQS data")
    ax.axhline(out["truth J_nn (rad/s)"], color=MUTED, ls=":", label="truth")
    ax.set_xticks(x, [f"[{a:g}, {b:g}]" for a, b in WINDOWS])
    ax.set_xlabel("learning window (ms)")
    ax.set_ylabel("learned nearest-neighbour J (rad/s)")
    ax.set_title("(a) window dependence of J: synthetic truths vs data")
    ax.legend(fontsize=7)
    ax = axes[1]
    for i, (v, xpos) in enumerate((("Xb15", 0), ("Xb30", 1))):
        for case, c, off in ((("white", 1.0), CAT[0], -0.12), (("static", 1.0), CAT[1], 0.12)):
            sel = np.array([k == case for k in key])
            ax.plot(np.full(sel.sum(), xpos + off), B[sel][:, i], "o", color=c, ms=3.5, alpha=0.6,
                    label=(f"{case[0]} truth" if i == 0 else None))
        ax.plot([xpos - 0.3, xpos + 0.3], [out["measured: beta (Xb15, Xb30)"][i]] * 2, color=INK, lw=2,
                label="BEQS data" if i == 0 else None)
    ax.axhline(0.0, color=MUTED, lw=0.8)
    ax.set_xticks([0, 1], ["window 1.5 ms", "window 3 ms"])
    ax.set_ylabel("fitted slope $\\beta$ of the collective rate (rad/s per ms)")
    ax.set_title("(b) memory test: white vs quasi-static truth")
    ax.legend(fontsize=7)
    fig.suptitle("E11b: simulation-based calibration of the integral method (data: Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e11b_calibration")


if __name__ == "__main__":
    if "--replot" not in sys.argv:
        jobs, fits, data = run()
        np.savez_compressed(RESULTS / "e11b_calibration.npz", jobs=np.array(jobs, dtype=object),
                            full=np.array([f[0] for f in fits]), mem=np.array([f[1] for f in fits]),
                            data_full=data[0], data_mem=data[1])
    z = np.load(RESULTS / "e11b_calibration.npz", allow_pickle=True)
    jobs = [tuple(j) for j in z["jobs"]]
    fits = list(zip(z["full"], z["mem"]))
    out, J, B, key = analyse(jobs, fits, (z["data_full"], z["data_mem"]))
    save_json(RESULTS / "e11b_calibration.json", out)
    print(plot(J, B, key, out))
    for k, v in out.items():
        print(f"{k}: {v}")
