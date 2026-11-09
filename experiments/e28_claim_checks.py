"""E28: three checks of earlier claims raised by the audit Q30 (preregistered)

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0), analysed
here from the public raw shots.
Synthetic worlds (the World of E27): all share the setting average of the E11f control truth (Y15 couplings and
currents, z fields, quasi-static collective noise beta_s = 38.2 rad/s per ms; no transverse fields, flips or
emission) and are sampled like the data (200 settings x 200 shots, both measured initial states):
  shot-only      every shot independent;
  data-matched   a share sigma_d^2 / 2 of beta_s is a global detuning that is constant within a setting and drifts
                 across the settings (AR(1), lag-1 correlation 0.8), sigma_d = E27's estimate for the data; the rest
                 acts within every setting;
  full drift     all of beta_s drifts (sigma_d = 276 rad/s, the E26 world).
8 realisations per world and check; the oracle defect of the control truth (the common setting average).
C1. Is E9's certified collective rate conservative under the drift? E9's structured fit (E6a Hamiltonian ansatz with
    Gamma = gamma_c 11^T + diag(g), g >= 0) and E9's bound: the smaller of the percentile and the basic bootstrap
    95% lower bound over 100 setting bootstraps. Target: the same fit to the exact setting-averaged expectations,
    i.e. the value the estimator converges to (common to the three worlds). Rule C1, per world: the bound lies below
    the target in at least 7 of the 8 realisations (probability 0.94 for an exact 95% bound).
C2. Does the learning window identify the U(1)-breaking Hamiltonian terms? A truth with the control's Hamiltonian plus
    K = (XX - YY)/2 and L = (XY + YX)/2 on all 45 pairs and transverse fields Bx, By on all 10 ions, drawn once from
    N(0, (2 s_g)^2), s_g the median bootstrap sd of the group in E6a; its quasi-static collective noise is a global
    detuning integrated by a 20-point Gauss-Hermite rule in the full space (as the field truth of E11f); shot-only
    sampling; fitted with the E6a ansatz (least squares, oracle defect of this truth). Per group (K, L, B = Bx and By):
    the slope of the estimates on the truth through the origin, pooled over the realisations, and the residual sd over
    s_g; per realisation the number of the 110 terms with |estimate| > 3 times its E6a bootstrap sd. Rule C2: a group
    is identified if its slope lies in [0.8, 1.2] and its residual sd is at most 1.5 s_g.
C3. How much of the window dependence of J is estimator bias? The E6a ansatz fitted by least squares on the windows
    [0, 1.5], [0, 3] and [0, 5] ms (as E14) in the three worlds with the control truth's defect, and the data fitted
    the same way with the same defect. Reported: J_nn per window and its relative change against [0, 1.5] ms. Rule C3:
    the data's window dependence is explained by the estimator if the data's relative change at [0, 5] ms lies within
    the data-matched world's mean +- 2 sd over realisations; the share explained, (mean change in the data-matched
    world) / (change in the data), is reported for [0, 3] and [0, 5] ms either way.
Status: C1 and C2 use no data beyond the learning window. C3 fits the data up to 5 ms, which E14 had already done
with another defect.
Run: `--run` then `--analyse`. Workers: E28_WORKERS (default 6).
"""

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from common import CAT, INK, MUTED, RESULTS, save_figure, save_json, style  # noqa: E402
from e6_ion_learning import (N, PAIRS, ansatz, derived, observables, per_setting_tables, product_state,  # noqa: E402
                             series_from)
from e6c_validation import load  # noqa: E402
from e9_symmetry_data import e6a_system, extended_ansatz, structured_lower  # noqa: E402
from e11f_nonmarkov_truths import Propagator, hamiltonian  # noqa: E402
from e14_shot_calibration import jnn as jnn_e14  # noqa: E402
from e14_shot_calibration import truth_theta  # noqa: E402
from e24_eiv_correction import N_WIN, T_LEARN, strings_list  # noqa: E402
from e27_marginal_band import BETA_TRUTH, K0, SHOTS, SIGMA_REF, control_world, data_excess, drift_sigma  # noqa: E402
from qerr import spins  # noqa: E402
from qerr.learning import fit, fit_bounded, integral_system, refine, structured_map, trajectory_defect  # noqa: E402
from qerr.pauli import expectation_tables, expectations  # noqa: E402
from qerr.shots import pauli_tensor, sample_spins, setting_probabilities  # noqa: E402

N_REAL, N_BOOT, N_GH, WINDOWS = 8, 100, 20, (1.5, 3.0, 5.0)
WORKERS = int(os.environ.get("E28_WORKERS", "6"))
WORLDS = ("shot-only", "data-matched", "full drift")
PARTS = RESULTS / "e28_parts"
_W = {}


def world_sigma(world):
    return {"shot-only": 0.0, "data-matched": drift_sigma(data_excess()), "full drift": SIGMA_REF}[world]


def slug(world):
    return world.replace(" ", "_").replace("-", "_")


# ---------------- C1: E9's certified gamma_c ----------------
def struct_gamma(series, tw, d):
    ext, base = extended_ansatz(), ansatz()
    A, y, _ = integral_system(observables(), ext, series, tw, T_LEARN, defect=d)
    return float(fit_bounded(e6a_system(A, ext) @ structured_map(base), y, structured_lower(base))[K0])


def _init_c1(d):
    _W.update(d1=d)


def c1_job(args):
    world, r = args
    PARTS.mkdir(exist_ok=True)
    path = PARTS / f"c1_{slug(world)}_{r}.npz"
    if path.exists():
        z = np.load(path)
        return float(z["gamma"]), z["boot"]
    t0 = time.time()
    times, _, _, bases = load()
    tw, strings = times[:N_WIN], strings_list()
    rng = np.random.default_rng(281000 + 7907 * r + 100 * WORLDS.index(world))
    shots = control_world(world_sigma(world), tw).shots(rng, bases)
    tables = per_setting_tables(shots, bases, strings)
    g = struct_gamma(series_from(tables), tw, _W["d1"])
    n = len(bases)
    boot = np.array([struct_gamma(series_from(tables, np.bincount(rng.integers(0, n, n), minlength=n).astype(float)),
                                  tw, _W["d1"]) for _ in range(N_BOOT)])
    np.savez_compressed(path, gamma=g, boot=boot)
    print(f"C1 {world} {r}: {time.time() - t0:.0f} s", flush=True)
    return g, boot


# ---------------- C2: identification of the U(1)-breaking terms ----------------
def e6a_sd():
    """Per-parameter bootstrap sd of K (45), L (45), Bx and By (20) in E6a, and the median per group."""
    ans = ansatz()
    up = np.triu_indices(N, 1)
    vals = []
    for th in np.load(RESULTS / "e6_ion_learning.npz")["boot"]:
        d = derived(th, ans)
        vals.append(np.concatenate([d["K"][up], d["L"][up], d["Bx"], d["By"]]))
    sd = np.array(vals).std(axis=0, ddof=1)
    return sd, {"K": float(np.median(sd[:45])), "L": float(np.median(sd[45:90])), "B": float(np.median(sd[90:]))}


def c2_truth():
    """theta of the C2 truth (E6a indexing) and the injected values [K (45), L (45), Bx (10), By (10)]."""
    ans = ansatz()
    k = {p: m for m, p in enumerate(ans.h_terms)}
    _, s_g = e6a_sd()
    rng = np.random.default_rng(2828)
    K = rng.normal(0.0, 2.0 * s_g["K"], len(PAIRS))
    L = rng.normal(0.0, 2.0 * s_g["L"], len(PAIRS))
    B = rng.normal(0.0, 2.0 * s_g["B"], 2 * N)
    th = truth_theta().copy()
    th[K0:] = 0.0
    for m, (i, j) in enumerate(PAIRS):
        th[k[((i, "X"), (j, "X"))]] += K[m]
        th[k[((i, "Y"), (j, "Y"))]] -= K[m]
        th[k[((i, "X"), (j, "Y"))]] += L[m]
        th[k[((i, "Y"), (j, "X"))]] += L[m]
    for i in range(N):
        th[k[((i, "X"),)]] += B[i]
        th[k[((i, "Y"),)]] += B[N + i]
    return th, np.concatenate([K, L, B])


def c2_truth_job(s):
    """Fine-window trajectory and window outcome distributions of the C2 truth, initial state s."""
    PARTS.mkdir(exist_ok=True)
    path = PARTS / f"c2_truth_{s}.npz"
    if path.exists():
        return path
    t0 = time.time()
    times, s1, s2, bases = load()
    tw, strings = times[:N_WIN], strings_list()
    singles = [((i, a),) for i in range(N) for a in "XYZ"]
    rho0 = product_state(series_from(per_setting_tables((s1, s2), bases, singles)), s)
    th, _ = c2_truth()
    H = hamiltonian(th).toarray()
    zsum = spins.zvals(N).astype(float).sum(axis=1)
    sigma = np.sqrt(BETA_TRUTH / 2.0)  # detuning d sum_i Z_i with d ~ N(0, sigma^2): charge-q decay exp(-q^2 beta_s t^2)
    x, w = np.polynomial.hermite.hermgauss(N_GH)
    fine = refine(tw, 4)
    idx, ph = expectation_tables(strings, N)
    vals = np.zeros((len(fine), len(strings)))
    rho_bar = np.zeros((N_WIN, 2**N, 2**N), complex)
    at = np.searchsorted(fine, tw - 1e-12)
    for xk, wk in zip(x, w):
        P = Propagator(H + np.diag(np.sqrt(2.0) * sigma * xk * zsum), rho0)
        for m, t in enumerate(fine):
            rho = P(t)
            vals[m] += wk / np.sqrt(np.pi) * expectations(rho, idx, ph)
            if m in at:
                rho_bar[list(at).index(m)] += wk / np.sqrt(np.pi) * rho
    prob = np.array([[setting_probabilities(E, b) for b in bases] for E in (pauli_tensor(r, N) for r in rho_bar)],
                    np.float32)
    np.savez_compressed(path, fine=vals, prob=prob)
    print(f"C2 truth state {s + 1}: {time.time() - t0:.0f} s", flush=True)
    return path


def _init_c2(prob, d):
    _W.update(prob=prob, d2=d)


def c2_job(r):
    PARTS.mkdir(exist_ok=True)
    path = PARTS / f"c2_fit_{r}.npz"
    if path.exists():
        return np.load(path)["est"]
    times, _, _, bases = load()
    tw, strings = times[:N_WIN], strings_list()
    rng = np.random.default_rng(282000 + 7907 * r)
    shots = []
    for p in _W["prob"]:
        sp = np.empty((p.shape[0], p.shape[1], SHOTS, N), np.int8)
        for t in range(p.shape[0]):
            for k in range(p.shape[1]):
                sp[t, k] = sample_spins(p[t, k], SHOTS, rng, N)
        shots.append(sp)
    ans = ansatz()
    A, y, _ = integral_system(observables(), ans, series_from(per_setting_tables(tuple(shots), bases, strings)), tw,
                              T_LEARN, defect=_W["d2"])
    d = derived(fit(A, y), ans)
    up = np.triu_indices(N, 1)
    est = np.concatenate([d["K"][up], d["L"][up], d["Bx"], d["By"]])
    np.savez_compressed(path, est=est)
    return est


# ---------------- C3: J against the window ----------------
def _init_c3(defects):
    _W.update(d3=defects)


def window_j(series, times):
    ans = ansatz()
    out = []
    for T in WINDOWS:
        A, y, _ = integral_system(observables(), ans, series, times, T, defect=_W["d3"][T])
        out.append(float(jnn_e14(fit(A, y), ans)))
    return np.array(out)


def c3_job(args):
    world, r = args
    PARTS.mkdir(exist_ok=True)
    path = PARTS / f"c3_{slug(world)}_{r}.npz"
    if path.exists():
        return np.load(path)["J"]
    t0 = time.time()
    times, _, _, bases = load()
    t11 = times[times <= max(WINDOWS) + 1e-9]
    rng = np.random.default_rng(283000 + 7907 * r + 100 * WORLDS.index(world))
    shots = control_world(world_sigma(world), t11).shots(rng, bases)
    J = window_j(series_from(per_setting_tables(shots, bases, strings_list())), t11)
    np.savez_compressed(path, J=J)
    print(f"C3 {world} {r}: {time.time() - t0:.0f} s", flush=True)
    return J


# ---------------- run ----------------
def run():
    times, s1, s2, bases = load()
    tw, strings = times[:N_WIN], strings_list()
    t11 = times[times <= max(WINDOWS) + 1e-9]
    w0 = control_world(0.0, t11)
    fine11 = refine(t11, 4)
    traj11 = w0.trajectory(strings, fine11)
    d3 = {T: trajectory_defect(fine11, traj11, strings, t11, T) for T in WINDOWS}
    d1 = d3[T_LEARN]  # the window defect of the control truth (exact quadrature reference on [0, 1.5] ms)
    exact = {(s, p): traj11[s][np.searchsorted(fine11, tw - 1e-12), j] for s in (0, 1) for j, p in enumerate(strings)}
    target = struct_gamma(exact, tw, d1)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(c2_truth_job, (0, 1)))
    print(f"C2 truth: {time.time() - t0:.0f} s", flush=True)
    c2t = [np.load(PARTS / f"c2_truth_{s}.npz") for s in (0, 1)]
    d2 = trajectory_defect(refine(tw, 4), [z["fine"] for z in c2t], strings, tw, T_LEARN)
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_c2, initargs=([z["prob"] for z in c2t], d2)) as pool:
        c2 = np.array(list(pool.map(c2_job, range(N_REAL))))
    print(f"C2 fits: {time.time() - t0:.0f} s", flush=True)
    jobs3 = [(w, r) for w in WORLDS for r in range(N_REAL)]
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_c3, initargs=(d3,)) as pool:
        c3 = np.array(list(pool.map(c3_job, jobs3))).reshape(len(WORLDS), N_REAL, len(WINDOWS))
    _W.update(d3=d3)
    j_data = window_j(series_from(per_setting_tables((s1[:len(t11)], s2[:len(t11)]), bases, strings)), t11)
    print(f"C3: {time.time() - t0:.0f} s", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_c1, initargs=(d1,)) as pool:
        c1 = list(pool.map(c1_job, [(w, r) for w in WORLDS for r in range(N_REAL)]))
    print(f"C1: {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(RESULTS / "e28_claim_checks.npz", c1_gamma=np.array([g for g, _ in c1]).reshape(len(WORLDS), N_REAL),
                        c1_boot=np.array([b for _, b in c1]).reshape(len(WORLDS), N_REAL, N_BOOT), c1_target=target,
                        c2_est=c2, c2_truth=c2_truth()[1], c3_J=c3, c3_data=j_data)


# ---------------- analysis ----------------
def analyse():
    import json
    z = np.load(RESULTS / "e28_claim_checks.npz")
    out = {"worlds: sigma_d (rad/s)": {w: world_sigma(w) * 1e3 for w in WORLDS}}
    # C1
    tgt = float(z["c1_target"])
    c1 = {"target: structured gamma_c on the exact setting average (rad/s)": tgt * 1e3}
    lbs = {}
    for i, w in enumerate(WORLDS):
        g, b = z["c1_gamma"][i], z["c1_boot"][i]
        lo = np.minimum(np.percentile(b, 5, axis=1), 2 * g - np.percentile(b, 95, axis=1))
        lbs[w] = lo
        hits = int(np.sum(lo <= tgt))
        c1[w] = {"estimate mean, sd (rad/s)": [float(g.mean()) * 1e3, float(g.std(ddof=1)) * 1e3],
                 "bootstrap mean minus estimate, mean (rad/s)": float((b.mean(axis=1) - g).mean()) * 1e3,
                 "lower bound per realisation (rad/s)": (lo * 1e3).tolist(),
                 "bound below the target": f"{hits} of {N_REAL}", "rule C1 passes": bool(hits >= 7)}
    e9 = json.loads((RESULTS / "e9_symmetry_data.json").read_text(encoding="utf-8"))
    c1["E9 on the data (for reference)"] = {k: v for k, v in e9.items() if k.startswith("(b) structured gamma_c")}
    out["C1: E9's certified gamma_c"] = c1
    # C2
    sd, s_g = e6a_sd()
    est, tru = z["c2_est"], z["c2_truth"]
    groups = {"K": slice(0, 45), "L": slice(45, 90), "B": slice(90, 110)}
    c2 = {"E6a median bootstrap sd s_g (rad/s)": {g: v * 1e3 for g, v in s_g.items()},
          "injected rms (rad/s)": {g: float(np.sqrt(np.mean(tru[sl] ** 2))) * 1e3 for g, sl in groups.items()}}
    for g, sl in groups.items():
        e, t = est[:, sl], np.broadcast_to(tru[sl], est[:, sl].shape)
        slope = float(np.sum(e * t) / np.sum(t * t))
        resid = float(np.std(e - slope * t, ddof=1))
        c2[g] = {"slope": slope, "residual sd / s_g": resid / s_g[g],
                 "identified (slope in [0.8, 1.2], residual <= 1.5 s_g)": bool(0.8 <= slope <= 1.2 and resid <= 1.5 * s_g[g])}
    zc = np.abs(est) / sd[None, :]
    c2["terms with |estimate| > 3 E6a sd, per realisation (of 110)"] = (zc > 3).sum(axis=1).tolist()
    c2["true terms with |value| > 3 E6a sd (of 110)"] = int((np.abs(tru) / sd > 3).sum())
    out["C2: identification of the U(1)-breaking terms"] = c2
    # C3
    J, jd = z["c3_J"], z["c3_data"]
    rel = J / J[:, :, :1] - 1.0
    rel_d = jd / jd[0] - 1.0
    c3 = {"windows (ms)": [f"[0, {T:g}]" for T in WINDOWS],
          "data: J_nn (rad/s), relative change": [(jd * 1e3).tolist(), rel_d.tolist()]}
    for i, w in enumerate(WORLDS):
        c3[w] = {"J_nn mean (rad/s)": (J[i].mean(axis=0) * 1e3).tolist(),
                 "relative change: mean, sd": [rel[i].mean(axis=0).tolist(), rel[i].std(axis=0, ddof=1).tolist()]}
    i = WORLDS.index("data-matched")
    m, s = rel[i, :, -1].mean(), rel[i, :, -1].std(ddof=1)
    c3["share of the data's change explained by the data-matched world: [0, 3], [0, 5] ms"] = [
        float(rel[i, :, 1].mean() / rel_d[1]), float(m / rel_d[2])]
    c3["share explained by the shot-only world: [0, 3], [0, 5] ms"] = [
        float(rel[0, :, 1].mean() / rel_d[1]), float(rel[0, :, 2].mean() / rel_d[2])]
    c3["rule C3: data's change at [0, 5] ms within the data-matched mean +- 2 sd"] = bool(abs(rel_d[2] - m) <= 2 * s)
    out["C3: J against the window"] = c3
    return out, lbs, tgt


def plot(out, lbs, tgt):
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.0))
    ax = axes[0]
    for i, w in enumerate(WORLDS):
        ax.plot(np.full(N_REAL, i) + np.linspace(-0.15, 0.15, N_REAL), (lbs[w] - tgt) * 1e3, "o", color=CAT[i], ms=4)
    ax.axhline(0.0, color=INK, lw=0.8)
    ax.set_xticks(range(len(WORLDS)), WORLDS)
    ax.set_ylabel("E9 lower bound minus target (rad/s)")
    ax.set_title("(a) C1: certified gamma_c, below 0 = conservative", fontsize=9)
    ax = axes[1]
    c2 = out["C2: identification of the U(1)-breaking terms"]
    z = np.load(RESULTS / "e28_claim_checks.npz")
    est, tru = z["c2_est"], z["c2_truth"]
    for u, (g, sl) in enumerate({"K": slice(0, 45), "L": slice(45, 90), "B": slice(90, 110)}.items()):
        ax.plot(np.broadcast_to(tru[sl], est[:, sl].shape).ravel() * 1e3, est[:, sl].ravel() * 1e3, ".", ms=2.5,
                color=CAT[u], alpha=0.6, label=f"{g}: slope {c2[g]['slope']:.2f}")
    lim = float(np.abs(tru).max() * 1e3 * 1.2)
    ax.plot([-lim, lim], [-lim, lim], ":", color=MUTED, lw=0.8)
    ax.set_xlabel("truth (rad/s)")
    ax.set_ylabel("estimate, 8 realisations (rad/s)")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("(b) C2: U(1)-breaking terms in the window", fontsize=9)
    ax = axes[2]
    c3 = out["C3: J against the window"]
    x = np.array(WINDOWS)
    for i, w in enumerate(WORLDS):
        m, s = (np.array(v) for v in c3[w]["relative change: mean, sd"])
        ax.errorbar(x + 0.06 * (i - 1), 100 * m, yerr=200 * s, fmt="-o", color=CAT[i], ms=4, capsize=2, lw=1.2,
                    label=f"{w} (+- 2 sd)")
    ax.plot(x, 100 * np.array(c3["data: J_nn (rad/s), relative change"][1]), "-s", color=INK, ms=5, lw=1.4, label="data")
    ax.axhline(0.0, color=MUTED, lw=0.6, ls=":")
    ax.set_xticks(x, c3["windows (ms)"])
    ax.set_ylabel("change of J_nn against [0, 1.5] ms (%)")
    ax.legend(fontsize=6.5, loc="lower left")
    ax.set_title("(c) C3: J against the learning window", fontsize=9)
    fig.suptitle("E28: checks of earlier claims (data: Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e28_claim_checks")


if __name__ == "__main__":
    if "--run" in sys.argv:
        run()
    if "--analyse" in sys.argv:
        out, lbs, tgt = analyse()
        save_json(RESULTS / "e28_claim_checks.json", out)
        print(plot(out, lbs, tgt))
        for key, val in out.items():
            print(key, val, flush=True)
