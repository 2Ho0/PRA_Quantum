"""E23k: where does the charge-2 imbalance K come from?

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0), analysed
here from the public raw shots. K is the site mean of S_XX - S_YY of the E11c noise block, learned on t <= 1.5 ms.
E22 (C) found the excess in state 2 when each initial state is fitted alone.
(A) K for both states together and for each state alone, with all time points and with t = 0.5 ms removed from the
    data (series, spline and defect), fitted without and with the collective slope (memory); the E11f control truth
    (12 shot-level realisations, oracle defect) calibrates every configuration.
(B) The Xb15uf generator learned on each state alone (bootstrap over the settings): are the two states described by one
    generator?
(C) A run-to-run detuning as a truth: the E11f control truth with a global detuning delta sum_i Z_i added for state 1
    only (exact sector propagation, the same collective noise), delta = 124 rad/s (the difference of the mean z field
    between the single-state fits of (B)) and 62 rad/s; 12 realisations each, K as in (A).
(D) Post hoc (`--part-d`): the joint fit with run-specific nuisance parameters, i.e. a global detuning and a collective
    dephasing rate of its own for state 2, for the data and for the 12 control realisations, with all time points and
    with t = 0.5 ms removed.
Status: (A) and (B) are diagnostics whose central values were first seen in scratch runs. Fixed before the (C) run:
a run-to-run detuning explains the imbalance if, with t = 0.5 ms removed, the data's joint K and both single-state K
lie inside the 5-95% ranges of the delta = 124 rad/s truth. (D) was added after (C).
"""

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2" if "--truths" in sys.argv else "1")

import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import lsq_linear  # noqa: E402

from common import CAT, INK, MUTED, RESULTS, save_figure, save_json, style  # noqa: E402
from e6_ion_learning import (N, PAIRS, RAD_S, ansatz, observables, per_setting_tables, product_state,  # noqa: E402
                             series_from)
from e6c_validation import load  # noqa: E402
from e9_symmetry_data import extended_ansatz  # noqa: E402
from e11_model_selection import N_SUB  # noqa: E402
from e11c_noise_ansatz import constrained_map, summary  # noqa: E402
from e11f_nonmarkov_truths import Propagator, grids, hamiltonian, load_truth, sectors  # noqa: E402
from e14_shot_calibration import chi_exponent, synthetic_series, truth_theta  # noqa: E402
from e20_flip_models import S_51, structure  # noqa: E402
from qerr import spins  # noqa: E402
from qerr.learning import integral_system, needed_strings, refine, trajectory_defect  # noqa: E402
from qerr.pauli import expectation_tables, expectations  # noqa: E402
from qerr.shots import pauli_tensor, setting_probabilities  # noqa: E402

T_LEARN, DROP = 1.5, 0.5
N_REAL, N_BOOT, WORKERS = 12, 40, 6
DELTAS = (124.0, 62.0)  # rad/s, global detuning of state 1 in the truths of (C)
CONFIGS = [(drop, mem) for drop in (False, True) for mem in (False, True)]
STATE_SETS = ((0, 1), (0,), (1,))
PARTS = RESULTS / "e23k_parts"
_W = {}


def strings_list():
    obs = observables()
    return sorted(set(obs) | set(needed_strings(obs, ansatz())))


def keep_mask(times, drop):
    return ~np.isclose(times, DROP) if drop else np.ones(len(times), bool)


def k_values(series, times, fine_traj, fine):
    """K (rad/s) for every configuration (drop t = 0.5?, memory?) and state set; shape (4, 3)."""
    obs, ext, strings = observables(), extended_ansatz(), strings_list()
    Tm = constrained_map(ext)
    cvec = np.zeros(ext.n_params)
    cvec[len(ext.h_terms):len(ext.h_terms) + len(ext.deph_pairs)] = 1.0
    out = np.zeros((len(CONFIGS), len(STATE_SETS)))
    for c, (drop, mem) in enumerate(CONFIGS):
        keep = keep_mask(times, drop)
        tk = times[keep]
        sk = {key: v[keep] for key, v in series.items()}
        d = trajectory_defect(fine, fine_traj, strings, tk, T_LEARN)
        dt = trajectory_defect(fine, fine_traj, strings, tk, T_LEARN, weight=lambda s: s)
        for j, st in enumerate(STATE_SETS):
            A, y, _ = integral_system(obs, ext, sk, tk, T_LEARN, states=st, defect=d)
            M = A @ Tm
            if mem:
                At, _, _ = integral_system(obs, ext, sk, tk, T_LEARN, states=st, defect=dt, weight=lambda s: s)
                M = np.column_stack([M, At @ cvec])
            phi = lsq_linear(M, y, method="bvls", tol=1e-12).x
            out[c, j] = float(summary(Tm @ phi[:Tm.shape[1]], ext)[0]["SXX-SYY"].mean())
    return out


# ---------------- (A) data and control ----------------
def data_k():
    times, s1, s2, bases = load()
    strings = strings_list()
    series = series_from(per_setting_tables((s1, s2), bases, strings))
    z = np.load(RESULTS / "e11_stage1.npz")
    return k_values(series, times, [z["traj0"], z["traj1"]], refine(times, N_SUB))


def _init_truth(prob, fine_traj):
    _W.update(prob=prob, fine_traj=fine_traj)


def realisation(seed):
    times, s1, s2, bases = load()
    series = synthetic_series(_W["prob"], bases, strings_list(), np.random.default_rng(seed))
    fine, _ = grids(times)
    return k_values(series, times, _W["fine_traj"], fine)


def sbc(prob, fine_traj, offset):
    seeds = [104729 * i + 77 + offset for i in range(N_REAL)]
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_truth, initargs=(prob, fine_traj)) as pool:
        return np.array(list(pool.map(realisation, seeds)))


# ---------------- (B) single-state generators ----------------
def _init_boot(tables, times, d, dt):
    _W.update(tables=tables, times=times, d=d, dt=dt)


def generator_summary(w):
    """(J at d = 1, 2, 3, mean and spread of h_z, h_x, h_y, gamma_c, beta) in rad/s for state 1, state 2, both."""
    times, obs, ext = _W["times"], observables(), extended_ansatz()
    sr = series_from(_W["tables"], w)
    T, fix, flip, coll = structure(ext, uniform=True)
    k = {p: m for m, p in enumerate(ext.h_terms)}
    dist = np.array([j - i for i, j in PAIRS])
    out = []
    for st in ((0,), (1,), (0, 1)):
        A, y, _ = integral_system(obs, ext, sr, times, T_LEARN, states=st, defect=_W["d"])
        At, _, _ = integral_system(obs, ext, sr, times, T_LEARN, states=st, defect=_W["dt"], weight=lambda s: s)
        M = np.column_stack([A @ T, At @ coll])
        phi = lsq_linear(M, y - A @ fix - S_51 * (A @ flip), method="bvls", tol=1e-12).x
        th = T @ phi[:-1] + fix
        J = np.array([(th[k[((i, "X"), (j, "X"))]] + th[k[((i, "Y"), (j, "Y"))]]) / 2 for i, j in PAIRS])
        hz = np.array([th[k[((i, "Z"),)]] for i in range(N)])
        out.append(np.r_[[J[dist == dd].mean() for dd in (1, 2, 3)], hz.mean(), np.ptp(hz),
                         th[k[((0, "X"),)]], th[k[((0, "Y"),)]], th[len(ext.h_terms)], phi[-1]] * RAD_S)
    return np.array(out)


def single_state_generators():
    times, s1, s2, bases = load()
    strings = strings_list()
    tables = per_setting_tables((s1, s2), bases, strings)
    z = np.load(RESULTS / "e11_stage1.npz")
    fine = refine(times, N_SUB)
    d = trajectory_defect(fine, [z["traj0"], z["traj1"]], strings, times, T_LEARN)
    dt = trajectory_defect(fine, [z["traj0"], z["traj1"]], strings, times, T_LEARN, weight=lambda s: s)
    rng = np.random.default_rng(77)
    n_set = bases.shape[0]
    ws = [None] + [np.bincount(rng.integers(0, n_set, n_set), minlength=n_set).astype(float) for _ in range(N_BOOT)]
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_boot, initargs=(tables, times, d, dt)) as pool:
        return np.array(list(pool.map(generator_summary, ws)))


# ---------------- (C) run-to-run detuning truths ----------------
def truth_job(args):
    """State-1 part of the control truth with a global detuning delta (rad/s); returns the part file."""
    delta = args
    PARTS.mkdir(exist_ok=True)
    path = PARTS / f"detuned_{delta:.0f}_0.npz"
    if path.exists():
        return path
    t0 = time.time()
    times, s1, s2, bases = load()
    strings = strings_list()
    singles = [((i, a),) for i in range(N) for a in "XYZ"]
    rho0 = product_state(series_from(per_setting_tables((s1, s2), bases, singles)), 0)
    theta = truth_theta()
    gamma_c = theta[len(ansatz().h_terms)]
    Ztot = np.diag(spins.zvals(N).sum(axis=1).astype(float))
    P = Propagator(hamiltonian(theta).toarray() + (delta / RAD_S) * Ztot, rho0, sectors())
    fine, allt = grids(times)
    idx, ph = expectation_tables(strings, N)
    m = spins.total_sz(N)
    q2 = (m[:, None] - m[None, :]) ** 2
    vals, rho_bar = [], []
    coarse = np.searchsorted(allt, times - 1e-12)
    for k, t in enumerate(allt):
        rf = P(t) * np.exp(-q2 * chi_exponent(1.0, gamma_c, t))
        vals.append(expectations(rf, idx, ph))
        if k in coarse:
            rho_bar.append(rf)
    prob = np.array([[setting_probabilities(E, b) for b in bases] for E in (pauli_tensor(r, N) for r in rho_bar)],
                    np.float32)
    vals = np.array(vals)
    np.savez_compressed(path, fine=vals[np.searchsorted(allt, fine - 1e-12)], coarse=vals[coarse], prob=prob)
    print(f"{path.stem}: {time.time() - t0:.0f} s", flush=True)
    return path


def detuned_truth(delta):
    """prob and fine trajectories of the truth: state 1 detuned by delta, state 2 as the control."""
    ctrl = load_truth("control")
    part = np.load(PARTS / f"detuned_{delta:.0f}_0.npz")
    return [part["prob"], ctrl["prob"][1]], [part["fine"], ctrl["fine"][1]]


def run():
    out = {"data": data_k()}
    ctrl = load_truth("control")
    out["control"] = sbc(ctrl["prob"], ctrl["fine"], 0)
    for delta in DELTAS:
        prob, fine_traj = detuned_truth(delta)
        out[f"detuned {delta:.0f}"] = sbc(prob, fine_traj, 7)
    out["generators"] = single_state_generators()
    return out


# ---------------- (D) run-specific nuisance parameters ----------------
def k_run_parameters(series, times, fine, fine_traj):
    """Joint K without and with a global detuning and a collective rate of state 2's own; all points, t = 0.5 removed."""
    obs, ext, strings = observables(), extended_ansatz(), strings_list()
    Tm = constrained_map(ext)
    k = {p: m for m, p in enumerate(ext.h_terms)}
    hz = np.zeros(ext.n_params)
    for i in range(N):
        hz[k[((i, "Z"),)]] = 1.0
    gc = np.zeros(ext.n_params)
    gc[len(ext.h_terms):len(ext.h_terms) + len(ext.deph_pairs)] = 1.0
    out = []
    for drop in (False, True):
        keep = keep_mask(times, drop)
        tk = times[keep]
        sk = {key: v[keep] for key, v in series.items()}
        d = trajectory_defect(fine, fine_traj, strings, tk, T_LEARN)
        blocks = [integral_system(obs, ext, sk, tk, T_LEARN, states=(s,), defect=d) for s in (0, 1)]
        y = np.concatenate([b[1] for b in blocks])
        base = np.vstack([A @ Tm for A, _, _ in blocks])
        A1 = blocks[1][0]
        run = np.vstack([np.zeros((blocks[0][0].shape[0], 2)), np.column_stack([A1 @ hz, A1 @ gc])])
        for M in (base, np.column_stack([base, run])):
            phi = lsq_linear(M, y, method="bvls", tol=1e-12).x
            out.append(float(summary(Tm @ phi[:Tm.shape[1]], ext)[0]["SXX-SYY"].mean()))
    return out  # [all: joint, joint + run; dropped: joint, joint + run]


def realisation_d(seed):
    times, s1, s2, bases = load()
    series = synthetic_series(_W["prob"], bases, strings_list(), np.random.default_rng(seed))
    return k_run_parameters(series, times, grids(times)[0], _W["fine_traj"])


def part_d():
    times, s1, s2, bases = load()
    series = series_from(per_setting_tables((s1, s2), bases, strings_list()))
    z = np.load(RESULTS / "e11_stage1.npz")
    data = k_run_parameters(series, times, refine(times, N_SUB), [z["traj0"], z["traj1"]])
    ctrl = load_truth("control")
    seeds = [104729 * i + 77 for i in range(N_REAL)]
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_truth, initargs=(ctrl["prob"], ctrl["fine"])) as pool:
        control = np.array(list(pool.map(realisation_d, seeds)))
    labels = ["all points, joint", "all points, joint + run parameters", "t = 0.5 removed, joint",
              "t = 0.5 removed, joint + run parameters"]
    return {lab: {"data": data[j], "control 5/50/95": np.percentile(control[:, j], [5, 50, 95]).tolist()}
            for j, lab in enumerate(labels)}


# ---------------- analysis ----------------
def analyse(res):
    labels = ["all points, no memory", "all points, memory", "t = 0.5 removed, no memory", "t = 0.5 removed, memory"]
    sets = ["both states", "state 1", "state 2"]
    out = {"(A) K (rad/s): data and 5/50/95 percentiles of the truths": {}}
    for c, lab in enumerate(labels):
        row = {}
        for j, sname in enumerate(sets):
            row[sname] = {"data": float(res["data"][c, j]),
                          **{name: np.percentile(res[name][:, c, j], [5, 50, 95]).tolist()
                             for name in ["control"] + [f"detuned {d:.0f}" for d in DELTAS]}}
        out["(A) K (rad/s): data and 5/50/95 percentiles of the truths"][lab] = row
    g = res["generators"]
    names = ["J(d=1)", "J(d=2)", "J(d=3)", "mean h_z", "spread h_z", "h_x", "h_y", "gamma_c", "beta"]
    diff = g[:, 0] - g[:, 1]
    out["(B) generator learned on one state (rad/s): value, bootstrap sd; difference state 1 - state 2 and its z"] = {
        n: {"state 1": [float(g[0, 0, m]), float(g[1:, 0, m].std(ddof=1))],
            "state 2": [float(g[0, 1, m]), float(g[1:, 1, m].std(ddof=1))], "both": float(g[0, 2, m]),
            "difference": [float(diff[0, m]), float(diff[1:, m].std(ddof=1))],
            "z": float(diff[0, m] / diff[1:, m].std(ddof=1))} for m, n in enumerate(names)}
    c = labels.index("t = 0.5 removed, no memory")
    lo, hi = np.percentile(res["detuned 124"][:, c], [5, 95], axis=0)
    inside = [bool(lo[j] <= res["data"][c, j] <= hi[j]) for j in range(3)]
    out["(C) verdict (t = 0.5 removed, no memory, delta = 124 rad/s)"] = {
        **{f"data inside the 5-95% range, {s}": v for s, v in zip(sets, inside)},
        "a run-to-run detuning explains the imbalance": bool(all(inside))}
    return out


def plot(res):
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
    for ax, c, title in ((axes[0], 0, "(a) all time points"), (axes[1], 2, "(b) t = 0.5 ms removed")):
        for j, sname in enumerate(("both", "state 1", "state 2")):
            for k, (name, col) in enumerate((("control", MUTED), ("detuned 124", CAT[1]), ("detuned 62", CAT[2]))):
                v = res[name][:, c, j]
                ax.plot(v, np.full(v.size, j + 0.22 * (k - 1)), "o", color=col, ms=3, alpha=0.7,
                        label=name if j == 0 else None)
            ax.plot(res["data"][c, j], j, "D", color=INK, ms=7, label="data" if j == 0 else None)
        ax.set_yticks(range(3), ["both states", "state 1 alone", "state 2 alone"])
        ax.invert_yaxis()
        ax.set_xlabel("site-mean $S_{XX} - S_{YY}$ (rad/s)")
        ax.set_title(title + ", fit without memory", fontsize=9)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("E23k: sources of the charge-2 imbalance (data: Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e23k_imbalance_sources")


if __name__ == "__main__":
    path = RESULTS / "e23k_imbalance_sources.npz"
    if "--truths" in sys.argv:
        with ProcessPoolExecutor(max_workers=2) as pool:
            print(list(pool.map(truth_job, DELTAS)))
    elif "--part-d" in sys.argv:
        d_out = part_d()
        save_json(RESULTS / "e23k_run_parameters.json", d_out)
        for key, v in d_out.items():
            print(f"{key}: {v}")
    else:
        if "--replot" not in sys.argv:
            res = run()
            np.savez_compressed(path, **{k.replace(" ", "_"): v for k, v in res.items()})
        z = np.load(path)
        res = {k.replace("_", " "): z[k] for k in z.keys()}
        out = analyse(res)
        save_json(RESULTS / "e23k_imbalance_sources.json", out)
        print(plot(res))
        for key, v in out.items():
            print(f"{key}: {v}")
