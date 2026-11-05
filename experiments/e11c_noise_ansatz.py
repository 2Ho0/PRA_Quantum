"""E11c: are the unphysical single-ion noise parameters of E9 an artefact of the over-parameterised fit?

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0).
E9 fitted the full single-ion Kossakowski block (8 parameters per ion) on top of the full E6a Hamiltonian and dephasing
matrix and found a negative total flip rate (site mean -50 rad/s) and a common charge +-2 imbalance S_XX - S_YY of
about -160 rad/s on every ion. Here the same noise block sits on top of the minimal physical model of E11 (X15:
U(1)-symmetric couplings, single-site fields, collective dephasing; the noise block replaces the emission), learned on
t <= 1.5 ms with the defect of the E11 reference trajectory (results/e11_stage1.npz). Bootstrap over the 200 settings.
If the anomalies vanish, they were compensating the spurious local dephasing and U(1)-breaking couplings of the
full fit, not a property of the device.
"""

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import lsq_linear

from common import RESULTS, save_json
from e6_ion_learning import N, PAIRS, RAD_S, ansatz, observables, per_setting_tables, series_from
from e6c_validation import load
from e9_symmetry_data import COMBOS, extended_ansatz
from e11_model_selection import N_SUB
from qerr.learning import NOISE_KEYS, flip_rates, integral_system, needed_strings, refine, site_kossakowski, trajectory_defect

N_BOOT, T_LEARN = 100, 1.5
_W = {}


def constrained_map(ext):
    """theta_ext = T phi: U(1)-symmetric couplings, all fields, collective Gamma, the full noise block."""
    k = {p: m for m, p in enumerate(ext.h_terms)}
    k0, nd = len(ext.h_terms), len(ext.deph_pairs)
    cols = []

    def col(entries):
        c = np.zeros(ext.n_params)
        for m, v in entries:
            c[m] = v
        cols.append(c)

    for i, j in PAIRS:
        xx, yy, xy, yx = (k[((i, a), (j, b))] for a, b in ("XX", "YY", "XY", "YX"))
        col([(xx, 1.0), (yy, 1.0)])
        col([(xy, 1.0), (yx, -1.0)])
    for i in range(N):
        for a in "XYZ":
            col([(k[((i, a),)], 1.0)])
    col([(k0 + m, 1.0) for m in range(nd)])
    for m in range(len(NOISE_KEYS) * N):
        col([(k0 + nd + m, 1.0)])
    return np.array(cols).T


def summary(theta, ext):
    kn = len(ext.h_terms) + len(ext.deph_pairs)
    block = theta[kn:].reshape(N, len(NOISE_KEYS))
    c = {key: block[:, u] for u, key in enumerate(NOISE_KEYS)}
    combos = {name: sum(sign * c[key] for key, sign in combo) * RAD_S for name, combo in COMBOS.items()}
    em, ab = flip_rates(site_kossakowski(theta, ext))
    return combos, em * RAD_S, ab * RAD_S, theta[len(ext.h_terms)] * RAD_S


def _init(tables, times, defect, defect_t):
    _W.update(tables=tables, times=times, defect=defect, defect_t=defect_t)


def replicate(w):
    """[theta without memory, theta with the growing collective rate gamma_c + beta t, beta]."""
    ext = extended_ansatz()
    series = series_from(_W["tables"], w)
    A, y, _ = integral_system(observables(), ext, series, _W["times"], T_LEARN, defect=_W["defect"])
    At, _, _ = integral_system(observables(), ext, series, _W["times"], T_LEARN, defect=_W["defect_t"],
                               weight=lambda s: s)
    T = constrained_map(ext)
    M = A @ T
    markov = T @ lsq_linear(M, y, method="bvls", tol=1e-12).x
    c = np.zeros(ext.n_params)
    c[len(ext.h_terms):len(ext.h_terms) + len(ext.deph_pairs)] = 1.0
    phi = lsq_linear(np.column_stack([M, At @ c]), y, method="bvls", tol=1e-12).x
    return np.r_[markov, T @ phi[:-1], phi[-1]]


def run():
    times, s1, s2, bases = load()
    ext, obs = extended_ansatz(), observables()
    strings = sorted(set(obs) | set(needed_strings(obs, ext)) | set(needed_strings(obs, ansatz())))
    tables = per_setting_tables((s1, s2), bases, strings)
    z = np.load(RESULTS / "e11_stage1.npz")
    traj_strings = needed_strings(obs, ansatz())
    assert set(needed_strings(obs, ext)) <= set(traj_strings), "the noise block needs strings outside the trajectory"
    fine = refine(times, N_SUB)
    defect = trajectory_defect(fine, [z["traj0"], z["traj1"]], traj_strings, times, T_LEARN)
    defect_t = trajectory_defect(fine, [z["traj0"], z["traj1"]], traj_strings, times, T_LEARN, weight=lambda s: s)
    _init(tables, times, defect, defect_t)
    central = replicate(None)
    rng = np.random.default_rng(2029)
    n_set = bases.shape[0]
    weights = [np.bincount(rng.integers(0, n_set, n_set), minlength=n_set).astype(float) for _ in range(N_BOOT)]
    with ProcessPoolExecutor(max_workers=4, initializer=_init, initargs=(tables, times, defect, defect_t)) as pool:
        boot = list(pool.map(replicate, weights))
    return central, np.array(boot)


def model_summary(central, boot, ext):
    cv, em, ab, gc = summary(central, ext)
    bs = [summary(th, ext) for th in boot]
    out = {"collective gamma_c (rad/s): estimate, bootstrap sd": [float(gc), float(np.std([b[3] for b in bs], ddof=1))]}
    for name in COMBOS:
        vals = np.array([b[0][name] for b in bs])
        sd = vals.std(axis=0, ddof=1)
        zs = cv[name] / np.where(sd > 0, sd, np.inf)
        out[name] = {"site mean (rad/s)": float(cv[name].mean()),
                     "bootstrap sd of the site mean": float(vals.mean(axis=1).std(ddof=1)),
                     "ions with |z| > 3": int(np.sum(np.abs(zs) > 3)), "max |z|": float(np.abs(zs).max())}
    S = em + ab
    Sb = np.array([b[1] + b[2] for b in bs])
    out["total flip rate (emission + absorption) per ion (rad/s)"] = [float(x) for x in S]
    out["total flip rate: site mean, bootstrap sd"] = [float(S.mean()), float(Sb.mean(axis=1).std(ddof=1))]
    out["ions whose XY block violates |S_XX - S_YY| <= S_XX + S_YY (not PSD)"] = int(np.sum(
        np.abs(cv["SXX-SYY"]) > cv["flip S=SXX+SYY"]))
    return out


def analyse(central, boot):
    ext = extended_ansatz()
    P = ext.n_params
    out = {"without memory": model_summary(central[:P], boot[:, :P], ext),
           "with gamma_c + beta t": model_summary(central[P:2 * P], boot[:, P:2 * P], ext)}
    be = boot[:, -1] * RAD_S
    out["beta (rad/s per ms): estimate, bootstrap mean, sd"] = [float(central[-1] * RAD_S), float(be.mean()),
                                                               float(be.std(ddof=1))]
    out["E9 (full ansatz) for comparison"] = {"flip S site mean": -49.9, "S_XX - S_YY per ion": "about -160",
                                               "combinations beyond 3 sigma": "10 of 60"}
    m = out["without memory"]
    charged = [n for n in COMBOS if not n.startswith(("flip", "imbalance"))]
    out["acceptance"] = {
        "total flip rate consistent with >= 0 (site mean > -2 sd)": bool(
            m["total flip rate: site mean, bootstrap sd"][0] > -2 * m["total flip rate: site mean, bootstrap sd"][1]),
        "no charged combination beyond 3 sigma on more than one ion": bool(all(m[n]["ions with |z| > 3"] <= 1
                                                                              for n in charged)),
    }
    return out


if __name__ == "__main__":
    if "--replot" not in sys.argv:
        central, boot = run()
        np.savez_compressed(RESULTS / "e11c_noise_ansatz.npz", central=central, boot=boot)
    zz = np.load(RESULTS / "e11c_noise_ansatz.npz")
    out = analyse(zz["central"], zz["boot"])
    save_json(RESULTS / "e11c_noise_ansatz.json", out)
    for k, v in out.items():
        print(f"{k}: {v}")
