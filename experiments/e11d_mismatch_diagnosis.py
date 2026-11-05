"""E11d: where does the charged-sector mismatch of E11c come from?

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0).
E11c found that the single-ion noise block on top of the minimal model (U(1)-symmetric couplings, single-site fields,
collective dephasing) returns non-PSD Kossakowski blocks with large charge +-1 and +-2 entries. This script refits that
model under four alternative explanations and checks whether the unphysical entries go away:
  base     t <= 1.5 ms (E11c);
  late     window [0.5, 2] ms: the first point is excluded (an initial transient or state-preparation error);
  late3    window [0.5, 3] ms;
  spam     t <= 1.5 ms on contrast-corrected data (readout and preparation contrast, E9);
  drift    t <= 1.5 ms with linearly drifting single-site fields B_i + B_i' t (AC Stark transients);
  rotation t <= 1.5 ms after undoing per-ion readout-axis tilts estimated (Kabsch) from the t = 0 Bloch vectors;
  couplings t <= 1.5 ms with the charge +-2 couplings XX - YY and XY + YX of every pair freed (not U(1)-symmetric).
Defects from the E11 reference trajectory (results/e11_stage1.npz); bootstrap over the 200 settings (50 replicates).
Verdict from the central values, not from significance counts (a noisier variant passes a 3 sd test more easily):
removed if no charged site mean exceeds 3 sd and at most one ion has a non-PSD XY block, reduced if the rms of the
charged site means falls below half of the base value.
"""

import itertools
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import lsq_linear

from common import RESULTS, save_json
from e6_ion_learning import N, PAIRS, RAD_S, ansatz, observables, per_setting_tables, series_from
from e6c_validation import load
from e9_symmetry_data import COMBOS, contrast_corrected, extended_ansatz
from e11_model_selection import N_SUB
from e11c_noise_ansatz import constrained_map, summary
from qerr.learning import integral_system, needed_strings, refine, trajectory_defect

VARIANTS = {"base": (0.0, 1.5, False, False), "late": (0.5, 2.0, False, False), "late3": (0.5, 3.0, False, False),
            "spam": (0.0, 1.5, True, False), "drift": (0.0, 1.5, False, True), "rotation": (0.0, 1.5, "rot", False),
            "couplings": (0.0, 1.5, False, False)}
N_BOOT = 50
_W = {}


def readout_rotations(series):
    """Per-ion rotation R_i (Kabsch) taking the nominal t = 0 Bloch vectors of both states to the measured ones."""
    th = 0.33 * np.pi
    nominal = np.array([[np.sin(th), 0.0, -np.cos(th)], [0.0, 1.0, 0.0]])
    rots = []
    for i in range(N):
        meas = np.array([[series[(s, ((i, a),))][0] for a in "XYZ"] for s in (0, 1)])
        U, _, Vt = np.linalg.svd(meas.T @ nominal)
        d = np.sign(np.linalg.det(U @ Vt))
        rots.append(U @ np.diag([1.0, 1.0, d]) @ Vt)   # meas ~ R nominal
    return rots


def unrotated(series, rots):
    """Undo tilted readout axes: measured <A_i B_j ..> = sum (R_i)_{A a} (R_j)_{B b} .. <a_i b_j ..>, so true = R^T m."""
    out = {}
    for (s, S), _ in series.items():
        if S == () or (s, S) in out:
            continue
        val = 0.0
        sites = [i for i, _ in S]
        for letters in np.ndindex(*(3,) * len(S)):
            coef = np.prod([rots[i][b, "XYZ".index(a)] for (i, a), b in zip(S, letters)])
            key = (s, tuple((i, "XYZ"[b]) for i, b in zip(sites, letters)))
            if key not in series:
                break
            val = val + coef * series[key]
        else:
            out[(s, S)] = val
    return out


def charge2_columns(ext):
    """theta_ext columns of the charge +-2 couplings XX - YY and XY + YX of every pair."""
    k = {p: m for m, p in enumerate(ext.h_terms)}
    cols = []
    for i, j in PAIRS:
        xx, yy, xy, yx = (k[((i, a), (j, b))] for a, b in ("XX", "YY", "XY", "YX"))
        for m, n, sign in ((xx, yy, -1.0), (xy, yx, 1.0)):
            c = np.zeros(ext.n_params)
            c[m], c[n] = 1.0, sign
            cols.append(c)
    return np.array(cols).T


def fit_variant(series, name, defects):
    ta, tb, spam, drift = VARIANTS[name]
    ext, obs = extended_ansatz(), observables()
    if spam == "rot":
        series = unrotated(series, readout_rotations(series))
    elif spam:
        series = contrast_corrected(series)[0]
    A, y, _ = integral_system(obs, ext, series, _W["times"], tb, t_start=ta, defect=defects[(ta, tb, False)])
    T = constrained_map(ext)
    if name == "couplings":
        T = np.column_stack([T, charge2_columns(ext)])
    M = A @ T
    if drift:  # B_i(t) = B_i + B_i' t: the field columns weighted by t
        At, _, _ = integral_system(obs, ext, series, _W["times"], tb, t_start=ta, defect=defects[(ta, tb, True)],
                                   weight=lambda s: s)
        k = {p: m for m, p in enumerate(ext.h_terms)}
        fields = [k[((i, a),)] for i in range(N) for a in "XYZ"]
        M = np.column_stack([M, At[:, fields]])
    phi = lsq_linear(M, y, method="bvls", tol=1e-12).x
    return T @ phi[:T.shape[1]]


def _init(tables, times, defects):
    _W.update(tables=tables, times=times, defects=defects)


def replicate(w):
    series = series_from(_W["tables"], w)
    return np.array([fit_variant(series, v, _W["defects"]) for v in VARIANTS])


def run():
    times, s1, s2, bases = load()
    ext, obs = extended_ansatz(), observables()
    traj_strings = needed_strings(obs, ansatz())
    strings = sorted({tuple(zip(sites, letters)) for w in (1, 2, 3)  # every letter combination, for the rotation
                      for sites in itertools.combinations(range(N), w) for letters in itertools.product("XYZ", repeat=w)})
    tables = per_setting_tables((s1, s2), bases, strings)
    z = np.load(RESULTS / "e11_stage1.npz")
    fine = refine(times, N_SUB)
    traj = [z["traj0"], z["traj1"]]
    defects = {}
    for ta, tb, _, drift in VARIANTS.values():
        defects[(ta, tb, False)] = trajectory_defect(fine, traj, traj_strings, times, tb, t_start=ta)
        if drift:
            defects[(ta, tb, True)] = trajectory_defect(fine, traj, traj_strings, times, tb, t_start=ta,
                                                        weight=lambda s: s)
    _init(tables, times, defects)
    central = replicate(None)
    rng = np.random.default_rng(2030)
    n_set = bases.shape[0]
    weights = [np.bincount(rng.integers(0, n_set, n_set), minlength=n_set).astype(float) for _ in range(N_BOOT)]
    with ProcessPoolExecutor(max_workers=4, initializer=_init, initargs=(tables, times, defects)) as pool:
        boot = np.array(list(pool.map(replicate, weights)))
    return central, boot


def analyse(central, boot):
    ext = extended_ansatz()
    out = {}
    charged = [n for n in COMBOS if not n.startswith(("flip", "imbalance"))]
    for i, v in enumerate(VARIANTS):
        cv, em, ab, gc = summary(central[i], ext)
        bs = [summary(th, ext) for th in boot[:, i]]
        row = {"collective gamma_c (rad/s)": float(gc)}
        for name in COMBOS:
            vals = np.array([b[0][name] for b in bs]).mean(axis=1)
            row[name] = [float(cv[name].mean()), float(vals.std(ddof=1))]
        S = em + ab
        row["total flip rate site mean (rad/s)"] = [float(S.mean()), float(np.std([(b[1] + b[2]).mean() for b in bs],
                                                                                  ddof=1))]
        row["ions with non-PSD XY block"] = int(np.sum(np.abs(cv["SXX-SYY"]) > cv["flip S=SXX+SYY"]))
        row["charged site means beyond 3 sd"] = int(sum(abs(row[n][0]) > 3 * row[n][1] for n in charged))
        row["rms of charged site means (rad/s)"] = float(np.sqrt(np.mean([row[n][0] ** 2 for n in charged])))
        row["median bootstrap sd of charged site means (rad/s)"] = float(np.median([row[n][1] for n in charged]))
        if v == "couplings":
            k = {p: m for m, p in enumerate(ext.h_terms)}
            th = central[i] * RAD_S
            # H = J (XX+YY)/2 + K (XX-YY)/2 + K' (XY+YX)/2 + ..., so J = th_XX + th_YY and K = th_XX - th_YY (E11)
            c2 = [np.hypot(th[k[((a, "X"), (b, "X"))]] - th[k[((a, "Y"), (b, "Y"))]],
                           th[k[((a, "X"), (b, "Y"))]] + th[k[((a, "Y"), (b, "X"))]]) for a, b in PAIRS]
            nn = [th[k[((a, "X"), (a + 1, "X"))]] + th[k[((a, "Y"), (a + 1, "Y"))]] for a in range(N - 1)]
            row["charge-2 coupling |K + i K'|, max over pairs (rad/s)"] = float(np.max(c2))
            row["J_nn, mean (rad/s)"] = float(np.mean(nn))
        out[v] = row
    base = out["base"]

    def verdict(r):
        if r["charged site means beyond 3 sd"] == 0 and r["ions with non-PSD XY block"] <= 1:
            return "removes the charged mismatch"
        if r["rms of charged site means (rad/s)"] < 0.5 * base["rms of charged site means (rad/s)"]:
            return "reduces it"
        return "does not remove it"

    out["verdict"] = {v: verdict(out[v]) for v in VARIANTS if v != "base"}
    return out


if __name__ == "__main__":
    if "--replot" not in sys.argv:
        central, boot = run()
        np.savez_compressed(RESULTS / "e11d_mismatch_diagnosis.npz", central=central, boot=boot)
    zz = np.load(RESULTS / "e11d_mismatch_diagnosis.npz")
    out = analyse(zz["central"], zz["boot"])
    save_json(RESULTS / "e11d_mismatch_diagnosis.json", out)
    for k, v in out.items():
        print(f"{k}: {v}")
