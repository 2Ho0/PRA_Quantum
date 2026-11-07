"""E22: which generator acts in the first milliseconds, and which features of the record does no model reproduce?

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0), analysed
here from the public raw shots. E20 left the neutral classes of the best model (Xb15uf) at calibration factors of
1.25-1.41; E11h found a flip burst (tau = 0.42 ms) that fits the mean-Z balance but erases the coherences.
(A) Candidate generators added to Xb15uf and learned with it on t <= 1.5 ms (integral method, defect of E11):
      isotropic burst       S_b e^{-t/tau} on every ion (S_XX = S_YY = S_b / 2)
      anisotropic burst     (S_XX, S_YY, S_XY) e^{-t/tau}, uniform over the ions
      z-dephasing burst     local Gamma_ii = g_b e^{-t/tau}
      field burst           uniform (h_x, h_y) e^{-t/tau}
      anisotropic constant  (S_XX, S_YY, S_XY), uniform and constant
      collective constant   D(r) = sum_{i,j} C_AB (A_i r B_j - 1/2 {B_j A_i, r}) over all ion pairs including i = j,
                            A, B in {X, Y}: a transverse noise common to all ions (qerr.learning.correlated_adjoint)
    with tau = 0.42 ms from E11h. Fit quality: chi2 of the integral rows, each weighted by the shot noise of
    <O>(t) - <O>(0); bootstrap over the 200 settings.
(B) Features of the late record, t >= 2 ms, for the data (bootstrap errors) and for every model and truth that covers
    it (Xb15, Y15, the E20 variants, the E11f truths):
      charge-2 floor   alpha_1, alpha_2 from the regression of (XX - YY)/2 and (XY + YX)/2 on C = (XX + YY)/2 over
                       the 45 pairs
      M^2 slope        slope of the mean ZZ over all pairs, (M^2 - N) / (N (N - 1)); every U(1)-covariant generator
                       conserves it and independent flips pull it towards 0
      mean Z           late slope, late level, and the step from 1.5 to 2 ms
(C) Where the charge-2 imbalance K (site mean S_XX - S_YY of the E11c noise block, fit without memory) comes from:
      (C1) K from each initial state alone, for the data and for 20 shot-level realisations of the E11f control
           truth (oracle defect);
      (C2) K after removing the state-1 floor with a per-ion readout map A = [[1 + kappa, eta], [eta, 1 - kappa]]
           on (X, Y), applied to every string and fitted to the late floor of each state;
      (C3) K with the collective block added to the E11c fit.
Status: diagnostic. The central fits of (A), the features (B) and (C2), (C3) were computed in scratch runs while this
script was written; the bootstrap errors and (C1) are new. Fixed before the (C1) run: the imbalance is a property of
each initial state if the data's single-state K lies below the 5th percentile of the control's single-state K for
both states.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from scipy.optimize import lsq_linear  # noqa: E402

from common import CAT, INK, MUTED, RESULTS, save_figure, save_json, style  # noqa: E402
from e6_ion_learning import N, PAIRS, RAD_S, ansatz, observables, per_setting_tables, series_from  # noqa: E402
from e6c_validation import load  # noqa: E402
from e9_symmetry_data import extended_ansatz  # noqa: E402
from e11_model_selection import N_SUB  # noqa: E402
from e11c_noise_ansatz import constrained_map, summary  # noqa: E402
from e11f_nonmarkov_truths import defects, grids, load_truth  # noqa: E402
from e14_shot_calibration import synthetic_series  # noqa: E402
from e20_flip_models import S_51, burst, structure  # noqa: E402
from qerr.learning import (CORRELATED_KEYS, NOISE_KEYS, correlated_adjoint, integral_system,  # noqa: E402
                           needed_strings, refine, trajectory_defect)

T_LEARN, T_LATE = 1.5, 2.0
N_BOOT, N_REAL, WORKERS = 60, 20, 6
CANDIDATES = ("isotropic burst", "anisotropic burst", "z-dephasing burst", "field burst", "anisotropic constant",
              "collective constant")
TRANSVERSE = ("anisotropic burst", "anisotropic constant", "collective constant")  # (S_XX, S_YY, S_XY) blocks
STATE_SETS = ((0, 1), (0,), (1,))
E11F_TRUTHS = ("control", "local white 5", "local qs 50", "local qs 150", "global J 10%", "X15 fields + qs detuning")
_W = {}


# ---------------- shared set-up ----------------
def setup():
    times, s1, s2, bases = load()
    obs = observables()
    strings = sorted(set(obs) | set(needed_strings(obs, ansatz())))
    z = np.load(RESULTS / "e11_stage1.npz")
    traj = [z["traj0"], z["traj1"]]
    fine = refine(times, N_SUB)
    tau = burst()[2]
    return {"times": times, "bases": bases, "obs": obs, "strings": strings, "tau": tau,
            "tables": per_setting_tables((s1, s2), bases, strings),
            "d": trajectory_defect(fine, traj, strings, times, T_LEARN),
            "dt": trajectory_defect(fine, traj, strings, times, T_LEARN, weight=lambda s: s),
            "db": trajectory_defect(fine, traj, strings, times, T_LEARN, weight=lambda s: np.exp(-s / tau)),
            "sem": np.load(RESULTS / "e11_model_selection.npz")["sem"]}


def directions(ext):
    k = {p: m for m, p in enumerate(ext.h_terms)}
    k0, nd = len(ext.h_terms), len(ext.deph_pairs)
    u = {key: n for n, key in enumerate(NOISE_KEYS)}
    out = {}
    for key in ("SXX", "SYY", "SXY"):
        v = np.zeros(ext.n_params)
        for i in range(N):
            v[k0 + nd + len(NOISE_KEYS) * i + u[key]] = 1.0
        out[key] = v
    out["zd"] = np.zeros(ext.n_params)
    for m, (i, j) in enumerate(ext.deph_pairs):
        if i == j:
            out["zd"][k0 + m] = 1.0
    for a in "XY":
        out["h" + a] = np.zeros(ext.n_params)
        for i in range(N):
            out["h" + a][k[((i, a),)]] = 1.0
    return out


def cross_columns(series, rows, times, d):
    """Columns of the cross-ion block (cXX, cYY, cXY) for the rows of an integral system on t <= T_LEARN.

    A row whose terms need a string missing from `series` gets NaN (fit_candidates drops it for every candidate).
    """
    win = times <= T_LEARN + 1e-9
    tw = times[win]
    at = {float(t): m for m, t in enumerate(tw)}
    cache, terms = {}, {}

    def integral(s, S):
        if (s, S) not in cache:
            vals = np.ones(len(tw)) if S == () else series.get((s, S), np.full(len(times), np.nan))[win]
            if np.any(~np.isfinite(vals)):
                cache[(s, S)] = np.full(len(tw), np.nan)
            else:
                sp = CubicSpline(tw, vals)
                cache[(s, S)] = np.array([sp.integrate(tw[0], t) for t in tw]) + d.get((s, S), 0.0)
        return cache[(s, S)]

    cols = []
    for s, O, t in rows:
        if O not in terms:
            terms[O] = {key: correlated_adjoint(O, key) for key in CORRELATED_KEYS}
        cols.append([sum(c * integral(s, R)[at[float(t)]] for R, c in terms[O][key].items()) for key in CORRELATED_KEYS])
    return np.array(cols)


def row_sem(rows, ctx):
    oi = {O: m for m, O in enumerate(ctx["obs"])}
    ti = {float(t): m for m, t in enumerate(ctx["times"])}
    sem = ctx["sem"]
    return np.array([np.hypot(sem[s, ti[float(t)], oi[O]], sem[s, 0, oi[O]]) for s, O, t in rows])


# ---------------- (A) candidates on the learning window ----------------
def fit_candidates(series, ctx):
    ext, obs, times = extended_ansatz(), ctx["obs"], ctx["times"]
    tau = ctx["tau"]
    A, y, rows = integral_system(obs, ext, series, times, T_LEARN, defect=ctx["d"])
    At, _, _ = integral_system(obs, ext, series, times, T_LEARN, defect=ctx["dt"], weight=lambda s: s)
    Ab, _, _ = integral_system(obs, ext, series, times, T_LEARN, defect=ctx["db"], weight=lambda s: np.exp(-s / tau))
    Ac = cross_columns(series, rows, times, ctx["d"])
    keep = np.all(np.isfinite(Ac), axis=1)
    A, At, Ab, Ac, y = A[keep], At[keep], Ab[keep], Ac[keep], y[keep]
    e = row_sem([r for r, k in zip(rows, keep) if k], ctx)
    T, fix, flip, coll = structure(ext, uniform=True)
    v = directions(ext)
    target = y - A @ fix - S_51 * (A @ flip)
    base = [A @ T, At @ coll]
    trans = lambda B: [B @ v["SXX"], B @ v["SYY"], B @ v["SXY"]]
    extra = {"Xb15uf": [], "isotropic burst": [Ab @ flip], "anisotropic burst": trans(Ab),
             "z-dephasing burst": [Ab @ v["zd"]], "field burst": [Ab @ v["hX"], Ab @ v["hY"]],
             "anisotropic constant": trans(A),
             "collective constant": [c + Ac[:, m] for m, c in enumerate(trans(A))]}
    out = {}
    for name, cols in extra.items():
        M = np.column_stack(base + cols)
        phi = np.linalg.lstsq(M, target, rcond=None)[0]
        chi2 = float(np.sum(((target - M @ phi) / e) ** 2))
        out[name] = (chi2, phi[base[0].shape[1] + 1:] * RAD_S)
    return out


# ---------------- (B) late features ----------------
def features(get, times):
    """[(alpha_1, alpha_2, M^2 slope, mean-Z slope, late mean Z, Z(2) - Z(1.5)) for each state]; slopes per ms."""
    late = times >= T_LATE - 1e-9
    i15, i20 = int(np.argmin(np.abs(times - 1.5))), int(np.argmin(np.abs(times - 2.0)))
    out = []
    for s in (0, 1):
        g = lambda i, j, a, b: get(s, ((i, a), (j, b)))[late]
        C = np.concatenate([(g(i, j, "X", "X") + g(i, j, "Y", "Y")) / 2 for i, j in PAIRS])
        F1 = np.concatenate([(g(i, j, "X", "X") - g(i, j, "Y", "Y")) / 2 for i, j in PAIRS])
        F2 = np.concatenate([(g(i, j, "X", "Y") + g(i, j, "Y", "X")) / 2 for i, j in PAIRS])
        zz = np.mean([get(s, ((i, "Z"), (j, "Z"))) for i, j in PAIRS], axis=0)
        zm = np.mean([get(s, ((i, "Z"),)) for i in range(N)], axis=0)
        out.append([C @ F1 / (C @ C), C @ F2 / (C @ C), np.polyfit(times[late], zz[late], 1)[0],
                    np.polyfit(times[late], zm[late], 1)[0], zm[late].mean(), zm[i20] - zm[i15]])
    return np.array(out)


def _init(ctx):
    _W.update(ctx)


def replicate(w):
    series = series_from(_W["tables"], w)
    return fit_candidates(series, _W), features(lambda s, S: series[(s, S)], _W["times"])


# ---------------- (C) sources of K ----------------
def k_of(series, times, d, states, extra=None):
    """Site mean K (rad/s) of the E11c noise block fit without memory; `extra` adds columns (e.g. the cross block)."""
    ext = extended_ansatz()
    A, y, rows = integral_system(observables(), ext, series, times, T_LEARN, states=states, defect=d)
    Tm = constrained_map(ext)
    cols = [A @ Tm]
    if extra is not None:
        X = extra(series, rows)
        keep = np.all(np.isfinite(X), axis=1)
        cols, y = [cols[0][keep], X[keep]], y[keep]
    phi = lsq_linear(np.column_stack(cols), y, method="bvls", tol=1e-12).x
    return float(summary(Tm @ phi[:Tm.shape[1]], ext)[0]["SXX-SYY"].mean())


def _init_control(prob, bases, strings, times, d):
    _W.update(prob=prob, bases=bases, strings=strings, times=times, d=d)


def control_job(seed):
    series = synthetic_series(_W["prob"], _W["bases"], _W["strings"], np.random.default_rng(seed))
    return [k_of(series, _W["times"], _W["d"], st) for st in STATE_SETS]


def xy_partners(S):
    slots = [[(i, b)] if b == "Z" else [(i, "X"), (i, "Y")] for i, b in S]
    return [tuple(c) for c in itertools.product(*slots)]


def readout_corrected(series, strings, times):
    """Series with the per-ion map A^{-1} applied on every X/Y factor; A fitted to the late floor of each state."""
    out, info = dict(series), {}
    ix = {"X": 0, "Y": 1}
    for s in (0, 1):
        a1, a2 = features(lambda st, S: series[(st, S)], times)[s, :2]
        kappa, eta = a1 / 2, a2 / 2
        for _ in range(20):  # alpha_1 = 2 kappa / (1 + kappa^2 + eta^2), alpha_2 = 2 eta / (1 + kappa^2 + eta^2)
            nrm = 1 + kappa**2 + eta**2
            kappa, eta = a1 * nrm / 2, a2 * nrm / 2
        Ainv = np.linalg.inv(np.array([[1 + kappa, eta], [eta, 1 - kappa]]))
        skipped = 0
        for S in strings:
            if all(b == "Z" for _, b in S) or (s, S) not in series:
                continue
            parts = xy_partners(S)
            if any((s, P) not in series for P in parts):
                skipped += 1
                continue
            out[(s, S)] = sum(np.prod([Ainv[ix[a], ix[b]] for (_, a), (_, b) in zip(S, P) if a != "Z"]) * series[(s, P)]
                              for P in parts)
        info[f"state {s + 1}"] = {"kappa": float(kappa), "eta": float(eta), "strings left uncorrected": skipped}
    return out, info


def run_c(ctx):
    times, strings, bases = ctx["times"], ctx["strings"], ctx["bases"]
    series = series_from(ctx["tables"])
    data = {"data, " + "+".join(str(s + 1) for s in st): k_of(series, times, ctx["d"], st) for st in STATE_SETS}
    corrected, info = readout_corrected(series, strings, times)
    data["data after the readout-map correction, 1+2"] = k_of(corrected, times, ctx["d"], (0, 1))
    data["data with the collective block, 1+2"] = k_of(
        series, times, ctx["d"], (0, 1), extra=lambda sr, rows: cross_columns(sr, rows, times, ctx["d"]))
    fine, _ = grids(times)
    ctrl = load_truth("control")
    d_ctrl = defects(fine, ctrl["fine"], strings, times)[0]
    seeds = [104729 * i + 77 for i in range(N_REAL)]
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_control,
                             initargs=(ctrl["prob"], bases, strings, times, d_ctrl)) as pool:
        control = np.array(list(pool.map(control_job, seeds)))
    return data, info, control


def run():
    ctx = setup()
    series = series_from(ctx["tables"])
    central = (fit_candidates(series, ctx), features(lambda s, S: series[(s, S)], ctx["times"]))
    rng = np.random.default_rng(2032)
    n_set = ctx["bases"].shape[0]
    weights = [np.bincount(rng.integers(0, n_set, n_set), minlength=n_set).astype(float) for _ in range(N_BOOT)]
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init, initargs=(ctx,)) as pool:
        boot = list(pool.map(replicate, weights))
    return ctx, central, boot, run_c(ctx)


# ---------------- analysis ----------------
def model_features(times, strings):
    obs = observables()
    z, e = np.load(RESULTS / "e11_model_selection.npz"), np.load(RESULTS / "e20_flip_models.npz")
    oi = {O: m for m, O in enumerate(obs)}
    si = {S: m for m, S in enumerate(strings)}
    preds = {"Xb15": z["pred_Xb15"], "Y15": z["pred_Y15"], **{v: e[f"pred_{v}"] for v in ("Xb15u", "Xb15uf", "Xb15ub")}}
    out = {name: features(lambda s, S, p=p: p[s][:, oi[S]], times) for name, p in preds.items()}
    for name in E11F_TRUTHS:
        c = load_truth(name)["coarse"]
        out["E11f truth: " + name] = features(lambda s, S, c=c: c[s][:, si[S]], times)
    return out


FEATURES = ("floor alpha_1 (XX-YY)", "floor alpha_2 (XY+YX)", "M^2 slope: mean ZZ per ms", "mean-Z slope per ms",
            "late mean Z", "mean-Z step 1.5 -> 2 ms")


def analyse(ctx, central, boot, c_part):
    fits, feat = central
    bfits = [b[0] for b in boot]
    bfeat = np.array([b[1] for b in boot])
    out = {"tau of the bursts (ms)": ctx["tau"], "bootstrap replicates": len(boot)}
    ref = fits["Xb15uf"][0]
    rows = {}
    for name in CANDIDATES:
        dchi = np.array([b[name][0] - b["Xb15uf"][0] for b in bfits])
        par = np.array([b[name][1] for b in bfits])
        row = {"delta chi2 vs Xb15uf": fits[name][0] - ref,
               "delta chi2, bootstrap 5/50/95 percentile": np.percentile(dchi, [5, 50, 95]).tolist(),
               "parameters (rad/s)": fits[name][1].tolist(), "bootstrap sd": par.std(axis=0, ddof=1).tolist()}
        if name in TRANSVERSE:
            ev = lambda p: float(np.linalg.eigvalsh(np.array([[p[0], p[2]], [p[2], p[1]]]))[0])
            row["smallest eigenvalue of the (X, Y) block (rad/s), bootstrap sd"] = [
                ev(fits[name][1]), float(np.std([ev(p) for p in par], ddof=1))]
            row["physical (smallest eigenvalue >= -2 sd)"] = bool(
                row["smallest eigenvalue of the (X, Y) block (rad/s), bootstrap sd"][0]
                >= -2 * row["smallest eigenvalue of the (X, Y) block (rad/s), bootstrap sd"][1])
        elif name in ("isotropic burst", "z-dephasing burst"):
            row["physical (rate >= -2 sd)"] = bool(fits[name][1][0] >= -2 * par[:, 0].std(ddof=1))
        rows[name] = row
    best = min(CANDIDATES, key=lambda n: fits[n][0])
    out["(A) candidates on t <= 1.5 ms"] = rows
    out["(A) reading"] = {"lowest chi2": best,
                          "lowest chi2 and physical": bool(rows[best].get("physical (smallest eigenvalue >= -2 sd)",
                                                                           rows[best].get("physical (rate >= -2 sd)",
                                                                                          True)))}
    sd = bfeat.std(axis=0, ddof=1)
    models = model_features(ctx["times"], ctx["strings"])
    fb = {}
    for s in (0, 1):
        for f, name in enumerate(FEATURES):
            fb[f"state {s + 1}: {name}"] = {"data": [float(feat[s, f]), float(sd[s, f])],
                                            **{m: float(v[s, f]) for m, v in models.items()}}
    out["(B) late features (data: value, bootstrap sd; models: value)"] = fb
    data_k, info, control = c_part
    out["(C) K (rad/s)"] = data_k
    out["(C2) readout map"] = info
    pct = {st: np.percentile(control[:, j], [5, 50, 95]).tolist() for j, st in enumerate(("1+2", "1", "2"))}
    out["(C1) control truth, K 5/50/95 percentile (oracle defect)"] = pct
    single = [data_k["data, 1"] < pct["1"][0], data_k["data, 2"] < pct["2"][0]]
    out["(C1) verdict"] = {"state 1 below the control's 5th percentile": bool(single[0]),
                           "state 2 below the control's 5th percentile": bool(single[1]),
                           "the imbalance is a property of each initial state": bool(all(single))}
    return out, bfeat, models


def plot(out, feat, bfeat, models, times, ctx):
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    ax = axes[0]
    rows = out["(A) candidates on t <= 1.5 ms"]
    for k, name in enumerate(CANDIDATES):
        lo, med, hi = rows[name]["delta chi2, bootstrap 5/50/95 percentile"]
        phys = rows[name].get("physical (smallest eigenvalue >= -2 sd)", rows[name].get("physical (rate >= -2 sd)", True))
        c = CAT[0] if phys else CAT[1]
        ax.plot([lo, hi], [k, k], "-", color=c, lw=1.6)
        ax.plot(rows[name]["delta chi2 vs Xb15uf"], k, "D", color=c, ms=5)
    ax.axvline(0.0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(CANDIDATES)), CANDIDATES)
    ax.invert_yaxis()
    ax.set_xlabel("$\\Delta\\chi^2$ vs Xb15uf on $t \\leq 1.5$ ms (5-95% bootstrap)")
    ax.set_title("(a) candidates (orange: not physical)", fontsize=9)
    series = series_from(ctx["tables"])
    zz = lambda s: np.mean([series[(s, ((i, "Z"), (j, "Z")))] for i, j in PAIRS], axis=0)
    ax = axes[1]
    preds = {"Xb15uf": np.load(RESULTS / "e20_flip_models.npz")["pred_Xb15uf"]}
    oi = {O: m for m, O in enumerate(observables())}
    for s, mk in ((0, "o"), (1, "s")):
        ax.plot(times, zz(s) - zz(s)[0], mk, color=INK, ms=3.5, label=f"data, state {s + 1}")
        p = np.mean([preds["Xb15uf"][s][:, oi[((i, "Z"), (j, "Z"))]] for i, j in PAIRS], axis=0)
        ax.plot(times, p - p[0], "-" if s == 0 else "--", color=CAT[0], lw=1.3, label=f"Xb15uf, state {s + 1}")
    ax.axvspan(0.0, T_LEARN, color=MUTED, alpha=0.08, lw=0)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("mean ZZ over all pairs, change since t = 0")
    ax.legend(fontsize=7)
    ax.set_title("(b) $M^2$ grows in the data", fontsize=9)
    ax = axes[2]
    nn = [(i, i + 1) for i in range(N - 1)]
    for lab, terms, c in (("(XX-YY)/2", (("X", "X", 1), ("Y", "Y", -1)), CAT[0]),
                          ("(XY+YX)/2", (("X", "Y", 1), ("Y", "X", 1)), CAT[2])):
        d = np.mean([sum(w * series[(0, ((i, a), (j, b)))] for a, b, w in terms) / 2 for i, j in nn], axis=0)
        p = np.mean([sum(w * preds["Xb15uf"][0][:, oi[((i, a), (j, b))]] for a, b, w in terms) / 2 for i, j in nn],
                    axis=0)
        ax.plot(times, d, "o", color=c, ms=3.5, label=f"data {lab}")
        ax.plot(times, p, "-", color=c, lw=1.2, label=f"Xb15uf {lab}")
    ax.axhline(0.0, color=MUTED, lw=0.8)
    ax.set_ylim(-0.1, 0.1)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("charge-2 pair combination, nearest neighbours")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("(c) the state-1 charge-2 floor", fontsize=9)
    fig.suptitle("E22: the first milliseconds and the late record (data: Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e22_first_milliseconds")


if __name__ == "__main__":
    path = RESULTS / "e22_first_milliseconds.npz"
    if "--replot" not in sys.argv:
        ctx, central, boot, c_part = run()
        data_k, info, control = c_part
        np.savez_compressed(path, boot_chi2=np.array([[b[0][n][0] for n in ("Xb15uf",) + CANDIDATES] for b in boot]),
                            boot_feat=np.array([b[1] for b in boot]), control=control)
        save_json(RESULTS / "e22_central.json", {"fits": {n: [v[0], v[1].tolist()] for n, v in central[0].items()},
                                                  "features": central[1].tolist(), "K": data_k, "readout": info,
                                                  "boot_params": {n: [b[0][n][1].tolist() for b in boot]
                                                                  for n in CANDIDATES}})
    import json
    ctx = setup()
    cj = json.loads((RESULTS / "e22_central.json").read_text())
    z = np.load(path)
    central = ({n: (v[0], np.array(v[1])) for n, v in cj["fits"].items()}, np.array(cj["features"]))
    names = ("Xb15uf",) + CANDIDATES
    boot = [({n: (z["boot_chi2"][r, k], np.array(cj["boot_params"][n][r]) if n in CANDIDATES else np.zeros(0))
              for k, n in enumerate(names)}, z["boot_feat"][r]) for r in range(z["boot_chi2"].shape[0])]
    out, bfeat, models = analyse(ctx, central, boot, (cj["K"], cj["readout"], z["control"]))
    save_json(RESULTS / "e22_first_milliseconds.json", out)
    print(plot(out, central[1], bfeat, models, ctx["times"], ctx))
    for key, v in out.items():
        print(f"{key}: {v}")
