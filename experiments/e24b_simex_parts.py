"""E24b (post hoc): which part of the SIMEX correction acts on the neutral classes?

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0), analysed
here from the public raw shots.
E24 (preregistered) passed its rule B narrowly but changed the neutral factors little (1.34/1.41/1.25 ->
1.27/1.39/1.23), while one common scale 1.045 of all two-body terms (E23d grid) gives 1.15/1.21/1.05. The SIMEX
correction is not a common scale: it raises the couplings at distance 1-2 by 4-5% and the current terms (XY - YX) by
46%. On the synthetic data of E24 step 1 (truth known), SIMEX lowers the error of the couplings at distance <= 2 (rms
36.4 -> 33.2 rad/s, slope 0.960 -> 1.000), of gamma_c (12.8 -> 3.4 rad/s) and of beta (16.0 -> 36.3 against 38.2),
leaves the fields about equal, and raises the error of the currents (101 -> 122 rad/s; the currents are hardly
identifiable in the window, slope 0.26 against the truth even without SIMEX) and of the couplings at distance >= 3
(35.0 -> 37.2 rad/s).
Variants, built from the parameters of E24 (no new learning) and predicted as in E20:
  V1  SIMEX for every parameter except the currents, which stay as learned;
  V2  SIMEX only where step 1 shows a gain: the couplings at distance <= 2, gamma_c and beta; the rest as learned;
  V3  (added after V1 and V2) a structured correction: one common SIMEX scale for all two-body terms, the slope of
      the corrected over the learned couplings (1.045 on the data), with SIMEX for gamma_c and beta and the fields as
      learned. On the synthetic data of step 1 the common scale is 1.040 +- 0.002 per realisation and gives the
      smallest coupling error (rms 34.5 rad/s against 35.6 plain and 35.7 per parameter), an unbiased scale (0.9998)
      and no amplification of the currents (102.9 against 101.1 plain and 122.2 per parameter).
Reported: the factors of every class with the Xb15 band, the p-values and Holm decision of E24's rule B, the
no-smoothing guard and the pooled neutral z-score per time.
Status: diagnostic after the E24 verdict. The variants follow from the synthetic study, but they are evaluated on the
same late data as E20-E24, so they cannot confirm anything.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from common import CAT, INK, MUTED, ORDINAL, RESULTS, save_figure, save_json, style  # noqa: E402
from e6_ion_learning import PAIRS, observables  # noqa: E402
from e6c_validation import charge_classes, combine, combine_sem  # noqa: E402
from e9_symmetry_data import extended_ansatz  # noqa: E402
from e20_flip_models import factors, predict_job  # noqa: E402
from e24_eiv_correction import NEUTRAL, holm  # noqa: E402

PATH = RESULTS / "e24b_simex_parts.npz"
T_LEARN = 1.5


def variants():
    """{name: (theta, beta)} of V1 and V2 from the learned and the SIMEX-corrected Xb15uf."""
    e20 = np.load(RESULTS / "e20_flip_models.npz")
    s2 = np.load(RESULTS / "e24_step2.npz")
    th_l, b_l = e20["theta_Xb15uf"], float(e20["beta_Xb15uf"])
    th_c, b_c = s2["theta_c"], float(s2["beta_c"])
    ext = extended_ansatz()
    k = {p: m for m, p in enumerate(ext.h_terms)}
    k0, nd = len(ext.h_terms), len(ext.deph_pairs)
    current = [k[((i, a), (j, b))] for i, j in PAIRS for a, b in ("XY", "YX")]
    near = [k[((i, a), (j, b))] for i, j in PAIRS if j - i <= 2 for a, b in ("XX", "YY")]
    collective = list(range(k0, k0 + nd))
    v1 = th_c.copy()
    v1[current] = th_l[current]
    v2 = th_l.copy()
    v2[near + collective] = th_c[near + collective]
    flip_flop = [k[((i, a), (j, b))] for i, j in PAIRS for a, b in ("XX", "YY")]
    two_body = flip_flop + current
    J_l = th_l[flip_flop].reshape(45, 2).sum(axis=1)
    J_c = th_c[flip_flop].reshape(45, 2).sum(axis=1)
    scale = float(J_c @ J_l / (J_l @ J_l))
    v3 = th_l.copy()
    v3[two_body] = scale * th_l[two_body]
    v3[collective] = th_c[collective]
    return {"V1: SIMEX except the currents": (v1, b_c), "V2: SIMEX for J(d <= 2), gamma_c, beta": (v2, b_c),
            f"V3: common SIMEX scale {scale:.3f} of the couplings, SIMEX gamma_c and beta": (v3, b_c)}


def run():
    names = list(variants())
    done = dict(np.load(PATH)) if PATH.exists() else {}
    todo = [m for m in range(len(names)) if f"pred{m}" not in done]
    if todo:
        bp = (0.0, 0.0, 1.0)
        vs = list(variants().values())
        jobs = [("Xb15uf", vs[m][0], vs[m][1], s, bp) for m in todo for s in (0, 1)]
        with ProcessPoolExecutor(max_workers=2) as pool:
            preds = list(pool.map(predict_job, jobs))
        for j, m in enumerate(todo):
            done[f"pred{m}"] = np.stack(preds[2 * j:2 * j + 2])
        np.savez_compressed(PATH, **done)
    return {n: done[f"pred{m}"] for m, n in enumerate(names)}


def synthetic_parts():
    """Errors per parameter group on the synthetic data of E24 step 1 (truth known): plain, per-parameter SIMEX and
    one common SIMEX scale of the two-body terms (rad/s)."""
    from e14_shot_calibration import truth_theta
    from e24_eiv_correction import to_theta
    from qerr.learning import simex_extrapolate
    z = np.load(RESULTS / "e24_step1.npz")
    phi0, lv = z["phi0"], z["levels"]
    th0 = np.array([to_theta(p)[0] for p in phi0])
    ths = np.array([to_theta(p)[0] for p in simex_extrapolate([phi0, lv[:, 0]])])
    ext = extended_ansatz()
    k = {p: m for m, p in enumerate(ext.h_terms)}
    ff = np.array([[k[((i, "X"), (j, "X"))], k[((i, "Y"), (j, "Y"))]] for i, j in PAIRS])
    cu = np.array([[k[((i, "X"), (j, "Y"))], k[((i, "Y"), (j, "X"))]] for i, j in PAIRS])
    J = lambda t: t[..., ff[:, 0]] + t[..., ff[:, 1]]
    D = lambda t: t[..., cu[:, 0]] - t[..., cu[:, 1]]
    tt = truth_theta()
    Jt, Dt = J(tt), D(tt)
    scale = np.sum(J(ths) * J(th0), axis=1) / np.sum(J(th0) ** 2, axis=1)
    thc = th0.copy()  # as V3: the common scale for the two-body terms, SIMEX for the collective rate
    two = np.r_[ff.ravel(), cu.ravel()]
    thc[:, two] = scale[:, None] * th0[:, two]
    collective = np.arange(len(ext.h_terms), len(ext.h_terms) + len(ext.deph_pairs))
    thc[:, collective] = ths[:, collective]
    dist = np.array([j - i for i, j in PAIRS])
    hz = [k[((i, "Z"),)] for i in range(len(set(i for p in PAIRS for i in p)))]
    rms = lambda x: float(np.sqrt(np.mean(x**2)) * 1e3)
    out = {"common SIMEX scale per realisation: mean, sd": [float(scale.mean()), float(scale.std(ddof=1))]}
    for name, th in (("plain", th0), ("per-parameter SIMEX", ths), ("common SIMEX scale", thc)):
        e = J(th) - Jt
        out[name] = {"J error rms: all, d <= 2, d >= 3": [rms(e), rms(e[:, dist <= 2]), rms(e[:, dist >= 3])],
                     "J scale against the truth": float(np.mean(J(th) @ Jt / (Jt @ Jt))),
                     "current error rms": rms(D(th) - Dt),
                     "current slope against the truth": float(np.mean(D(th) @ Dt / (Dt @ Dt))),
                     "h_z error rms": rms(th[:, hz] - tt[hz]),
                     "gamma_c error rms": rms(th[:, len(ext.h_terms)])}
    out["truth: current rms"] = rms(Dt)
    return out


def analyse(preds):
    z = np.load(RESULTS / "e11_model_selection.npz")
    e20 = np.load(RESULTS / "e20_flip_models.npz")
    s2 = np.load(RESULTS / "e24_step2.npz")
    band = np.load(RESULTS / "e15_coverage.npz")["band_Xb15"]
    nulls = np.load(RESULTS / "e23d_scale_bias.npz")
    times = z["times"]
    oos = times > T_LEARN + 1e-9
    late = times >= 4.0 - 1e-9
    cls = charge_classes(observables())
    i45 = int(np.argmin(np.abs(nulls["scan_lambda"] - 1.045)))
    allp = {"Xb15uf as learned (E20)": e20["pred_Xb15uf"], "SIMEX, all parameters (E24)": s2["pred"], **preds}
    out = {"calibration factors, t > 1.5 ms (Xb15 band)": {n: factors(p, z, band, oos) for n, p in allp.items()}}
    names = list(out["calibration factors, t > 1.5 ms (Xb15 band)"][next(iter(allp))])
    out["calibration factors, t > 1.5 ms (Xb15 band)"]["common scale 1.045 of all two-body terms (E23d grid)"] = {
        n: float(nulls["scan_factors"][i45, m]) for m, n in enumerate(names)}
    rule_b, guard, zt = {}, {}, {}
    for n, p in allp.items():
        f = out["calibration factors, t > 1.5 ms (Xb15 band)"][n]
        pv = np.array([float(np.mean(nulls["null_" + c.replace(" ", "_")] >= f[c])) for c in NEUTRAL])
        rule_b[n] = {"p-values": pv.tolist(), "Holm rejections": holm(pv).tolist(), "passes": bool(not holm(pv).any())}
        cells = [float(np.mean(combine(p, cls[c])[s][late].var(axis=0, ddof=1))
                       / np.mean(combine(e20["pred_Xb15uf"], cls[c])[s][late].var(axis=0, ddof=1)))
                 for c in NEUTRAL for s in (0, 1)]
        guard[n] = [cells, float(np.mean(cells))]
        num = sum(np.mean((combine(z["data"], cls[c]) - combine(p, cls[c])) ** 2, axis=(0, 2)) for c in NEUTRAL)
        den = sum(np.mean(combine(band, cls[c]).std(axis=0, ddof=1) ** 2 + combine_sem(z["sem"], cls[c]) ** 2,
                          axis=(0, 2)) for c in NEUTRAL)
        zt[n] = np.sqrt(num / den).tolist()
    out["rule B of E24 (Holm over the neutral classes, E23d null)"] = rule_b
    out["guard of E24: late variance / as learned, six cells and mean"] = guard
    out["pooled neutral z-score per time"] = {"times": times.tolist(), **zt}
    out["synthetic data of E24 step 1 (truth known): errors per parameter group (rad/s)"] = synthetic_parts()
    return out


def plot(out):
    plt = style()
    fac = out["calibration factors, t > 1.5 ms (Xb15 band)"]
    v3 = next(n for n in fac if n.startswith("V3"))
    order = ["Xb15uf as learned (E20)", "SIMEX, all parameters (E24)", "V1: SIMEX except the currents",
             "V2: SIMEX for J(d <= 2), gamma_c, beta", v3, "common scale 1.045 of all two-body terms (E23d grid)"]
    colors = [MUTED, ORDINAL[0], CAT[0], CAT[2], INK, CAT[1]]
    marks = ["D", "s", "o", "o", "*", "^"]
    short = ["as learned (E20)", "SIMEX, all parameters (E24)", "V1: SIMEX, currents as learned",
             "V2: SIMEX for J(d <= 2) and the rates", "V3: common SIMEX scale + SIMEX rates",
             "common scale only (E23d)"]
    classes = list(fac[order[0]])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    ax = axes[0]
    for j, (v, c, mk, lab) in enumerate(zip(order, colors, marks, short)):
        ax.plot([fac[v][n] for n in classes], np.arange(len(classes)) + 0.11 * (j - 2.5), mk, color=c,
                ms=7 if mk == "*" else 5, label=lab)
    ax.axvline(1.0, color=MUTED, ls=":", lw=1)
    ax.axvline(1.2, color=MUTED, ls="--", lw=0.8)
    ax.set_yticks(range(len(classes)), classes)
    ax.invert_yaxis()
    ax.set_xlabel("calibration factor, t > 1.5 ms")
    ax.legend(fontsize=6.3, loc="upper left")
    ax.set_title("(a) which part of the correction acts", fontsize=9)
    ax = axes[1]
    zt = out["pooled neutral z-score per time"]
    t = np.array(zt["times"])
    for v, c, lab in zip(order[:5], colors[:5], short[:5]):
        ax.plot(t, zt[v], "-o", color=c, lw=1.3, ms=3, label=lab)
    ax.axvspan(0.0, T_LEARN, color=MUTED, alpha=0.08, lw=0)
    ax.axhline(1.0, color=MUTED, lw=0.6, ls=":")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("neutral z-score (rms deviation / rms spread)")
    ax.legend(fontsize=6.3, loc="upper left")
    ax.set_title("(b) neutral error against time", fontsize=9)
    fig.suptitle("E24b: parts of the SIMEX correction (post hoc; data: Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e24b_simex_parts")


if __name__ == "__main__":
    preds = run()
    out = analyse(preds)
    save_json(RESULTS / "e24b_simex_parts.json", out)
    print(plot(out))
    for key, v in out.items():
        if key != "pooled neutral z-score per time":
            print(f"{key}: {v}")
    sys.stdout.flush()
