"""E25b: where the in-model bias of estimator S comes from (post hoc diagnostic, not preregistered)

Data: the synthetic control truth of E11f (Y15 couplings, currents and z fields on the measured initial states of
Kraft et al.'s data; quasi-static collective noise; no transverse fields, flips or emission). No data of the late
record are used.
Question (after E25's rule S failed): on synthetic data inside the model class, the late-time predictions of S are
biased in the neutral classes by 1.5-2.2 times the rms of its band, while their spread across realisations matches
the band (1.02-1.05). Which parameter group carries the bias?
Method: the 8 fits of E25 step 2 are recomputed with the same seeds (checked against the saved lambda and beta). The
truth's model is predicted as in E25 with one parameter group replaced by its mean estimate over the 8 fits:
  truth            route check, must reproduce the exact truth;
  currents zero    S's choice of E25 step 0;
  couplings S      lambda times the learned flip-flop couplings;
  couplings LS     the flip-flop couplings of least squares;
  z fields         the z fields as learned (S keeps them);
  all of S         the mean theta and beta of S, a first-order stand-in for S's mean prediction.
Charge-0 observables do not see the collective dephasing, so the hybrids keep the truth's beta (quasi-static noise as a
rate beta t). Reported: bias rms / band rms (E25 step-1 band, t > 1.5 ms) per neutral class for every hybrid, next to
the measured bias of S and least squares over the 8 realisations of E25 step 2.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from common import CAT, INK, MUTED, RESULTS, save_figure, save_json, style  # noqa: E402
from e6_ion_learning import N, PAIRS  # noqa: E402
from e6c_validation import load  # noqa: E402
from e9_symmetry_data import extended_ansatz  # noqa: E402
from e11f_nonmarkov_truths import grids, load_truth  # noqa: E402
from e14_shot_calibration import truth_theta  # noqa: E402
from e24_eiv_correction import N_WIN, NEUTRAL, T_LEARN, strings_list  # noqa: E402
from e25_structured_eiv import (N_SYN, WORKERS, _init_syn, class_matrices, indices, predict,  # noqa: E402
                                synthetic_fit)
from qerr.learning import trajectory_defect  # noqa: E402

FITS = RESULTS / "e25b_fits.npz"
HYBRIDS = ("truth", "currents zero", "couplings S", "couplings LS", "z fields", "all of S")


def fits():
    """theta and beta of least squares and of S for the 8 realisations of E25 step 2 (same seeds)."""
    if FITS.exists():
        return dict(np.load(FITS))
    times = load()[0]
    ctrl = load_truth("control")
    fine, _ = grids(times)
    strings, tw = strings_list(), times[:N_WIN]
    d = trajectory_defect(fine, ctrl["fine"], strings, tw, T_LEARN)
    dt = trajectory_defect(fine, ctrl["fine"], strings, tw, T_LEARN, weight=lambda s: s)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_syn,
                             initargs=([p[:N_WIN] for p in ctrl["prob"]], d, dt, "zero")) as pool:
        out = list(pool.map(synthetic_fit, range(N_SYN)))
    print(f"fits: {time.time() - t0:.0f} s", flush=True)
    res = {"theta_ls": np.array([o[0][0] for o in out]), "beta_ls": np.array([o[0][1] for o in out]),
           "theta_s": np.array([o[1][0] for o in out]), "beta_s": np.array([o[1][1] for o in out]),
           "lam": np.array([o[1][2] for o in out])}
    np.savez_compressed(FITS, **res)
    return res


def truth_parameters():
    """theta of the control truth without Markovian dephasing, and its quasi-static noise as the rate beta t."""
    th = truth_theta().copy()
    k0 = len(extended_ansatz().h_terms)
    beta = 2.0 * th[k0] / 1.5  # beta_s of E14 (f = 1)
    th[k0:] = 0.0
    return th, float(beta)


def hybrids(f):
    ff, cu, _ = indices()
    ext = extended_ansatz()
    k = {p: m for m, p in enumerate(ext.h_terms)}
    hz = np.array([k[((i, "Z"),)] for i in range(N)])
    th_t, beta_t = truth_parameters()
    ts, tl = f["theta_s"].mean(axis=0), f["theta_ls"].mean(axis=0)
    out = {"truth": (th_t, beta_t)}
    h = th_t.copy()
    h[cu] = 0.0
    out["currents zero"] = (h, beta_t)
    h = th_t.copy()
    h[ff] = ts[ff]
    out["couplings S"] = (h, beta_t)
    h = th_t.copy()
    h[ff] = tl[ff]
    out["couplings LS"] = (h, beta_t)
    h = th_t.copy()
    h[hz] = ts[hz]
    out["z fields"] = (h, beta_t)
    out["all of S"] = (ts, float(f["beta_s"].mean()))
    return out


def run(f):
    hy = hybrids(f)
    jobs = [(f"e25b_{name.replace(' ', '_')}", th, be, s, 0.0, 0.0) for name, (th, be) in hy.items() for s in (0, 1)]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        preds = list(pool.map(predict, jobs))
    return {name: np.stack(preds[2 * j:2 * j + 2]) for j, name in enumerate(hy)}


def analyse(f, preds):
    times = load()[0]
    oos = times > T_LEARN + 1e-9
    s1, s2 = np.load(RESULTS / "e25_step1.npz"), np.load(RESULTS / "e25_step2.npz")
    band, truth = s1["band"], s2["truth"]
    Cs = class_matrices()
    classes = list(NEUTRAL) + ["all 435 observables"]
    sd2 = {n: float(np.mean(((band @ Cs[n]).std(axis=0, ddof=1) ** 2)[:, oos])) for n in classes}

    def ratio(dev, n):
        return float(np.sqrt(np.mean(((dev @ Cs[n])[:, oos]) ** 2) / sd2[n]))

    ff, cu, _ = indices()
    nn = np.array([m for m, (i, j) in enumerate(PAIRS) if j == i + 1])

    def current(th):
        return th[..., cu[:, 0]] - th[..., cu[:, 1]]

    def rms(x):
        return float(np.sqrt(np.mean(np.asarray(x) ** 2)) * 1e3)

    tr, _ = truth_parameters()
    out = {"refit check: lambda and beta of S equal E25 step 2": bool(
        np.allclose(f["lam"], s2["lam"]) and np.allclose(f["beta_s"], s2["beta_S"])),
        "lambda of S per realisation": f["lam"].tolist(),
        "mean coupling scale, learned over truth (J_nn): least squares, S": [
            float(np.mean(f[key][:, ff].sum(axis=2)[:, nn].mean(axis=0) / tr[ff].sum(axis=1)[nn]))
            for key in ("theta_ls", "theta_s")],
        "rms of the mean current error (rad/s): least squares, S (zero); truth current rms": [
            rms(current(f[key]).mean(axis=0) - current(tr)) for key in ("theta_ls", "theta_s")] + [rms(current(tr))],
        "bias rms / band rms (E25 step-1 band, t > 1.5 ms)": {}}
    tab = out["bias rms / band rms (E25 step-1 band, t > 1.5 ms)"]
    for name, P in preds.items():
        tab[f"hybrid: {name}"] = {n: ratio(P - truth, n) for n in classes}
    for est, key in (("measured: S, mean of 8 predictions", "S"), ("measured: least squares, mean of 8", "plain")):
        tab[est] = {n: ratio(s2[key].mean(axis=0) - truth, n) for n in classes}
    # additivity of the single-group hybrids against "all of S" (neutral classes, rms over t > 1.5 ms)
    d_sum = sum(preds[h] - preds["truth"] for h in ("currents zero", "couplings S", "z fields"))
    out["single groups summed (currents zero + couplings S + z fields) / all of S, rms ratio per class"] = {
        n: ratio(d_sum, n) / max(ratio(preds["all of S"] - preds["truth"], n), 1e-12) for n in NEUTRAL}
    # time profile of the neutral bias (pooled over the three neutral classes)
    def prof(P):
        num = sum(np.mean(((P - truth) @ Cs[c]) ** 2, axis=(0, 2)) for c in NEUTRAL)
        den = sum(np.mean((band @ Cs[c]).std(axis=0, ddof=1) ** 2, axis=(0, 2)) for c in NEUTRAL)
        r = np.sqrt(num / np.where(den > 1e-12, den, np.nan))
        r[0] = np.nan  # t = 0: the initial state is fixed, the band has no width
        return r.tolist()
    out["neutral bias / band rms per time"] = {"times": times.tolist(),
                                               **{h: prof(preds[h]) for h in ("currents zero", "couplings S",
                                                                              "z fields", "all of S")},
                                               "measured S": prof(s2["S"].mean(axis=0))}
    return out


def plot(out):
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    ax = axes[0]
    tab = out["bias rms / band rms (E25 step-1 band, t > 1.5 ms)"]
    rows = [f"hybrid: {h}" for h in HYBRIDS] + ["measured: S, mean of 8 predictions",
                                                 "measured: least squares, mean of 8"]
    cols = (INK, CAT[0], CAT[1])
    for j, (n, c) in enumerate(zip(NEUTRAL, cols)):
        ax.plot([tab[r][n] for r in rows], np.arange(len(rows)) + 0.2 * (j - 1), "o", color=c, ms=4.5, label=n)
    ax.axhline(len(HYBRIDS) - 0.5, color=MUTED, lw=0.6)
    ax.axvline(1.0, color=MUTED, lw=0.6, ls=":")
    ax.set_yticks(range(len(rows)), [r.split(": ")[1] for r in rows])
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlabel("late bias rms / band rms (t > 1.5 ms)")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.set_title("(a) truth with one group replaced by its mean estimate", fontsize=9)
    ax = axes[1]
    p = out["neutral bias / band rms per time"]
    t = np.array(p["times"])
    for h, c in (("currents zero", CAT[1]), ("couplings S", CAT[0]), ("z fields", MUTED), ("all of S", INK)):
        ax.plot(t, p[h], "-", color=c, lw=1.3, label=f"hybrid: {h}")
    ax.plot(t, p["measured S"], "o", color=INK, ms=3, label="measured S (mean of 8)")
    ax.axvspan(0.0, T_LEARN, color=MUTED, alpha=0.08, lw=0)
    ax.axhline(1.0, color=MUTED, lw=0.6, ls=":")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("neutral bias / band rms")
    ax.set_ylim(0.0, 3.1)
    ax.legend(fontsize=6.5, loc="upper right", ncol=2)
    ax.set_title("(b) neutral bias against time", fontsize=9)
    fig.suptitle("E25b: the in-model bias of S by parameter group (synthetic control truth, post hoc)")
    fig.tight_layout()
    return save_figure(fig, "e25b_bias_decomposition")


if __name__ == "__main__":
    f = fits()
    print("lambda:", np.round(f["lam"], 4), flush=True)
    preds = run(f)
    out = analyse(f, preds)
    save_json(RESULTS / "e25b_bias_decomposition.json", out)
    print(plot(out))
    for key, val in out.items():
        print(key, val, flush=True)
