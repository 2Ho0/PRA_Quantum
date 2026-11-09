"""E23: the shape of the correlated U(1)-breaking process, and why the neutral classes stay above 1.2 (diagnostic).

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0), analysed
here from the public raw shots. E22 found that M^2 (the mean ZZ over all pairs) grows in both initial states; E22b
modelled it with collective flips and failed the rule of E20 (neutral classes <= 1.2).
(A) Shape of the M^2 growth on t >= 2 ms (bootstrap over the 200 settings): curvature of a quadratic fit, the split of
    the mean ZZ into the product of the single-site means and the connected part, the per-pair connected slopes by
    ion distance and against the pair's transverse correlation FF_ij = <X_iX_j + Y_iY_j>, and the excursion of the
    connected part at t = 0.5 ms.
(B) Where the neutral error of Xb15, Xb15u, Xb15uf and Xb15uc sits: calibration factor per state and per time window,
    and the share of the squared deviation along the uniform (conserved-sum) direction of each class.
(C) Window dependence: neutral factors on t > 3 and t > 5 ms of models learned on 1.5, 3 and 5 ms (E11), with the Xb15
    bootstrap spread for every model.
(D) Screening of mechanisms: the response of each E11f truth (truth minus control, both U(1)-symmetric except the
    control's collective noise) is added to Xb15uf with an amplitude fitted on the neutral classes of one initial state,
    and the factors are evaluated on the other state.
Status: diagnostic; all numbers were first computed in scratch runs while the script was written. (D) chose the
mechanism tested in E23c.
"""

import sys

import numpy as np

from common import CAT, INK, MUTED, ORDINAL, RESULTS, save_figure, save_json, style
from e6_ion_learning import N, PAIRS, ansatz, observables, per_setting_tables, series_from
from e6c_validation import charge_classes, combine, combine_sem, load
from e11f_nonmarkov_truths import load_truth
from qerr.learning import needed_strings

N_BOOT, T_LATE, T_LEARN = 100, 2.0, 1.5
NEUTRAL = ("q=0  Z_i", "q=0  Z_iZ_j", "q=0  flip-flop")
SCREEN = ("local white 5", "local qs 50", "local qs 150", "global J 10%")


# ---------------- (A) shape of the M^2 growth ----------------
def m2_shape(tables, times, w=None):
    sr = series_from(tables, w)
    late = times >= T_LATE - 1e-9
    dist = np.array([j - i for i, j in PAIRS])
    out = []
    for s in (0, 1):
        z = np.array([sr[(s, ((i, "Z"),))] for i in range(N)])
        zz = np.array([sr[(s, ((i, "Z"), (j, "Z")))] for i, j in PAIRS])
        ff = np.array([sr[(s, ((i, "X"), (j, "X")))] + sr[(s, ((i, "Y"), (j, "Y")))] for i, j in PAIRS])
        conn = zz - np.array([z[i] * z[j] for i, j in PAIRS])
        q = np.polyfit(times[late] - 6.0, zz.mean(0)[late], 2)
        qc = np.polyfit(times[late] - 6.0, conn.mean(0)[late], 2)
        slope = np.array([np.polyfit(times[late], c[late], 1)[0] for c in conn])
        r = np.corrcoef(slope, ff[:, late].mean(1))[0, 1]
        spike = (conn[:, 1] - 0.5 * (conn[:, 0] + conn[:, 2])).mean()
        out.append(np.r_[q[0], q[1], qc[0], qc[1], r, [slope[dist == d].mean() for d in range(1, N)], spike])
    return np.array(out)


SHAPE = (["mean ZZ: curvature (per ms^2)", "mean ZZ: slope at 6 ms (per ms)", "connected: curvature",
          "connected: slope at 6 ms", "corr(pair slope, FF_ij)"] + [f"connected slope, distance {d}" for d in range(1, N)]
         + ["connected excursion at 0.5 ms"])


# ---------------- (B), (C), (D) ----------------
def factor(data, sem, bands, pred, coeffs, states, mask):
    d, p, s = combine(data, coeffs), combine(pred, coeffs), combine_sem(sem, coeffs)
    sd = combine(bands, coeffs).std(axis=0, ddof=1)
    st = list(states)
    return float(np.sqrt(np.mean((d - p)[st][:, mask] ** 2) / np.mean((sd**2 + s**2)[st][:, mask])))


def decomposition(z, bands, preds, cls, times):
    oos = times > T_LEARN + 1e-9
    windows = {"1.5 < t <= 4": oos & (times <= 4 + 1e-9), "4 < t <= 10": times > 4 + 1e-9}
    out = {}
    for name in NEUTRAL:
        co = cls[name]
        d, s = combine(z["data"], co), combine_sem(z["sem"], co)
        sd = combine(bands, co).std(axis=0, ddof=1)
        u = np.ones(d.shape[-1]) / np.sqrt(d.shape[-1])
        row = {}
        for v, p in preds.items():
            dev = d - combine(p, co)
            du = (dev @ u)[..., None] * u
            rest = dev - du
            row[v] = {"both states": factor(z["data"], z["sem"], bands, p, co, (0, 1), oos),
                      "state 1": factor(z["data"], z["sem"], bands, p, co, (0,), oos),
                      "state 2": factor(z["data"], z["sem"], bands, p, co, (1,), oos),
                      **{wn: factor(z["data"], z["sem"], bands, p, co, (0, 1), w) for wn, w in windows.items()},
                      "share of the uniform direction": float(np.sum(du[:, oos] ** 2) / np.sum(dev[:, oos] ** 2)),
                      "factor of the rest": float(np.sqrt(np.mean(rest[:, oos] ** 2) / np.mean((sd**2 + s**2)[:, oos])))}
        out[name] = row
    return out


def window_dependence(z, bands, cls, times):
    e20, e22b = np.load(RESULTS / "e20_flip_models.npz"), np.load(RESULTS / "e22b_collective_flips.npz")
    preds = {k: z["pred_" + k] for k in ("X15", "Xb15", "X30", "Xb30", "X50")}
    preds.update({"Xb15uf": e20["pred_Xb15uf"], "Xb15uc": e22b["pred"]})
    names = list(NEUTRAL) + ["all 435 observables"]
    return {f"t > {t0:g} ms": {v: {n: factor(z["data"], z["sem"], bands, p, cls[n], (0, 1), times > t0 + 1e-9) for n in names}
                               for v, p in preds.items()} for t0 in (3.0, 5.0)}


def screening(z, bands, base, cls, times):
    obs = observables()
    strings = sorted(set(obs) | set(needed_strings(obs, ansatz())))
    idx = [strings.index(O) for O in obs]
    ctrl = load_truth("control")["coarse"]
    oos = times > T_LEARN + 1e-9
    names = list(NEUTRAL) + ["all 435 observables"]
    out = {"Xb15uf": {f"state {s + 1}": {n: factor(z["data"], z["sem"], bands, base, cls[n], (s,), oos) for n in names}
                      for s in (0, 1)}}
    for name in SCREEN:
        c = load_truth(name)["coarse"]
        R = np.stack([c[s][:, idx] - ctrl[s][:, idx] for s in (0, 1)])
        row = {}
        for cal in (0, 1):
            num = den = 0.0
            for n in NEUTRAL:
                co = cls[n]
                d, p, r = combine(z["data"], co)[cal], combine(base, co)[cal], combine(R, co)[cal]
                w = 1.0 / (combine(bands, co).std(axis=0, ddof=1)[cal] ** 2 + combine_sem(z["sem"], co)[cal] ** 2)
                num += np.sum((w * r * (d - p))[oos])
                den += np.sum((w * r * r)[oos])
            a = num / den
            pred = base + a * R
            row[f"amplitude from state {cal + 1}"] = float(a)
            row[f"factors on state {2 - cal}"] = {n: factor(z["data"], z["sem"], bands, pred, cls[n], (1 - cal,), oos)
                                                  for n in names}
        out[name] = row
    return out


def run():
    times, s1, s2, bases = load()
    tables = per_setting_tables((s1, s2), bases, observables())
    rng = np.random.default_rng(5)
    n_set = bases.shape[0]
    shape = [m2_shape(tables, times)] + [m2_shape(tables, times, np.bincount(rng.integers(0, n_set, n_set),
                                                                             minlength=n_set).astype(float))
                                         for _ in range(N_BOOT)]
    return np.array(shape)


def analyse(shape):
    z = np.load(RESULTS / "e11_model_selection.npz")
    e20, e22b = np.load(RESULTS / "e20_flip_models.npz"), np.load(RESULTS / "e22b_collective_flips.npz")
    bands = np.load(RESULTS / "e15_coverage.npz")["band_Xb15"]
    times = z["times"]
    cls = dict(charge_classes(observables()), **{"all 435 observables": [{m: 1.0} for m in range(435)]})
    sd = shape[1:].std(axis=0, ddof=1)
    out = {"(A) shape of the M^2 growth, t >= 2 ms (value, bootstrap sd)": {
        f"state {s + 1}": {n: [float(shape[0, s, m]), float(sd[s, m])] for m, n in enumerate(SHAPE)} for s in (0, 1)}}
    preds = {"Xb15": z["pred_Xb15"], "Xb15u": e20["pred_Xb15u"], "Xb15uf": e20["pred_Xb15uf"], "Xb15uc": e22b["pred"]}
    out["(B) neutral error by state, window and direction"] = decomposition(z, bands, preds, cls, times)
    out["(C) neutral factors of models learned on longer windows"] = window_dependence(z, bands, cls, times)
    out["(D) screening: truth responses added to Xb15uf, amplitude from one state"] = screening(
        z, bands, e20["pred_Xb15uf"], cls, times)
    return out


def plot(shape, out):
    plt = style()
    z = np.load(RESULTS / "e11_model_selection.npz")
    times = z["times"]
    obs = observables()
    k = {O: m for m, O in enumerate(obs)}
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    ax = axes[0]
    for s, (mk, c) in enumerate((("o", CAT[0]), ("s", CAT[1]))):
        zz = np.mean([z["data"][s][:, k[((i, "Z"), (j, "Z"))]] for i, j in PAIRS], axis=0)
        prod = np.mean([z["data"][s][:, k[((i, "Z"),)]] * z["data"][s][:, k[((j, "Z"),)]] for i, j in PAIRS], axis=0)
        ax.plot(times, zz - prod, mk + "-", color=c, ms=3.5, lw=1.0, label=f"connected, state {s + 1}")
    ax.axvspan(0.0, T_LEARN, color=MUTED, alpha=0.08, lw=0)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("connected mean ZZ over all pairs")
    ax.legend(fontsize=7)
    ax.set_title("(a) linear growth, early excursion in state 2", fontsize=9)
    ax = axes[1]
    sd = shape[1:].std(axis=0, ddof=1)
    first = SHAPE.index("connected slope, distance 1")
    for s, c in ((0, CAT[0]), (1, CAT[1])):
        ax.errorbar(np.arange(1, N) + 0.1 * (s - 0.5), shape[0, s, first:first + N - 1], yerr=sd[s, first:first + N - 1],
                    fmt="o", color=c, ms=4, capsize=2, label=f"state {s + 1}")
    ax.axhline(0.0, color=MUTED, lw=0.8)
    ax.set_xlabel("ion distance |i - j|")
    ax.set_ylabel("connected ZZ slope, t >= 2 ms (per ms)")
    ax.legend(fontsize=7)
    ax.set_title("(b) no decay with distance: a global process", fontsize=9)
    ax = axes[2]
    dec = out["(B) neutral error by state, window and direction"]
    models = ["Xb15", "Xb15u", "Xb15uf", "Xb15uc"]
    for j, (n, c) in enumerate(zip(NEUTRAL, ORDINAL[1:])):
        ax.plot(range(len(models)), [dec[n][v]["state 2"] for v in models], "D-", color=c, ms=4, lw=1, label=f"{n}, state 2")
        ax.plot(range(len(models)), [dec[n][v]["state 1"] for v in models], "o:", color=c, ms=4, lw=1, label=f"{n}, state 1")
    ax.axhline(1.2, color=MUTED, ls="--", lw=0.8)
    ax.set_xticks(range(len(models)), models)
    ax.set_ylabel("calibration factor, t > 1.5 ms")
    ax.legend(fontsize=6, ncol=2)
    ax.set_title("(c) the neutral error does not depend on the flips", fontsize=9)
    fig.suptitle("E23: shape of the correlated process and the neutral error (data: Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e23_correlated_process")


if __name__ == "__main__":
    path = RESULTS / "e23_correlated_process.npz"
    if "--replot" not in sys.argv:
        np.savez_compressed(path, shape=run())
    shape = np.load(path)["shape"]
    out = analyse(shape)
    save_json(RESULTS / "e23_correlated_process.json", out)
    print(plot(shape, out))
    for key, v in out.items():
        print(f"{key}: {v}")
