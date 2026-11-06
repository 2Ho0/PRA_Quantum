"""E17: where does the 1.5x structural error of the neutral classes (E15) come from?

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0), analysed
here from the public raw shots. Models and predictions from E11 (results/e11_model_selection.npz), bands from E15.
The learned models conserve three neutral quantities exactly or nearly: the magnetisation (mean Z), its square (the
mean ZZ over all pairs) and the energy of the symmetric Hamiltonian (Y15). Collective dephasing, white or
quasi-static, conserves all three; a drift of the symmetric couplings conserves the first two. So the split of the
residual between these directions and the rest says which kind of process is missing.
Diagnostics, no acceptance rule; all parts were fixed before this script was run. Part (A) repeats an exploratory
calculation made while designing it.
(A) The neutral residuals of X15 and Xb15 (t > 1.5 ms) projected on the conserved directions (uniform sum of Z_i,
    uniform sum of Z_iZ_j, the J-weighted flip-flops) and on the rest, each with its own calibration factor.
(B) The three conserved quantities over time, data against Y15, X15 and Xb15, with errors from a bootstrap over the
    200 settings (shot noise and setting sampling; not run-to-run drift).
(C) Time-point jitter: second differences of the data series of (B) against the bootstrap errors.
(D) The exact balance d(sum_i Z_i)/dt = sum_i 2 (h^x_i <Y_i> - h^y_i <X_i>) - (non-conserving dissipation), which holds
    for any U(1)-symmetric remainder. Evaluated with the X15 fields on the measured <X_i>, <Y_i>, and fitted with
    uniform coherent fields, an incoherent flip rate or a constant pumping rate.
(E) Collision-like shots: all ten spins equal, against the independent-ion expectation from the per-setting means.
(F) Drift over the acquisition: the conserved-quantity changes estimated from the four quarters of the settings
    (settings are stored in acquisition order k = 1..200 per state; that this is the time order is an assumption).
"""

import sys

import numpy as np
from scipy.interpolate import CubicSpline

from common import CAT, INK, MUTED, ORDINAL, RESULTS, save_figure, save_json, style
from e6_ion_learning import N, PAIRS, RAD_S, ansatz, observables, per_setting_tables, series_from
from e6c_validation import load

T_LEARN, N_BOOT, Z95 = 1.5, 200, 1.96
EMISSION_D52 = 1.0 / 1168.0  # 1/ms, fixed in X15 and Xb15
MODELS = ("Y15", "X15", "Xb15")
QUANTITIES = ("mean Z", "mean ZZ (all pairs)", "energy of Y15 (rad/s)")


def indices():
    k = {p: m for m, p in enumerate(observables())}
    return {"Z": [k[((i, "Z"),)] for i in range(N)], "X": [k[((i, "X"),)] for i in range(N)],
            "Y": [k[((i, "Y"),)] for i in range(N)], "ZZ": [k[((i, "Z"), (j, "Z"))] for i, j in PAIRS],
            "XX": [k[((i, "X"), (j, "X"))] for i, j in PAIRS], "YY": [k[((i, "Y"), (j, "Y"))] for i, j in PAIRS]}


def energy_weights(theta):
    """(observable indices, weights in rad/s) with <H> = sum_k theta_k <P_k> over the terms of the ansatz."""
    k = {p: m for m, p in enumerate(observables())}
    cols, w = zip(*[(k[p], theta[m] * RAD_S) for m, p in enumerate(ansatz().h_terms) if theta[m] != 0.0])
    return list(cols), np.array(w)


def conserved(values, ix, ecols, ew):
    """(..., 3) mean Z, mean ZZ over all pairs and <H_Y15> from (..., n_obs) expectation values."""
    return np.stack([values[..., ix["Z"]].mean(-1), values[..., ix["ZZ"]].mean(-1), values[..., ecols] @ ew], -1)


def as_array(series, s, fill=None):
    """(times, n_obs) values of initial state s; a string without settings in a subset takes the value of `fill`."""
    out = np.array([series.get((s, p), np.full(21, np.nan)) for p in observables()]).T
    return out if fill is None else np.where(np.isnan(out), fill, out)


def bootstrap(tables, rng, n_set=200):
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, n_set, n_set), minlength=n_set).astype(float)
        yield series_from(tables, w)


def calib(dev, band, sem, oos):
    tot = np.sqrt(band**2 + sem**2)
    return float(np.sqrt(np.mean(dev[:, oos] ** 2) / np.mean(tot[:, oos] ** 2)))


def part_a(z, bands, ix, oos):
    """Calibration factors along the conserved directions and in their complements."""
    out = {}
    d, sem = z["data"], z["sem"]
    for v in ("X15", "Xb15"):
        pred, B = z[f"pred_{v}"], bands[f"band_{v}"]
        k = {p: m for m, p in enumerate(ansatz().h_terms)}
        th = z[f"theta_{v}"]
        J = np.array([th[k[((i, "X"), (j, "X"))]] + th[k[((i, "Y"), (j, "Y"))]] for i, j in PAIRS])
        classes = {"Z_i": (d[..., ix["Z"]] - pred[..., ix["Z"]], B[..., ix["Z"]], sem[..., ix["Z"]],
                           np.ones(N)),
                   "Z_iZ_j": (d[..., ix["ZZ"]] - pred[..., ix["ZZ"]], B[..., ix["ZZ"]], sem[..., ix["ZZ"]],
                              np.ones(len(PAIRS))),
                   "flip-flop (XX+YY)/2": (0.5 * (d[..., ix["XX"]] + d[..., ix["YY"]] - pred[..., ix["XX"]]
                                                  - pred[..., ix["YY"]]),
                                           0.5 * (B[..., ix["XX"]] + B[..., ix["YY"]]),
                                           0.5 * np.hypot(sem[..., ix["XX"]], sem[..., ix["YY"]]), J)}
        row = {}
        for name, (dev, Bc, s, u) in classes.items():
            u = u / np.linalg.norm(u)
            par, Bpar = dev @ u, Bc @ u
            spar = np.sqrt((s**2) @ u**2)
            rest, Brest = dev - par[..., None] * u, Bc - (Bc @ u)[..., None] * u
            srest = s * np.sqrt(np.clip(1.0 - u**2, 0.0, None))
            row[name] = {
                "class factor": calib(dev, Bc.std(axis=0, ddof=1), s, oos),
                "conserved direction: factor": calib(par[..., None], Bpar.std(axis=0, ddof=1)[..., None],
                                                     spar[..., None], oos),
                "rest: factor": calib(rest, Brest.std(axis=0, ddof=1), srest, oos),
                "share of the squared residual in the conserved direction (one of n directions)": [
                    float(np.mean(par[:, oos] ** 2) / np.mean(np.sum(dev[:, oos] ** 2, axis=-1))), len(u)]}
        out[v] = row
    return out


def smooth_jitter(t, y, se, deg=3):
    """Second-difference test on an evenly sampled series: var of Delta^2 y against 6 se^2 (independent errors), and
    standardised residuals from a cubic fit (outlier time points)."""
    d2 = y[2:] - 2 * y[1:-1] + y[:-2]
    exp = se[2:] ** 2 + 4 * se[1:-1] ** 2 + se[:-2] ** 2
    ratio = float(np.mean(d2**2 / exp))
    c = np.polyfit(t, y, deg, w=1 / se)
    r = (y - np.polyval(c, t)) / se
    jitter = float(np.sqrt(max(0.0, np.mean(d2**2) - np.mean(exp)) / 6.0))
    return ratio, jitter, r


def part_d(series_data, z, ix, times):
    """Mean-Z balance: the measured change against the integral of the field and dissipation terms."""
    a = ansatz()
    k = {p: m for m, p in enumerate(a.h_terms)}
    th = z["theta_X15"]
    hx = np.array([th[k[((i, "X"),)]] for i in range(N)])
    hy = np.array([th[k[((i, "Y"),)]] for i in range(N)])
    out, curves = {}, {}

    def cum(f):
        return CubicSpline(times, f, axis=0).antiderivative()(times)

    def balance(vals):
        X, Y, Zb = vals[:, ix["X"]], vals[:, ix["Y"]], vals[:, ix["Z"]].mean(1)
        return cum((2 * (Y @ hx) - 2 * (X @ hy)) / N - EMISSION_D52 * (1 + Zb)), X.mean(1), Y.mean(1), Zb

    regs, target, se_all = [], [], []
    for s in (0, 1):
        dd, Xb, Yb, Zb = balance(series_data[s])
        model, *_ = balance(z["pred_X15"][s])
        dZ = Zb - Zb[0]
        pred_change = z["pred_X15"][s][:, ix["Z"]].mean(1) - z["pred_X15"][s][0, ix["Z"]].mean()
        curves[s] = (dZ, dd, pred_change)
        out[f"initial state {s + 1}"] = {
            "measured change of mean Z at 1.5 / 5 / 10 ms": [float(dZ[j]) for j in (3, 10, 20)],
            "X15 fields on the measured <X_i>, <Y_i>, same times": [float(dd[j]) for j in (3, 10, 20)],
            "X15 prediction, same times": [float(pred_change[j]) for j in (3, 10, 20)],
            "identity check: X15 fields on the X15 trajectory minus the X15 prediction, max over t": float(
                np.max(np.abs(model - pred_change)))}
        regs.append(np.column_stack([cum(2 * Yb), cum(-2 * Xb), cum(-2 * Zb), cum(np.ones_like(Zb))]))
        target.append(dZ)
    out["_regressors"] = (regs, target)
    return out, curves, hx, hy


def fit_balance(regs, target, se):
    """Weighted least squares of the mean-Z change on subsets of (h_x, h_y, S, c), t > 0, both states."""
    A = np.vstack([r[1:] for r in regs])
    y = np.concatenate([t[1:] for t in target])
    w = 1 / np.concatenate([s[1:] for s in se])
    names = ("uniform field h_x (rad/s)", "uniform field h_y (rad/s)", "flip rate S (rad/s)", "pumping c (1/s)")
    out = {}
    for label, cols in (("fields", [0, 1]), ("flips", [2]), ("pumping", [3]), ("fields + flips + pumping", [0, 1, 2, 3])):
        p, *_ = np.linalg.lstsq(A[:, cols] * w[:, None], y * w, rcond=None)
        chi2 = float(np.sum(((A[:, cols] @ p - y) * w) ** 2))
        out[label] = {"parameters": {names[c]: float(v * RAD_S) for c, v in zip(cols, p)},
                      "chi2 / dof": [chi2, len(y) - len(cols)]}
    return out


def part_e(s1, s2, bases):
    """Shots with all ten spins equal, observed against prod_i p_i from the per-setting single-ion means."""
    out = {}
    for s, sp in enumerate((s1, s2)):
        m = sp.mean(axis=2)                                    # (times, settings, ions)
        obs_p = (sp == 1).all(axis=3).mean(axis=(1, 2))
        obs_m = (sp == -1).all(axis=3).mean(axis=(1, 2))
        exp_p = np.prod((1 + m) / 2, axis=2).mean(axis=1)
        exp_m = np.prod((1 - m) / 2, axis=2).mean(axis=1)
        out[f"initial state {s + 1}"] = {
            "all +1: observed fraction at 0 / 1.5 / 5 / 10 ms": [float(obs_p[j]) for j in (0, 3, 10, 20)],
            "all +1: independent-ion expectation, same times": [float(exp_p[j]) for j in (0, 3, 10, 20)],
            "all -1: observed fraction at 0 / 1.5 / 5 / 10 ms": [float(obs_m[j]) for j in (0, 3, 10, 20)],
            "all -1: independent-ion expectation, same times": [float(exp_m[j]) for j in (0, 3, 10, 20)],
            "observed / expected, all +1 and all -1 together, t > 0: mean, max": [
                float(np.mean((obs_p + obs_m)[1:] / (exp_p + exp_m)[1:])),
                float(np.max((obs_p + obs_m)[1:] / (exp_p + exp_m)[1:]))],
            "shots per time": int(sp.shape[1] * sp.shape[2])}
    return out


def run():
    times, s1, s2, bases = load()
    obs = observables()
    tables = per_setting_tables((s1, s2), bases, obs)
    z = np.load(RESULTS / "e11_model_selection.npz")
    bands = np.load(RESULTS / "e15_coverage.npz")
    ix = indices()
    ecols, ew = energy_weights(z["theta_Y15"])
    oos = times > T_LEARN + 1e-9
    series = series_from(tables)
    data = np.stack([as_array(series, s) for s in (0, 1)])
    assert np.allclose(data, z["data"], atol=1e-12), "the E11 data are the plain setting averages"
    cq = conserved(data, ix, ecols, ew)                                    # (state, time, 3)
    rng = np.random.default_rng(2031)
    boot = np.stack([conserved(np.stack([as_array(b, s) for s in (0, 1)]), ix, ecols, ew)
                     for b in bootstrap(tables, rng)])
    se = boot.std(axis=0, ddof=1)
    quarters = []
    for q in range(4):  # strings never measured in a quarter keep the full-data value (no drift information)
        w = np.zeros(200)
        w[50 * q:50 * (q + 1)] = 1.0
        sq = series_from(tables, w)
        quarters.append(conserved(np.stack([as_array(sq, s, fill=data[s]) for s in (0, 1)]), ix, ecols, ew))
    return dict(times=times, cq=cq, boot=boot, se=se, quarters=np.stack(quarters), data=data, z=z, bands=bands,
                ix=ix, oos=oos, shots=(s1, s2), bases=bases, series=[data[0], data[1]])


def analyse(r):
    times, cq, se, z, ix, oos = r["times"], r["cq"], r["se"], r["z"], r["ix"], r["oos"]
    ecols, ew = energy_weights(z["theta_Y15"])
    out = {"(A) neutral residual split, calibration factors t > 1.5 ms": part_a(z, r["bands"], ix, oos)}
    late = times >= 2.0 - 1e-9
    rowB, rowC = {}, {}
    for q, name in enumerate(QUANTITIES):
        for s in (0, 1):
            key = f"{name}, initial state {s + 1}"
            pm = {v: conserved(z[f"pred_{v}"][s], ix, ecols, ew)[:, q] for v in MODELS}
            change = cq[s, late, q].mean() - cq[s, 0, q]
            se_change = np.std(r["boot"][:, s, late, q].mean(axis=1) - r["boot"][:, s, 0, q], ddof=1)
            rowB[key] = {
                "data at 0 / 1.5 / 5 / 10 ms": [float(cq[s, j, q]) for j in (0, 3, 10, 20)],
                "bootstrap se at the same times": [float(se[s, j, q]) for j in (0, 3, 10, 20)],
                "data: mean over t >= 2 ms minus t = 0, and its z": [float(change), float(change / se_change)],
                **{f"{v}: mean over t >= 2 ms minus t = 0": float(pm[v][late].mean() - pm[v][0]) for v in MODELS},
                "Xb15: rms of (data - prediction) / se, t > 1.5 ms": float(
                    np.sqrt(np.mean(((cq[s, oos, q] - pm["Xb15"][oos]) / se[s, oos, q]) ** 2)))}
            ratio, jitter, res = smooth_jitter(times[1:], cq[s, 1:, q], se[s, 1:, q])
            rowC[key] = {"second differences / bootstrap expectation (1 = no jitter)": ratio,
                         "jitter sd beyond the bootstrap (same units)": jitter,
                         "time points with |standardised residual| > 3 about a cubic fit (ms)": [
                             float(t) for t, v in zip(times[1:], res) if abs(v) > 3]}
    out["(B) conserved quantities"] = rowB
    out["(C) time-point jitter, t >= 0.5 ms"] = rowC
    rowD, curves, hx, hy = part_d(r["series"], z, ix, times)
    regs, target = rowD.pop("_regressors")
    se_change = [np.hypot(se[s, :, 0], se[s, 0, 0]) for s in (0, 1)]  # the shared t = 0 value is not decorrelated
    rowD["fits of the mean-Z change, t > 0, both states"] = fit_balance(regs, target, se_change)
    rowD["X15 fields, site mean (rad/s): h_x, h_y"] = [float(hx.mean() * RAD_S), float(hy.mean() * RAD_S)]
    out["(D) mean-Z balance"] = rowD
    out["(E) collision-like shots"] = part_e(*r["shots"], r["bases"])
    qs = r["quarters"]
    rowF = {}
    for q, name in enumerate(QUANTITIES):
        for s in (0, 1):
            ch = qs[:, s, late, q].mean(axis=1) - qs[:, s, 0, q]
            se_q = np.std(r["boot"][:, s, late, q].mean(axis=1) - r["boot"][:, s, 0, q], ddof=1) * 2.0
            chi2 = float(np.sum(((ch - ch.mean()) / se_q) ** 2))
            rowF[f"{name}, initial state {s + 1}"] = {
                "change (t >= 2 ms minus t = 0) per quarter of the settings": ch.tolist(),
                "chi2 of homogeneity (3 dof; se of a quarter = 2 x full-data se)": chi2}
    out["(F) drift over the acquisition"] = rowF
    return out, curves


def plot(r, out, curves):
    plt = style()
    times, cq, se, z, ix = r["times"], r["cq"], r["se"], r["z"], r["ix"]
    ecols, ew = energy_weights(z["theta_Y15"])
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.6))
    for q, name in enumerate(QUANTITIES):
        for s in (0, 1):
            ax = axes[s, q]
            ax.axvspan(0.0, T_LEARN, color=MUTED, alpha=0.08, lw=0)
            ax.errorbar(times, cq[s, :, q] - cq[s, 0, q], yerr=Z95 * se[s, :, q], fmt="o", ms=3.5, color=INK, lw=0.8,
                        capsize=0, label="BEQS data")
            for v, c, ls in (("Y15", MUTED, ":"), ("X15", CAT[0], "-"), ("Xb15", CAT[1], "--")):
                pm = conserved(z[f"pred_{v}"][s], ix, ecols, ew)[:, q]
                ax.plot(times, pm - pm[0], color=c, ls=ls, lw=1.4, label=v)
            if q == 0:
                ax.plot(times, curves[s][1], color=CAT[2], lw=1.4, label="X15 fields on measured <X_i>, <Y_i>")
            ax.set_title(f"({'abcdef'[3 * s + q]}) {name}, initial state {s + 1}", fontsize=9)
            ax.set_xlabel("t (ms)")
            ax.set_ylabel("change since t = 0")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("E17: conserved quantities of the learned models in the BEQS data (measured by Kraft et al.)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return save_figure(fig, "e17_neutral_diagnosis")


if __name__ == "__main__":
    r = run()
    np.savez_compressed(RESULTS / "e17_neutral_diagnosis.npz", times=r["times"], cq=r["cq"], boot=r["boot"],
                        se=r["se"], quarters=r["quarters"])
    out, curves = analyse(r)
    save_json(RESULTS / "e17_neutral_diagnosis.json", out)
    print(plot(r, out, curves))
    for key, v in out.items():
        print(f"{key}: {v}")
