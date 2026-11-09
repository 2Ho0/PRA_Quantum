"""E23d: is the neutral late-time phase error a bias of the learning estimator? (diagnostic, saved results only)

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0), analysed
here from the public raw shots. Question Q27: how can the failed checks (E20, E22b, E23c, E23s) come out better, and
does a formula have to change? E23 traced the neutral error of Xb15uf to late-time phase errors of the coherent
oscillations with the right amplitude. This script asks two questions with the saved results only (no new learning):

(A) How large may a calibration factor be by chance? Null distribution if the Xb15 band were exactly right: truth =
    central Xb15 prediction + a draw from the covariance of the 12 bootstrap replicates of the band + shot noise
    (Gaussian, measured standard errors); the factor is computed as in E20. Reported: quantiles, the probability that
    all three neutral classes stay <= 1.2 (the E20 rule), p-values of the observed Xb15uf factors, and the jackknife
    error of the factors over the 12 replicates.
(B) Is the phase error a coupling-scale error? Xb15uf with every two-body coupling multiplied by a FIXED lambda (the
    E23c grid, cubic interpolation in lambda, no averaging, so no smoothing). The lambda the late data prefer for each
    state is an oracle (it uses t > 1.5 ms) and serves as a diagnostic only.
(C) A-priori scale corrections, from the learning window only. The integral method regresses on measured
    correlators, so its regressors carry shot noise (errors in variables) and least squares shrinks the couplings.
    Three independent sizes of that shrinkage: the bootstrap ratio mean(J*)/J of Xb15 (100 replicates, E11; corrected
    scale 2 - ratio), the simulated attenuation of E11b (independent Gaussian noise) and of E14 (shot-level noise,
    five noise mixtures). Factors at these lambdas.
(D) Late oscillation amplitude against lambda (temporal variance on t >= 4 ms): a fixed scale must not damp.
(E) Other estimator biases seen in the bootstrap (J by distance, gamma_c, beta), the prediction-level bootstrap bias
    correction 2 central - mean(replicates) of Xb15, and whether the current terms (XY - YX) form a pure gauge
    pattern J_ij exp(i(phi_i - phi_j)).
Status: diagnostic after the data were seen. Only (C) is free of the late data; it is a candidate for a preregistered
re-run with an errors-in-variables-consistent estimator, not a verdict.
"""

import json

import numpy as np
from scipy.interpolate import CubicSpline

from common import CAT, INK, MUTED, RESULTS, save_figure, save_json, style
from e6_ion_learning import PAIRS, observables
from e6c_validation import charge_classes

NEUTRAL = ("q=0  Z_i", "q=0  Z_iZ_j", "q=0  flip-flop")
ALL = "all 435 observables"
T_LEARN = 1.5
N_NULL = 4000
SEED = 0


def class_matrices(obs):
    """{class: (n_obs, n_combos) coefficient matrix}, so that combine(values) = values @ C."""
    cls = dict(charge_classes(obs), **{ALL: [{m: 1.0} for m in range(len(obs))]})
    out = {}
    for name, coeffs in cls.items():
        C = np.zeros((len(obs), len(coeffs)))
        for c, co in enumerate(coeffs):
            for m, v in co.items():
                C[m, c] = v
        out[name] = C
    return out


def spread2(band, sem, C):
    """sd_band^2 + sem^2 of every combination of one class (state, time, combination)."""
    return (band @ C).std(axis=0, ddof=1) ** 2 + (sem**2) @ (C**2)


def factor(pred, data, sem, band, C, oos, tot2=None):
    """The E20 calibration factor of one class: rms(data - pred) / rms(sqrt(sd_band^2 + sem^2)) on t > 1.5 ms."""
    tot2 = spread2(band, sem, C) if tot2 is None else tot2
    dev = (data - pred) @ C
    return float(np.sqrt(np.mean(dev[:, oos] ** 2) / np.mean(tot2[:, oos])))


def couplings(theta):
    """(flip-flop J_ij, current D_ij) of the 45 pairs from the first 180 entries (XX, YY, XY, YX per pair)."""
    t = np.asarray(theta)[..., :180].reshape(*np.shape(theta)[:-1], 45, 4)
    return t[..., 0] + t[..., 1], t[..., 2] - t[..., 3]


def run():
    z = np.load(RESULTS / "e11_model_selection.npz")
    e20 = np.load(RESULTS / "e20_flip_models.npz")
    band = np.load(RESULTS / "e15_coverage.npz")["band_Xb15"]
    g = np.load(RESULTS / "e23c_coupling_noise.npz")
    grid, lams = g["grid"], g["lambdas"]
    e11b = json.loads((RESULTS / "e11b_calibration.json").read_text())
    e14 = json.loads((RESULTS / "e14_shot_calibration.json").read_text())
    times, data, sem = z["times"], z["data"], z["sem"]
    oos = times > T_LEARN + 1e-9
    late = times >= 4.0 - 1e-9
    Cs = class_matrices(observables())
    names = list(Cs)
    T2 = {n: spread2(band, sem, Cs[n]) for n in names}
    fac = lambda pred, dat=data: {n: factor(pred, dat, sem, band, Cs[n], oos, T2[n]) for n in names}
    out, arrays = {}, {}

    # (A) null distribution of the factor
    observed = fac(e20["pred_Xb15uf"])
    cen = z["pred_Xb15"]
    dev_b = band - band.mean(axis=0)
    B = band.shape[0]
    rng = np.random.default_rng(SEED)
    null = {n: np.empty(N_NULL) for n in names}
    for r in range(N_NULL):
        truth = cen + np.tensordot(rng.normal(size=B) / np.sqrt(B - 1), dev_b, axes=1) + rng.normal(size=sem.shape) * sem
        for n in names:
            null[n][r] = factor(cen, truth, sem, band, Cs[n], oos, T2[n])
    jack = {n: np.array([factor(e20["pred_Xb15uf"], data, sem, np.delete(band, b, axis=0), Cs[n], oos)
                         for b in range(B)]) for n in names}
    out["(A) null distribution of the factor if the Xb15 band were exactly right"] = {
        n: {"median": float(np.median(null[n])), "90%": float(np.quantile(null[n], 0.9)),
            "95%": float(np.quantile(null[n], 0.95)), "observed Xb15uf": observed[n],
            "p = P(null >= observed)": float(np.mean(null[n] >= observed[n])),
            "jackknife SE of the observed factor (12 replicates)": float(
                np.sqrt((B - 1) / B * np.sum((jack[n] - jack[n].mean()) ** 2)))} for n in names}
    out["(A) P(all three neutral classes <= 1.2 | band exactly right)"] = float(
        np.mean(np.all([null[n] <= 1.2 for n in NEUTRAL], axis=0)))
    out["(A) P(all neutral <= 1.2 and all 435 <= 1.15 | band exactly right), the E20 rule"] = float(
        np.mean(np.all([null[n] <= 1.2 for n in NEUTRAL] + [null[ALL] <= 1.15], axis=0)))
    arrays.update({f"null {n}": null[n] for n in names})

    # (B) fixed coupling scale
    spl = CubicSpline(lams, grid, axis=0)
    fine = np.round(np.arange(0.94, 1.16 + 1e-9, 0.0025), 4)
    scan = np.array([[fac(spl(l))[n] for n in names] for l in fine])
    arrays.update(scan_lambda=fine, scan_factors=scan)

    def neutral_obj(pred, s):  # the E23c calibration target: sum of the squared neutral factors of one state
        return sum(np.mean(((data - pred) @ Cs[n])[s][oos] ** 2) / np.mean(T2[n][s][oos]) for n in NEUTRAL)

    lam_fine = np.linspace(0.9, 1.2, 1201)
    star = {}
    for s in (0, 1):
        vals = np.array([neutral_obj(spl(l), s) for l in lam_fine])
        star[s] = float(lam_fine[np.argmin(vals)])
    own = np.stack([spl(star[s])[s] for s in (0, 1)])
    cross = np.stack([spl(star[1 - s])[s] for s in (0, 1)])
    out["(B) lambda preferred by the late data (oracle, diagnostic only): state 1, state 2"] = [star[0], star[1]]
    out["(B) factors"] = {"lambda = 1 (Xb15uf)": observed, "own late-data lambda (in-sample)": fac(own),
                          "other state's late-data lambda": fac(cross)}

    # (C) a-priori scale corrections
    Jc, _ = couplings(z["theta_Xb15"])
    Jb, _ = couplings(z["boot_theta_Xb15"])
    ratio = float(Jb.mean(axis=0) @ Jc / (Jc @ Jc))
    truth = e14["truth J_nn (rad/s)"]
    shot = [row[0] for row in e14["J_nn mean per f and window"]]
    prior = {"bootstrap of Xb15: 2 - mean(J*)/J": 2.0 - ratio,
             "E11b, independent Gaussian noise, white truth": e11b["truth J_nn (rad/s)"]
             / e11b["white truth, noise x1.0: J_nn mean per window"][0],
             "E11b, independent Gaussian noise, static truth": e11b["truth J_nn (rad/s)"]
             / e11b["static truth, noise x1.0: J_nn mean per window"][0],
             "E14, shot-level noise, mean over the five mixtures": float(np.mean([truth / v for v in shot])),
             "E14, shot-level noise, range over the five mixtures": [float(min(truth / v for v in shot)),
                                                                     float(max(truth / v for v in shot))]}
    out["(C) a-priori coupling-scale corrections (learning window only)"] = prior
    out["(C) E11b check of the errors-in-variables signature: J_nn shrinkage at noise x0.25 / x1.0 (white truth)"] = [
        1 - e11b["white truth, noise x0.25: J_nn mean per window"][0] / e11b["truth J_nn (rad/s)"],
        1 - e11b["white truth, noise x1.0: J_nn mean per window"][0] / e11b["truth J_nn (rad/s)"]]
    lam_prior = prior["E14, shot-level noise, mean over the five mixtures"]
    out["(C) factors at the a-priori corrections"] = {
        f"{k}: lambda {v:.4f}": fac(spl(v)) for k, v in prior.items() if not isinstance(v, list)}
    p_corr = {n: float(np.mean(null[n] >= fac(spl(lam_prior))[n])) for n in names}
    out["(C) p-values of the factors at the E14 mean correction under the null of (A)"] = p_corr

    # (D) late oscillation amplitude
    amp = {}
    for n in NEUTRAL:
        C = Cs[n]
        for s in (0, 1):
            d = (data @ C)[s][late]
            dv = float(np.mean(d.var(axis=0, ddof=1) - np.mean(((sem**2) @ (C**2))[s][late], axis=0)))
            amp[f"{n}, state {s + 1}"] = {"data (shot noise removed)": dv, **{
                f"lambda {l:.3f}": float(np.mean((spl(l) @ C)[s][late].var(axis=0, ddof=1)))
                for l in (1.0, lam_prior, star[0], star[1])}}
    out["(D) temporal variance on t >= 4 ms (late oscillation amplitude squared)"] = amp

    # (E) other estimator biases and checks
    dist = np.array([abs(i - j) for i, j in PAIRS])
    k0 = 210  # gamma_c follows the 180 two-body and 30 one-body Hamiltonian terms
    bt = z["boot_theta_Xb15"]
    out["(E) bootstrap mean / central of the flip-flop couplings by distance"] = {
        str(d): float(Jb.mean(axis=0)[dist == d] @ Jc[dist == d] / (Jc[dist == d] @ Jc[dist == d])) for d in range(1, 10)}
    out["(E) gamma_c (rad/s): central, bootstrap mean, bootstrap sd"] = [
        float(z["theta_Xb15"][k0] * 1e3), float(bt[:, k0].mean() * 1e3), float(bt[:, k0].std() * 1e3)]
    out["(E) beta (rad/s per ms): central, bootstrap mean, bootstrap sd"] = [
        float(z["beta_Xb15"] * 1e3), float(z["boot_beta_Xb15"].mean() * 1e3), float(z["boot_beta_Xb15"].std() * 1e3)]
    out["(E) prediction-level bootstrap bias correction of Xb15 (2 central - mean of the 12 replicates)"] = {
        "Xb15 central": fac(cen), "corrected": fac(2 * cen - band.mean(axis=0))}
    J, D = couplings(z["theta_Xb15"])
    Jbt, Dbt = couplings(bt)
    ph, sdph = np.arctan2(D, J), np.arctan2(Dbt, Jbt).std(axis=0)
    sel = [p for p, (i, j) in enumerate(PAIRS) if j - i <= 3]
    A = np.zeros((len(sel), 9))
    for r, p in enumerate(sel):
        i, j = PAIRS[p]
        if i > 0:
            A[r, i - 1] = 1.0
        A[r, j - 1] -= 1.0
    w = 1.0 / np.maximum(sdph[sel], 1e-3)
    sol = np.linalg.lstsq(A * w[:, None], ph[sel] * w, rcond=None)[0]
    chi2 = float(np.sum(((A @ sol - ph[sel]) * w) ** 2))
    out["(E) pure-gauge currents: chi2, dof of phase_ij = phi_i - phi_j on pairs up to distance 3"] = [chi2, len(sel) - 9]
    def z_time(pred):  # pooled neutral z-score per time: sqrt(sum_n mean dev^2 / sum_n mean spread^2)
        num = sum(np.mean(((data - pred) @ Cs[n]) ** 2, axis=(0, 2)) for n in NEUTRAL)
        den = sum(np.mean(T2[n], axis=(0, 2)) for n in NEUTRAL)
        return np.sqrt(num / den)

    arrays.update(own=own, cross=cross, prior_pred=spl(lam_prior), z_as_learned=z_time(e20["pred_Xb15uf"]),
                  z_prior=z_time(spl(lam_prior)), z_own=z_time(own))
    out["(B) pooled neutral z-score per time: as learned / a priori lambda / own late-data lambda"] = {
        f"{t:.1f}": [float(arrays[k][m]) for k in ("z_as_learned", "z_prior", "z_own")] for m, t in enumerate(times)}
    return out, arrays, spl, star, lam_prior


def plot(out, arrays, spl, star, lam_prior):
    plt = style()
    z = np.load(RESULTS / "e11_model_selection.npz")
    times = z["times"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ax = axes[0]
    names = list(NEUTRAL) + [ALL]
    lam, scan = arrays["scan_lambda"], arrays["scan_factors"]
    order = list(out["(B) factors"]["lambda = 1 (Xb15uf)"])
    for n, c, ls in zip(names, [CAT[0], CAT[1], CAT[2], MUTED], ["-", "-", "-", "--"]):
        ax.plot(lam, scan[:, order.index(n)], ls, color=c, lw=1.6, label=n.replace("q=0  ", "q=0 ").replace("all 435 observables", "all 435"))
    prior = out["(C) a-priori coupling-scale corrections (learning window only)"]
    vals = [x for v in prior.values() for x in (v if isinstance(v, list) else [v])]
    ax.axvspan(min(vals), max(vals), color=CAT[0], alpha=0.10, lw=0)
    for s, st in ((0, ":"), (1, "-.")):
        ax.axvline(star[s], color=INK, lw=0.9, ls=st)
    ax.axhline(1.2, color=MUTED, lw=0.8, ls="--")
    ax.axhline(1.0, color=MUTED, lw=0.6, ls=":")
    ax.set_xlabel("fixed scale $\\lambda$ of all two-body couplings")
    ax.set_ylabel("calibration factor, t > 1.5 ms")
    ax.text(lam_prior, 1.47, "a priori\n(learning window)", fontsize=6.5, ha="center", va="top", color=INK)
    ax.text(star[1] + 0.002, 1.47, "late data (oracle)\nstate 1 dotted,\nstate 2 dash-dot", fontsize=6.5, ha="left",
            va="top", color=INK)
    ax.set_ylim(0.85, 1.5)
    ax.legend(fontsize=6.5, loc="lower left")
    ax.set_title("(a) factors against a fixed coupling scale", fontsize=9)
    ax = axes[1]
    A = out["(A) null distribution of the factor if the Xb15 band were exactly right"]
    corr = out["(C) factors at the a-priori corrections"][
        f"E14, shot-level noise, mean over the five mixtures: lambda {lam_prior:.4f}"]
    for y, n in enumerate(names):
        v = arrays[f"null {n}"]
        lo, hi = np.quantile(v, [0.05, 0.95])
        ax.plot([lo, hi], [y, y], "-", color=MUTED, lw=4, alpha=0.45, solid_capstyle="butt")
        ax.plot(np.median(v), y, "|", color=MUTED, ms=10, mew=1.5)
        ax.plot(A[n]["observed Xb15uf"], y - 0.12, "D", color=INK, ms=5, label="Xb15uf as learned" if y == 0 else None)
        ax.plot(corr[n], y + 0.12, "o", color=CAT[0], ms=5, label=f"$\\lambda$ = {lam_prior:.3f} (a priori)" if y == 0 else None)
    ax.axvline(1.2, color=MUTED, lw=0.8, ls="--")
    ax.set_yticks(range(len(names)), [n.replace("all 435 observables", "all 435") for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("calibration factor, t > 1.5 ms")
    ax.legend(fontsize=6.5, loc="lower right", title="grey: 5-95% if the band were exact", title_fontsize=6.5)
    ax.set_title("(b) chance spread of the factor", fontsize=9)
    ax = axes[2]
    ax.plot(times, arrays["z_as_learned"], "-o", color=INK, lw=1.4, ms=3.5, label="$\\lambda$ = 1 (Xb15uf as learned)")
    ax.plot(times, arrays["z_prior"], "-o", color=CAT[0], lw=1.4, ms=3.5, label=f"$\\lambda$ = {lam_prior:.3f} (a priori)")
    ax.plot(times, arrays["z_own"], "--", color=CAT[1], lw=1.4,
            label=f"$\\lambda$ = {star[0]:.3f} / {star[1]:.3f} (late data, in-sample)")
    ax.axvspan(0.0, T_LEARN, color=MUTED, alpha=0.08, lw=0)
    ax.axhline(1.0, color=MUTED, lw=0.6, ls=":")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("neutral z-score (rms deviation / rms spread)")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("(c) neutral error against time, both states", fontsize=9)
    fig.suptitle("E23d: estimator bias of the coupling scale and the chance spread of the factor (data: Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e23d_scale_bias")


if __name__ == "__main__":
    out, arrays, spl, star, lam_prior = run()
    save_json(RESULTS / "e23d_scale_bias.json", out)
    np.savez_compressed(RESULTS / "e23d_scale_bias.npz", **{k.replace(" ", "_"): v for k, v in arrays.items()})
    print(plot(out, arrays, spl, star, lam_prior))
    for key, v in out.items():
        print(f"{key}: {v}")
