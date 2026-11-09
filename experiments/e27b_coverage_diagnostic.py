"""E27b: why rule S'' of E27 failed in the Z_iZ_j class (post hoc diagnostic, not preregistered)

Data: the synthetic control truth of E11f (Y15 couplings, currents and z fields on the measured initial states of
Kraft et al.'s data, PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0) and the saved fits and
predictions of E27 steps A and B. No data of the late record are used beyond the step-A band that E27 already made.
Question (after E27): in the data-matched drift world the pooled factor of S_D against its step-A band is 0.79 (Z_i),
1.31 (Z_iZ_j) and 0.54 (flip-flop); rule S'' needs <= 1.18, 1.17, 1.21. The band is too wide in two neutral classes and
too narrow in one. Which parameter group carries the Z_iZ_j bias, and is the pattern what one draw of the truth's
currents from the band's own current prior would give?
Method (no new fits; exact propagation of U(1)-symmetric Hamiltonians in the magnetisation sectors, which is all the
neutral classes need, since they do not see the collective noise):
  hybrids     the truth with one group replaced by its mean over the 8 realisations of step B: currents zero,
              couplings of S, couplings of S_D, z fields of S_D, and the symmetric part of S_D (all three of currents
              zero, couplings of S_D, z fields of S_D). Route check: the truth itself against the saved exact truth.
  one draw    the truth's couplings and z fields with currents D ~ N(0, tau^2) per pair, tau from the realisations of
              step B (mean tau per distance group), K draws. Reported: the rms of these predictions per class against
              the step-A band (how much of the band is the currents), and, per class, the quantile of the truth's own
              "currents zero" bias in the distribution of the same quantity over the prior draws (pred(D = 0) against
              pred(D_k), relative to the currents-only spread). A quantile near 0 or 1 means the truth's currents are
              an atypical draw for that class.
  single-draw null  the measured bias of S_D (mean over the 8 realisations) per class against the null of a single
              draw from the step-A band (E25's null_pooled with one draw), since the 8 realisations share one truth.
Reported only; nothing here changes the verdict of E27.
Run: `python e27b_coverage_diagnostic.py`.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from common import CAT, INK, MUTED, RESULTS, save_figure, save_json, style  # noqa: E402
from e6_ion_learning import N, PAIRS, ansatz, observables, per_setting_tables, product_state, series_from  # noqa: E402
from e6c_validation import load  # noqa: E402
from e11f_nonmarkov_truths import Propagator, hamiltonian, sectors  # noqa: E402
from e14_shot_calibration import truth_theta  # noqa: E402
from e24_eiv_correction import NEUTRAL, T_LEARN  # noqa: E402
from e25_structured_eiv import class_matrices, indices, null_pooled  # noqa: E402
from qerr import spins  # noqa: E402
from qerr.pauli import expectation_tables, expectations  # noqa: E402

K_DRAWS, SEED, WORKERS = 200, 2737, int(os.environ.get("E27B_WORKERS", "4"))
K0 = len(ansatz().h_terms)
HYBRIDS = ("truth", "currents zero", "couplings S", "couplings S_D", "z fields S_D", "symmetric part of S_D")
_W = {}


def _init():
    times, s1, s2, bases = load()
    series = series_from(per_setting_tables((s1, s2), bases, [((i, a),) for i in range(N) for a in "XYZ"]))
    _W.update(times=times, rho=[product_state(series, s) for s in (0, 1)], tab=expectation_tables(observables(), N),
              blocks=sectors())


def neutral_prediction(theta_h):
    """(2, 21, 435) expectations of a U(1)-symmetric Hamiltonian (no noise: the neutral classes do not see it)."""
    if not _W:
        _init()
    th = np.zeros(ansatz().n_params)
    th[:K0] = np.asarray(theta_h, float)[:K0]
    H = hamiltonian(th).toarray()
    m = spins.total_sz(N)
    assert float(np.abs(H * (m[None, :] - m[:, None])).max()) < 1e-9, "the Hamiltonian is not U(1)-symmetric"
    idx, ph = _W["tab"]
    out = []
    for rho in _W["rho"]:
        P = Propagator(H, rho, _W["blocks"])
        out.append(np.array([expectations(P(t), idx, ph) for t in _W["times"]]))
    return np.array(out)


def groups():
    """Index arrays of the U(1)-symmetric groups: flip-flop pairs (XX, YY), currents (XY, YX), z fields."""
    ff, cu, _ = indices()
    k = {p: m for m, p in enumerate(ansatz().h_terms)}
    zf = np.array([k[((i, "Z"),)] for i in range(N)])
    return ff, cu, zf


def hybrid_thetas(b):
    tr = truth_theta()[:K0].copy()
    ff, cu, zf = groups()
    ths, thd = b["dm_theta_s"][:, :K0].mean(axis=0), b["dm_theta_d"][:, :K0].mean(axis=0)
    out = {"truth": tr.copy()}
    h = tr.copy()
    h[cu] = 0.0
    out["currents zero"] = h
    for name, src in (("couplings S", ths), ("couplings S_D", thd)):
        h = tr.copy()
        h[ff] = src[ff]
        out[name] = h
    h = tr.copy()
    h[zf] = thd[zf]
    out["z fields S_D"] = h
    h = tr.copy()
    h[cu], h[ff], h[zf] = 0.0, thd[ff], thd[zf]
    out["symmetric part of S_D"] = h
    return out


def draw_thetas(b, rng):
    """The truth with its currents replaced by K_DRAWS draws from the mean step-B current prior (and D = 0 first)."""
    tr = truth_theta()[:K0]
    _, cu, _ = groups()
    tau_g = b["dm_tau_g"].mean(axis=0)
    dist = np.array([j - i for i, j in PAIRS])
    tau = tau_g[np.where(dist == 1, 0, np.where(dist == 2, 1, 2))]
    out = []
    for D in rng.normal(size=(K_DRAWS, len(PAIRS))) * tau:
        h = tr.copy()
        h[cu[:, 0]], h[cu[:, 1]] = 0.5 * D, -0.5 * D
        out.append(h)
    return out, tau_g


def run():
    a, b = np.load(RESULTS / "e27_stepA.npz"), np.load(RESULTS / "e27_stepB.npz")
    saved = RESULTS / "e27b_coverage_diagnostic.npz"
    if saved.exists():
        z = np.load(saved)
        return a, b, {h: z[f"hybrid_{h.replace(' ', '_')}"] for h in HYBRIDS}, z["draws"], z["tau_g"]
    hy = hybrid_thetas(b)
    draws, tau_g = draw_thetas(b, np.random.default_rng(SEED))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init) as pool:
        preds = list(pool.map(neutral_prediction, list(hy.values()) + draws, chunksize=4))
    print(f"{len(preds)} sector predictions: {time.time() - t0:.0f} s", flush=True)
    P_h = dict(zip(hy, preds[:len(hy)]))
    P_d = np.array(preds[len(hy):])
    np.savez_compressed(RESULTS / "e27b_coverage_diagnostic.npz", **{f"hybrid_{k.replace(' ', '_')}": v
                                                                     for k, v in P_h.items()},
                        draws=P_d, tau_g=tau_g)
    return a, b, P_h, P_d, tau_g


def analyse(a, b, P_h, P_d, tau_g):
    times = load()[0]
    oos = times > T_LEARN + 1e-9
    Cs = class_matrices()
    band, truth, pred = a["band"], b["truth"], b["pred"]
    classes = list(NEUTRAL)

    def band_sd2(n):
        return float(np.mean(((band @ Cs[n]).std(axis=0, ddof=1) ** 2)[:, oos]))

    def ratio(dev, n, ref=None):
        return float(np.sqrt(np.mean(((dev @ Cs[n])[:, oos]) ** 2) / (band_sd2(n) if ref is None else ref)))

    out = {"route check: truth by sector propagation against the saved exact truth, max |difference| (neutral classes)":
           float(max(np.abs((P_h["truth"] - truth) @ Cs[n]).max() for n in classes))}
    bias = pred.mean(axis=0) - truth
    out["measured bias of S_D, mean of 8 (bias rms / step-A band rms)"] = {n: ratio(bias, n) for n in classes}
    out["hybrids: bias rms / step-A band rms, t > 1.5 ms"] = {
        h: {n: ratio(P_h[h] - truth, n) for n in classes} for h in HYBRIDS[1:]}
    fit = {}
    for n in classes:
        m = (bias @ Cs[n])[:, oos].ravel()
        h = ((P_h["symmetric part of S_D"] - truth) @ Cs[n])[:, oos].ravel()
        fit[n] = float(np.sqrt(np.mean((m - h) ** 2) / np.mean(m ** 2)))
    out["rms of (measured bias - symmetric-part hybrid bias) over rms of the measured bias"] = fit
    # the currents-only band and where the truth's currents sit in it
    cur_sd2 = {n: float(np.mean(((P_d @ Cs[n]).std(axis=0, ddof=1) ** 2)[:, oos])) for n in classes}
    out["current prior tau (rad/s), mean over the step-B realisations"] = (tau_g * 1e3).tolist()
    out["currents-only spread over the step-A band (rms ratio)"] = {n: float(np.sqrt(cur_sd2[n] / band_sd2(n)))
                                                                   for n in classes}
    q, val = {}, {}
    for n in classes:
        zero = P_h["currents zero"]
        dist = np.array([ratio(zero - P, n, cur_sd2[n]) for P in P_d])
        v = ratio(zero - P_h["truth"], n, cur_sd2[n])
        q[n] = float(np.mean(dist <= v))
        val[n] = [v, float(np.quantile(dist, 0.025)), float(np.median(dist)), float(np.quantile(dist, 0.975))]
    out["'currents zero' bias over the currents-only spread: truth; 2.5%, 50%, 97.5% over prior draws"] = val
    out["quantile of the truth's value among the prior draws"] = q
    rng = np.random.default_rng(SEED + 1)
    single = {n: null_pooled(band, Cs[n], oos, 1, rng=rng) for n in classes}
    out["single-draw null of bias rms / band rms: 2.5%, 50%, 97.5%; quantile of the measured bias"] = {
        n: [float(np.quantile(single[n], 0.025)), float(np.median(single[n])), float(np.quantile(single[n], 0.975)),
            float(np.mean(single[n] <= out["measured bias of S_D, mean of 8 (bias rms / step-A band rms)"][n]))]
        for n in classes}
    return out, single


def plot(out, single):
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    ax = axes[0]
    rows = list(out["hybrids: bias rms / step-A band rms, t > 1.5 ms"])
    classes = list(NEUTRAL)
    for c, n in enumerate(classes):
        vals = [out["hybrids: bias rms / step-A band rms, t > 1.5 ms"][h][n] for h in rows]
        ax.plot(vals, np.arange(len(rows)) + 0.15 * (c - 1), "o", color=CAT[c], ms=5, label=n)
        ax.plot(out["measured bias of S_D, mean of 8 (bias rms / step-A band rms)"][n], len(rows) + 0.15 * (c - 1),
                "D", color=CAT[c], ms=5)
    ax.set_yticks(range(len(rows) + 1), rows + ["measured S_D (mean of 8)"])
    ax.invert_yaxis()
    ax.axvline(1.0, color=MUTED, lw=0.6, ls=":")
    ax.set_xlabel("bias rms / step-A band rms, t > 1.5 ms")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.set_title("(a) which group carries the bias (hybrids of the truth)", fontsize=9)
    ax = axes[1]
    for c, n in enumerate(classes):
        lo, med, hi, _ = out["single-draw null of bias rms / band rms: 2.5%, 50%, 97.5%; quantile of the measured bias"][n]
        ax.plot([lo, hi], [c, c], "-", color=MUTED, lw=4, alpha=0.45, solid_capstyle="butt")
        ax.plot(med, c, "|", color=INK, ms=8)
        ax.plot(out["measured bias of S_D, mean of 8 (bias rms / step-A band rms)"][n], c, "D", color=CAT[c], ms=6)
    ax.set_yticks(range(len(classes)), classes)
    ax.invert_yaxis()
    ax.set_xlabel("bias rms / band rms (grey: 2.5-97.5%, one band draw)")
    ax.set_title("(b) the common bias of the 8 realisations as one draw", fontsize=9)
    fig.suptitle("E27b (post hoc): the Z_iZ_j failure of rule S'' (synthetic; initial states from Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e27b_coverage_diagnostic")


if __name__ == "__main__":
    a, b, P_h, P_d, tau_g = run()
    out, single = analyse(a, b, P_h, P_d, tau_g)
    save_json(RESULTS / "e27b_coverage_diagnostic.json", out)
    print(plot(out, single))
    for key, val in out.items():
        print(key, val, flush=True)
