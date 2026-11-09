"""E27c: how much does the setting drift shrink the learned couplings? (post hoc diagnostic, not preregistered)

Data: synthetic only. The worlds of E27 (the E11f control truth sampled on the measured initial states of Kraft et
al.'s data, PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0; 200 settings x 200 shots).
Question (after E27): the drift's extra coupling shrinkage disagrees between runs. E26 part B (all of beta_s as drift,
sigma_d = 276 rad/s, 8 realisations) gave a J_nn error of S of -3.9 +- 1.2% against -0.2 +- 0.7% without drift; the
E27 pre-registration diagnostic (276 rad/s, 3 realisations) a coupling scale of 0.967 +- 0.018 against 1.010 +- 0.004;
E27 step B (217 rad/s, 8 realisations) 1.014 +- 0.013 against 0.995 +- 0.002 in step C. Scaled by sigma_d^2, the first
two predict about -2.4% at 217 rad/s; step B shows +1.9%.
Method: one code path, three drift levels sigma_d = 0, 216.9 (the data's estimate) and 276.4 rad/s (all of beta_s),
16 realisations each, the rest of beta_s within every setting (control_world of E27). Per realisation: least squares
and S (20 chains, oracle defect of the common setting average), as in E27 steps B and C but without the S_D step.
Reported: coupling scale (sum J J_T / sum J_T^2) and J_nn error of least squares and S per level (mean, se), their
change against sigma_d = 0, and a weighted fit of the change against sigma_d^2 through zero, read at 216.9 rad/s.
Nothing here changes the verdicts of E26 or E27.
Run: `python e27c_drift_shrinkage.py` (workers: E27C_WORKERS, default 10).
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from common import CAT, INK, MUTED, RESULTS, save_figure, save_json, style  # noqa: E402
from e6c_validation import load  # noqa: E402
from e24_eiv_correction import N_WIN, T_LEARN, strings_list  # noqa: E402
from e25_structured_eiv import indices, to_theta  # noqa: E402
from e27_marginal_band import (CHAINS_W, K0, SIGMA_REF, control_world, coupling_scale, data_excess,  # noqa: E402
                               drift_sigma, excess_w1, fit_s, jnn_error)

N_REAL, SEED0 = 16, 273000
WORKERS = int(os.environ.get("E27C_WORKERS", "10"))
PARTS = RESULTS / "e27c_parts"
LEVELS = (0.0, drift_sigma(data_excess()), SIGMA_REF)  # rad/ms
_W = {}


def _init():
    times, _, _, bases = load()
    tw, strings = times[:N_WIN], strings_list()
    d, dt = control_world(0.0, tw).defects(strings, T_LEARN)  # the setting average is common to the three levels
    _W.update(tw=tw, strings=strings, bases=bases, d=d, dt=dt)


def job(args):
    level, r = args
    PARTS.mkdir(exist_ok=True)
    path = PARTS / f"c_{level}_{r}.npz"
    if path.exists():
        return dict(np.load(path))
    t0 = time.time()
    truth = control_world(LEVELS[level], _W["tw"])
    rng = np.random.default_rng(SEED0 + 7907 * r + 1000 * level)
    shots = truth.shots(rng, _W["bases"])
    th_s, b_s, lam, p0 = fit_s(shots, _W["d"], _W["dt"], rng, CHAINS_W, _W["bases"], _W["strings"], _W["tw"])
    th0, b0 = to_theta(p0, 0.0)
    ff, _, _ = indices()
    out = dict(scale=np.array([coupling_scale(th0), coupling_scale(th_s)]),
               jnn=np.array([jnn_error(th0), jnn_error(th_s)]), beta=np.array([b0, b_s]),
               gamma=np.array([th0[K0], th_s[K0]]), lam=lam, J=np.array([th0[ff].sum(axis=1), th_s[ff].sum(axis=1)]),
               sigma_hat=drift_sigma(excess_w1(shots, _W["bases"], _W["strings"])))
    np.savez_compressed(path, **out)
    print(f"level {level} realisation {r}: {time.time() - t0:.0f} s", flush=True)
    return out


def run():
    jobs = [(lv, r) for r in range(N_REAL) for lv in range(len(LEVELS))]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init) as pool:
        res = list(pool.map(job, jobs))
    print(f"{len(jobs)} fits: {time.time() - t0:.0f} s", flush=True)
    get = lambda key: np.array([[res[jobs.index((lv, r))][key] for r in range(N_REAL)] for lv in range(len(LEVELS))])
    z = {k: get(k) for k in ("scale", "jnn", "beta", "gamma", "lam", "sigma_hat")}
    np.savez_compressed(RESULTS / "e27c_drift_shrinkage.npz", levels=np.array(LEVELS), **z)
    return z


def analyse(z):
    lv = np.array(LEVELS)
    out = {"drift levels sigma_d (rad/s)": (lv * 1e3).tolist(), "realisations per level": N_REAL,
           "sigma_d estimated (rad/s): mean, sd per level": [[float(s.mean()) * 1e3, float(s.std(ddof=1)) * 1e3]
                                                              for s in z["sigma_hat"]]}
    fits = {}
    for key, label in (("scale", "coupling scale over the truth"), ("jnn", "J_nn relative error")):
        for e, est in enumerate(("least squares", "S")):
            x = z[key][:, :, e]
            m, se = x.mean(axis=1), x.std(axis=1, ddof=1) / np.sqrt(N_REAL)
            dm, dse = m[1:] - m[0], np.sqrt(se[1:] ** 2 + se[0] ** 2)
            s2 = lv[1:] ** 2
            w = 1.0 / dse**2
            slope = float(np.sum(w * s2 * dm) / np.sum(w * s2 * s2))
            slope_se = float(1.0 / np.sqrt(np.sum(w * s2 * s2)))
            at = lv[1] ** 2
            out[f"{label}, {est}: mean, se per level"] = [[float(a), float(b)] for a, b in zip(m, se)]
            out[f"{label}, {est}: change against sigma_d = 0 (mean, se) at 217 and 276 rad/s"] = [
                [float(a), float(b)] for a, b in zip(dm, dse)]
            out[f"{label}, {est}: fit through zero in sigma_d^2, read at 216.9 rad/s (value, se)"] = [
                slope * at, slope_se * at]
            fits[(key, est)] = (m, se)
    out["coupling scale, sd over realisations per level: least squares, S"] = [
        [float(s) for s in z["scale"][i].std(axis=0, ddof=1)] for i in range(len(LEVELS))]
    from scipy import stats
    b = np.load(RESULTS / "e27_stepB.npz")
    for e, (est, key) in enumerate((("least squares", "dm_theta_ls"), ("S", "dm_theta_s"))):
        sb = np.array([coupling_scale(t) for t in b[key]])
        w = stats.ttest_ind(sb, z["scale"][1, :, e], equal_var=False)
        u = stats.mannwhitneyu(sb, z["scale"][1, :, e])
        out[f"E27 step B against this run at 217 rad/s, coupling scale of {est}: means, Welch p, Mann-Whitney p"] = [
            float(sb.mean()), float(z["scale"][1, :, e].mean()), float(w.pvalue), float(u.pvalue)]
    out["beta (rad/s per ms, truth 38.2): least squares, S per level"] = [
        [float(b) * 1e3 for b in z["beta"][i].mean(axis=0)] for i in range(len(LEVELS))]
    out["lambda of S per level: mean"] = [float(x.mean()) for x in z["lam"]]
    out["earlier runs (for reference)"] = {
        "E26 part B, J_nn error of S: drift 276 rad/s, no drift": "-3.9 +- 1.2%, -0.2 +- 0.7% (8 and 6 realisations)",
        "E27 pre-registration diagnostic, coupling scale of S: drift 276 rad/s, no drift": "0.967 +- 0.018, 1.010 +- 0.004 (3 each)",
        "E27 steps B and C, coupling scale of S: drift 217 rad/s, no drift": "1.014 +- 0.013, 0.995 +- 0.002 (8 each)"}
    return out, fits


def earlier_runs():
    """S in earlier runs (sigma_d in rad/s, mean, se): E27 steps B and C (coupling scale and J_nn), E26 part B (J_nn)."""
    import json
    e27 = json.loads((RESULTS / "e27_marginal_band.json").read_text(encoding="utf-8"))["parameter level (reported)"]
    e26 = json.loads((RESULTS / "e26_setting_drift.json").read_text(encoding="utf-8"))["part B (synthetic, parameter level)"]
    lv = np.array(LEVELS) * 1e3
    b, c = e27["step B: data-matched drift world"]["S"], e27["step C: shot-only world"]["S"]
    key26 = "S with shot: J_nn relative error mean, se; lambda mean"
    return {"scale": {"E27 steps B, C (8 each)": [(lv[1], *b["coupling scale over the truth: mean, se"]),
                                                  (lv[0], *c["coupling scale over the truth: mean, se"])]},
            "jnn": {"E27 steps B, C (8 each)": [(lv[1], *b["J_nn relative error: mean, se"]),
                                                (lv[0], *c["J_nn relative error: mean, se"])],
                    "E26 part B (8 and 6)": [(lv[2], *e26["drift"][key26][:2]), (lv[0], *e26["shot-only"][key26][:2])]}}


def plot(out, fits):
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    x = np.array(LEVELS) * 1e3
    prev = earlier_runs()
    for ax, key, ylab in ((axes[0], "scale", "coupling scale over the truth"), (axes[1], "jnn", "J_nn relative error")):
        for e, (est, color, mk) in enumerate((("least squares", INK, "D"), ("S", CAT[0], "o"))):
            m, se = fits[(key, est)]
            ax.errorbar(x + 4 * (e - 0.5), m, yerr=2 * se, fmt=mk + "-", color=color, ms=5, capsize=2, lw=1, label=est)
        for j, (label, pts) in enumerate(prev[key].items()):
            px, pm, pse = np.array(pts).T
            ax.errorbar(px + 9 + 5 * j, pm, yerr=2 * pse, fmt="o" if j == 0 else "s", mfc="none", color=CAT[1 + j],
                        ms=5, capsize=2, lw=1, ls="none", label=f"S, {label}")
        ax.axhline(1.0 if key == "scale" else 0.0, color=MUTED, lw=0.6, ls=":")
        ax.set_xticks(x, [f"{v:.0f}" for v in x])
        ax.set_xlabel("drift sigma_d (rad/s)")
        ax.set_ylabel(ylab + ", mean +- 2 se")
        ax.legend(fontsize=6.5, loc="lower left")
    axes[0].set_title("(a) coupling scale, 16 realisations per level", fontsize=9)
    axes[1].set_title("(b) nearest-neighbour coupling", fontsize=9)
    fig.suptitle("E27c (post hoc): coupling shrinkage against the setting drift (synthetic; initial states from Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e27c_drift_shrinkage")


if __name__ == "__main__":
    z = run()
    out, fits = analyse(z)
    save_json(RESULTS / "e27c_drift_shrinkage.json", out)
    print(plot(out, fits))
    for key, val in out.items():
        print(key, val, flush=True)
