"""E6c: why the re-learned 10-ion generator misses <X5X6> in state 2, resolved by charge.

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0).
Variants of the E6a fit, all validated on 5 < t <= 10 ms (the common out-of-sample window):
  W15, W30, W50  full E6a ansatz learned on t <= 1.5, 3, 5 ms (three defect rounds; W15 is the E6a estimate);
  S15, S30, S50  the same windows with structured dephasing Gamma = gamma_c 11^T + diag(g), g >= 0.
A uniform z-field variant is not run: the E6a bootstrap rejects a uniform field (|z| up to 8.5).
Residuals are split into charge-definite combinations: Z_i, Z_iZ_j and the flip-flops (XX+YY)/2, (XY-YX)/2 (q = 0);
X_i, Y_i and the XZ-type pairs (q = +-1); (XX-YY)/2 and (XY+YX)/2 (q = +-2).
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import CAT, MUTED, ORDINAL, RESULTS, save_figure, save_json, style
from e6_ion_learning import (DEFECT, N, PAIRS, RAD_S, RAW, ansatz, derived, observables, per_setting_tables, physical,
                             predict, product_state, series_from)
from e9_symmetry_data import structured_lower
from qerr.iondata import infer_bases, load_10ion, pauli_means
from qerr.learning import (fit_bounded, gamma_matrix, integral_system, learn, needed_strings, quadrature_defect,
                           structured_map, to_openmodel)

WINDOWS = {"15": 1.5, "30": 3.0, "50": 5.0}
T_VALID = 5.0
VARIANTS = ["W15", "W30", "W50", "S15", "S30", "S50"]


def load():
    times, s1 = load_10ion(RAW, "state1")
    _, s2 = load_10ion(RAW, "state2")
    stats = lambda s: (s[0].mean(axis=1).astype(float), s[0].std(axis=1, ddof=1) / np.sqrt(s.shape[2]))
    (m1, e1), (m2, e2) = stats(s1), stats(s2)
    bases, _ = infer_bases(m1, m2, e1, e2)
    return times, s1, s2, bases


def learn_job(key):
    """Full and structured estimates for one learning window (W15 starts from the E6a estimate, no relearning)."""
    t_learn = WINDOWS[key]
    times, s1, s2, bases = load()
    ans, obs = ansatz(), observables()
    tables = per_setting_tables((s1, s2), bases, sorted(set(obs) | set(needed_strings(obs, ans))))
    series = series_from(tables)
    rho0s = [product_state(series, s) for s in (0, 1)]
    if key == "15":
        thetas = [np.load(RESULTS / "e6_ion_learning.npz")["theta"]]
    else:
        thetas = learn(series, ans, obs, times, t_learn, rho0s, n_iter=3, **DEFECT)
    defect = quadrature_defect(to_openmodel(thetas[-1], ans, sparse=True), rho0s, needed_strings(obs, ans), times,
                               t_learn, **DEFECT)
    A, y, _ = integral_system(obs, ans, series, times, t_learn, defect=defect)
    P = structured_map(ans)
    struct = P @ fit_bounded(A @ P, y, structured_lower(ans))
    rounds = [float(np.abs(a - b).max() * RAD_S) for a, b in zip(thetas[1:], thetas[:-1])]
    return {"full": thetas[-1], "struct": struct, "rounds": rounds}


def validate_job(args):
    theta, s = args
    times, s1, s2, bases = load()
    ans, obs = ansatz(), observables()
    tables = per_setting_tables((s1, s2), bases, [((i, a),) for i in range(N) for a in "XYZ"])
    rho0 = product_state(series_from(tables), s)
    return predict(physical(theta, ans), rho0, times, obs)


def run():
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=3) as pool:
        est = dict(zip(WINDOWS, pool.map(learn_job, list(WINDOWS))))
    print(f"learned at {time.time() - t0:.0f} s; defect rounds:", {k: v["rounds"] for k, v in est.items()}, flush=True)
    thetas = {f"W{k}": v["full"] for k, v in est.items()} | {f"S{k}": v["struct"] for k, v in est.items()}
    jobs = [(thetas[v], s) for v in VARIANTS for s in (0, 1)]
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        preds = list(pool.map(validate_job, jobs))
    print(f"validated at {time.time() - t0:.0f} s", flush=True)
    pred = {v: np.stack(preds[2 * i:2 * i + 2]) for i, v in enumerate(VARIANTS)}
    times, s1, s2, bases = load()
    obs = observables()
    mean1, sem1, _ = pauli_means(s1, bases, obs)
    mean2, sem2, _ = pauli_means(s2, bases, obs)
    return times, thetas, pred, np.stack([mean1, mean2]), np.stack([sem1, sem2]), est


def charge_classes(obs):
    """{class: [(coefficients over obs indices), ...]} of charge-definite combinations."""
    k = {p: m for m, p in enumerate(obs)}
    single = lambda i, a: k[((i, a),)]
    pair = lambda i, j, a, b: k[((i, a), (j, b))]
    cls = {"q=0  Z_i": [{single(i, "Z"): 1.0} for i in range(N)],
           "q=0  Z_iZ_j": [{pair(i, j, "Z", "Z"): 1.0} for i, j in PAIRS],
           "q=0  flip-flop": [c for i, j in PAIRS for c in (
               {pair(i, j, "X", "X"): 0.5, pair(i, j, "Y", "Y"): 0.5},
               {pair(i, j, "X", "Y"): 0.5, pair(i, j, "Y", "X"): -0.5})],
           "q=±1  X_i, Y_i": [{single(i, a): 1.0} for i in range(N) for a in "XY"],
           "q=±1  XZ-type": [{pair(i, j, a, b): 1.0} for i, j in PAIRS for a, b in ("XZ", "ZX", "YZ", "ZY")],
           "q=±2  pair": [c for i, j in PAIRS for c in (
               {pair(i, j, "X", "X"): 0.5, pair(i, j, "Y", "Y"): -0.5},
               {pair(i, j, "X", "Y"): 0.5, pair(i, j, "Y", "X"): 0.5})]}
    return cls


def combine(values, coeffs):
    """values (..., n_obs) -> (..., n_combos) for a list of {obs index: coefficient}."""
    return np.stack([sum(c * values[..., m] for m, c in co.items()) for co in coeffs], axis=-1)


def combine_sem(sem, coeffs):
    return np.stack([np.sqrt(sum(c**2 * sem[..., m] ** 2 for m, c in co.items())) for co in coeffs], axis=-1)


def flipflop_rates(theta, ans):
    """Charge-0 nearest-neighbour dephasing rates 2(G_ii + G_jj - 2 G_ij) of the PSD-projected Gamma (rad/s)."""
    w, V = np.linalg.eigh(gamma_matrix(theta, ans))
    G = (V * np.clip(w, 0.0, None)) @ V.T
    return np.array([2 * (G[i, i] + G[i + 1, i + 1] - 2 * G[i, i + 1]) for i in range(N - 1)]) * RAD_S


def analyse(times, thetas, pred, data, sem, est):
    ans, obs = ansatz(), observables()
    mask = times > T_VALID + 1e-9
    cls = charge_classes(obs)
    out, table = {}, {}
    out["uniform z-field variant"] = "not run: E6a bootstrap z-scores of B_z - mean reach -8.5 (site-resolved field)"
    out["defect rounds max|dtheta| (rad/s)"] = {k: v["rounds"] for k, v in est.items()}
    for name, co in cls.items():
        d = combine(data[:, mask], co)
        s = combine_sem(sem[:, mask], co)
        table[name] = {v: float(np.sqrt(np.nanmean((combine(pred[v][:, mask], co) - d) ** 2))) for v in VARIANTS}
        table[name]["SEM"] = float(np.sqrt(np.nanmean(s**2)))
    out[f"validation rms (t > {T_VALID:g} ms) by charge class"] = table
    allres = {v: float(np.sqrt(np.nanmean((pred[v][:, mask] - data[:, mask]) ** 2))) for v in VARIANTS}
    out[f"validation rms (t > {T_VALID:g} ms), all 435 Pauli observables"] = allres
    k56 = {p: m for m, p in enumerate(obs)}
    c56 = [{k56[((4, "X"), (5, "X"))]: 0.5, k56[((4, "Y"), (5, "Y"))]: 0.5},
           {k56[((4, "X"), (5, "X"))]: 0.5, k56[((4, "Y"), (5, "Y"))]: -0.5}]
    curves = {"data": combine(data[1], c56), "sem": combine_sem(sem[1], c56)}
    curves |= {v: combine(pred[v][1], c56) for v in VARIANTS}
    out["state 2, pair (5,6): C0 = (XX+YY)/2 at t = 10 ms, data and variants"] = {
        k: float(c[-1, 0]) for k, c in curves.items()}
    out["charge-0 nn dephasing rates 2(G_ii+G_jj-2G_ij) of the physical Gamma (rad/s), pairs (1,2)..(9,10)"] = {
        v: [round(float(x), 1) for x in flipflop_rates(thetas[v], ans)] for v in VARIANTS}
    for v in VARIANTS:
        d = derived(thetas[v], ans)
        out[f"{v}: J_nn (rad/s), structured gamma_c (rad/s), B_z spread (rad/s)"] = [
            float(d["J_d"][0] * RAD_S), float(d["gamma_c_struct"] * RAD_S), float(np.ptp(d["Bz"]) * RAD_S)]
    return out, curves, table


def plot(times, curves, table, thetas):
    plt = style()
    ans = ansatz()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), gridspec_kw={"width_ratios": [1.15, 1.2, 1.0]})
    ax = axes[0]
    shown = [("W15", CAT[0], "E6a (window 1.5 ms)"), ("W50", CAT[1], "window 5 ms"),
             ("S50", CAT[2], "window 5 ms, structured Γ")]
    for col_idx, (lab, mk) in enumerate((("C0 = (XX+YY)/2  (q = 0)", "o"), ("C2 = (XX−YY)/2  (q = ±2)", "s"))):
        ax.errorbar(times, curves["data"][:, col_idx], yerr=curves["sem"][:, col_idx], fmt=mk, ms=3, color=MUTED,
                    lw=0.8, label=f"data {lab}")
        for v, col, name in shown:
            ax.plot(times, curves[v][:, col_idx], color=col, lw=1.2, ls="-" if col_idx == 0 else "--",
                    label=name if col_idx == 0 else None)
    ax.axvspan(T_VALID, times[-1], color="#e4e3df", alpha=0.5, lw=0)
    ax.text(T_VALID + 0.15, 0.25, "validation window", transform=ax.get_xaxis_transform(), fontsize=7, color=MUTED)
    ax.set_xlabel("time  [ms]")
    ax.set_title("state 2, ions 5–6: charge-resolved correlations", fontsize=9)
    ax.legend(fontsize=6.0, loc="lower right", bbox_to_anchor=(1.0, 0.01), ncol=2)
    ax = axes[1]
    names = list(table)
    ramp = {"W15": ORDINAL[0], "W30": ORDINAL[1], "W50": ORDINAL[3], "S15": CAT[1], "S30": "#f29a6b", "S50": CAT[2]}
    for u, v in enumerate(VARIANTS):
        ax.plot(np.arange(len(names)) + (u - 2.5) * 0.09, [table[n][v] for n in names], "o", ms=3.6,
                color=ramp[v], lw=0, label=v)
    ax.plot(np.arange(len(names)), [table[n]["SEM"] for n in names], "_", ms=14, color=MUTED, mew=1.5,
            label="statistical (SEM)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("  ", "\n") for n in names], fontsize=7)
    ax.set_ylabel(f"rms(model − data), t > {T_VALID:g} ms")
    ax.set_title("out-of-sample error by charge class", fontsize=9)
    ax.legend(fontsize=6.3, ncol=2, loc="upper right")
    ax = axes[2]
    for v in ("W15", "W50", "S15", "S50"):
        ax.plot(np.arange(1, N), flipflop_rates(thetas[v], ans), "o-", ms=3, lw=1.0, color=ramp[v], label=v)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks(range(1, N))
    ax.set_xticklabels([f"{i},{i + 1}" for i in range(1, N)], fontsize=6.5)
    ax.set_xlabel("ion pair")
    ax.set_ylabel("2(Γᵢᵢ + Γⱼⱼ − 2Γᵢⱼ)  [rad/s]")
    ax.set_title("learned flip-flop dephasing (q = 0)", fontsize=9)
    ax.legend(fontsize=6.5)
    fig.suptitle("E6c  charge-resolved validation of the re-learned 10-ion generator", fontsize=10, y=1.03)
    fig.tight_layout()
    return save_figure(fig, "e6c_validation")


if __name__ == "__main__":
    if "--replot" not in sys.argv:
        times, thetas, pred, data, sem, est = run()
        np.savez_compressed(RESULTS / "e6c_validation.npz", times=times, data=data, sem=sem,
                            rounds=np.array([str(est[k]["rounds"]) for k in WINDOWS]),
                            **{f"theta_{v}": thetas[v] for v in VARIANTS}, **{f"pred_{v}": pred[v] for v in VARIANTS})
    z = np.load(RESULTS / "e6c_validation.npz")
    times, data, sem = z["times"], z["data"], z["sem"]
    thetas = {v: z[f"theta_{v}"] for v in VARIANTS}
    pred = {v: z[f"pred_{v}"] for v in VARIANTS}
    est = {k: {"rounds": r} for k, r in zip(WINDOWS, z["rounds"])}
    out, curves, table = analyse(times, thetas, pred, data, sem, est)
    save_json(RESULTS / "e6c_validation.json", {"results": out})
    print(plot(times, curves, table, thetas))
    for k, v in out.items():
        print(f"{k}: {v}")
