"""E11: which generator do the BEQS 10-ion data support? Model selection, emission, and a memory test.

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0).
Every variant is learned with the integral method and validated out of sample on 5 < t <= 10 ms (the E6c metric).
Variants (Hamiltonian / dephasing / emission / learning window):
  W15   full E6a ansatz on t <= 1.5 ms: all XX, YY, XY, YX couplings, X, Y, Z fields, full Gamma, free sigma^- rates
  S15   structured Gamma = gamma_c 11^T + diag(g), g >= 0
  C15   collective dephasing only, Gamma = gamma_c 11^T (no local dephasing)
  C15e  C15 with the emission fixed at the D5/2 decay rate 1/(1.168 s)
  X15   U(1)-symmetric couplings J (XX + YY)/2 and D (XY - YX)/2, all single-site fields; collective; fixed emission
  Y15   X15 without the transverse fields (fully U(1)-symmetric Hamiltonian)
  X30, X50     X15 learned on t <= 3 and 5 ms
  Xb15, Xb30   X15, X30 with the collective rate gamma_c + beta t (white plus quasi-static noise, Theorem G of
               docs/theory/phase4-theory.md); the t-weighted collective column carries beta, and beta > 0 is memory
The quadrature defect of every window (and of the weighted column) comes from one fine trajectory (dt = 0.125 ms) of
the C15 model, which is itself fitted with the E6a defect. Bootstrap over the 200 settings for the X and Y variants.
Internally ms and rad/ms; reported in rad/s.
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import lsq_linear

from common import CAT, INK, MUTED, ORDINAL, RESULTS, save_figure, save_json, style
from e6_ion_learning import (DEFECT, N, PAIRS, RAD_S, ansatz, observables, per_setting_tables, physical,
                             product_state, series_from)
from e6c_validation import charge_classes, combine, combine_sem, load
from qerr.iondata import pauli_means
from qerr.learning import (integral_system, model_trajectory, needed_strings, quadrature_defect, refine,
                           to_openmodel, trajectory_defect)
from qerr.lindblad import evolve, evolve_td
from qerr.pauli import expectation_tables, expectations

EMISSION_D52 = 1.0 / 1168.0  # 1/ms
N_BOOT, WORKERS, T_VALID, N_SUB = 100, 12, 5.0, 4
VARIANTS = {  # name: (hamiltonian, dephasing, emission, beta, window in ms)
    "W15": ("full", "full", "free", False, 1.5),
    "S15": ("full", "structured", "free", False, 1.5),
    "C15": ("full", "collective", "free", False, 1.5),
    "C15e": ("full", "collective", "fixed", False, 1.5),
    "X15": ("symxy", "collective", "fixed", False, 1.5),
    "Y15": ("sym", "collective", "fixed", False, 1.5),
    "Xb15": ("symxy", "collective", "fixed", True, 1.5),
    "X30": ("symxy", "collective", "fixed", False, 3.0),
    "Xb30": ("symxy", "collective", "fixed", True, 3.0),
    "X50": ("symxy", "collective", "fixed", False, 5.0),
}
BOOT = ["X15", "Y15", "Xb15", "X30", "Xb30"]
STAGE1 = RESULTS / "e11_stage1.npz"  # learning and bootstrap; delete it to relearn
_W = {}


def variant_map(ans, ham, deph, emis):
    """theta = T phi + fix and the lower bounds of phi, for one variant of the E6a ansatz."""
    k = {p: m for m, p in enumerate(ans.h_terms)}
    k0, nd = len(ans.h_terms), len(ans.deph_pairs)
    cols, lower = [], []

    def col(entries, lo=-np.inf):
        c = np.zeros(ans.n_params)
        for m, v in entries:
            c[m] = v
        cols.append(c)
        lower.append(lo)

    for i, j in PAIRS:
        xx, yy, xy, yx = (k[((i, a), (j, b))] for a, b in ("XX", "YY", "XY", "YX"))
        if ham == "full":
            for m in (xx, yy, xy, yx):
                col([(m, 1.0)])
        else:
            col([(xx, 1.0), (yy, 1.0)])  # theta_XX = theta_YY = J_ij / 2
            col([(xy, 1.0), (yx, -1.0)])  # D_ij (XY - YX)/2 ... as theta_XY = -theta_YX
    for i in range(N):
        for a in "XYZ":
            if ham != "sym" or a == "Z":
                col([(k[((i, a),)], 1.0)])
    if deph == "full":
        for m in range(nd):
            col([(k0 + m, 1.0)])
    else:
        col([(k0 + m, 1.0) for m in range(nd)])  # gamma_c enters every Gamma_ij
        if deph == "structured":
            for m, (i, j) in enumerate(ans.deph_pairs):
                if i == j:
                    col([(k0 + m, 1.0)], lo=0.0)
    fix = np.zeros(ans.n_params)
    ke, ne = k0 + nd, len(ans.emission_sites)
    if emis == "free":
        for m in range(ne):
            col([(ke + m, 1.0)])
    else:
        fix[ke:ke + ne] = EMISSION_D52
    return np.array(cols).T, fix, np.array(lower)


def collective_direction(ans):
    c = np.zeros(ans.n_params)
    c[len(ans.h_terms):len(ans.h_terms) + len(ans.deph_pairs)] = 1.0
    return c


def fit_variant(name, systems, ans):
    """(theta, beta) of one variant from systems[window] = (A, y, A_t)."""
    ham, deph, emis, beta, win = VARIANTS[name]
    A, y, At = systems[win]
    T, fix, lower = variant_map(ans, ham, deph, emis)
    M = A @ T
    if beta:
        M = np.column_stack([M, At @ collective_direction(ans)])
        lower = np.r_[lower, -np.inf]
    phi = lsq_linear(M, y - A @ fix, bounds=(lower, np.inf), method="bvls", tol=1e-12).x
    theta = T @ phi[:T.shape[1]] + fix
    return theta, (float(phi[-1]) if beta else 0.0), float(np.sqrt(np.mean((M @ phi - (y - A @ fix)) ** 2)))


def build_systems(series, times, defects, obs, ans, windows):
    out = {}
    for win in windows:
        A, y, _ = integral_system(obs, ans, series, times, win, defect=defects[(win, False)])
        At, _, _ = integral_system(obs, ans, series, times, win, defect=defects[(win, True)], weight=lambda s: s)
        out[win] = (A, y, At)
    return out


def _init(tables, times, defects):
    _W.update(tables=tables, times=times, defects=defects)


def replicate(w):
    ans, obs = ansatz(), observables()
    windows = sorted({VARIANTS[v][4] for v in BOOT})
    systems = build_systems(series_from(_W["tables"], w), _W["times"], _W["defects"], obs, ans, windows)
    return {v: fit_variant(v, systems, ans)[:2] for v in BOOT}


def trajectory_job(args):
    theta, s = args
    times, s1, s2, bases = load()
    ans, obs = ansatz(), observables()
    tables = per_setting_tables((s1, s2), bases, [((i, a),) for i in range(N) for a in "XYZ"])
    rho0 = product_state(series_from(tables), s)
    return model_trajectory(physical(theta, ans), [rho0], needed_strings(obs, ans), refine(times, N_SUB),
                            rtol=DEFECT["rtol"], atol=DEFECT["atol"])[0]


def validate_job(args):
    name, theta, beta, s = args
    times, s1, s2, bases = load()
    ans, obs = ansatz(), observables()
    tables = per_setting_tables((s1, s2), bases, [((i, a),) for i in range(N) for a in "XYZ"])
    rho = product_state(series_from(tables), s)
    model = physical(theta, ans)
    gamma_c = float(theta[len(ans.h_terms)])
    idx, ph = expectation_tables(obs, N)
    out = [expectations(rho, idx, ph)]
    for a, b in zip(times[:-1], times[1:]):
        if beta == 0.0:
            rho = evolve(model, rho, [0.0, b - a], rtol=1e-7, atol=1e-9)[-1]
        else:  # collective rate gamma_c + beta t, kept >= 0
            rate = lambda u, a=a: max(beta * (a + u), -gamma_c)
            rho = evolve_td(model, rho, [0.0, b - a], np.ones((N, N)), rate, rtol=1e-7, atol=1e-9)[-1]
        out.append(expectations(rho, idx, ph))
    return np.array(out)


def run():
    times, s1, s2, bases = load()
    if STAGE1.exists():  # learning and bootstrap are cached; only the validation runs again
        z = np.load(STAGE1)
        est = {v: (z[f"theta_{v}"], float(z[f"beta_{v}"]), float(z[f"res_{v}"])) for v in VARIANTS}
        boot = [{v: (z[f"boot_theta_{v}"][i], float(z[f"boot_beta_{v}"][i])) for v in BOOT}
                for i in range(z["boot_theta_X15"].shape[0])]
        print(f"stage 1 loaded from {STAGE1.name}", flush=True)
    else:
        est, boot = learn_and_bootstrap(times, s1, s2, bases)
    t0 = time.time()
    jobs = [(v, est[v][0], est[v][1], s) for v in VARIANTS for s in (0, 1)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        preds = list(pool.map(validate_job, jobs))
    print(f"validated in {time.time() - t0:.0f} s", flush=True)
    pred = {v: np.stack(preds[2 * i:2 * i + 2]) for i, v in enumerate(VARIANTS)}
    obs = observables()
    mean1, sem1, _ = pauli_means(s1, bases, obs)
    mean2, sem2, _ = pauli_means(s2, bases, obs)
    return times, est, boot, pred, np.stack([mean1, mean2]), np.stack([sem1, sem2])


def learn_and_bootstrap(times, s1, s2, bases):
    t0 = time.time()
    ans, obs = ansatz(), observables()
    strings = needed_strings(obs, ans)
    tables = per_setting_tables((s1, s2), bases, sorted(set(obs) | set(strings)))
    series = series_from(tables)
    rho0s = [product_state(series, s) for s in (0, 1)]
    theta_w15 = np.load(RESULTS / "e6_ion_learning.npz")["theta"]
    d_e6a = quadrature_defect(to_openmodel(theta_w15, ans, sparse=True), rho0s, strings, times, 1.5, **DEFECT)
    d_e6a_t = {key: 0.0 for key in d_e6a}  # the reference fit has no weighted column
    ref = fit_variant("C15", build_systems(series, times, {(1.5, False): d_e6a, (1.5, True): d_e6a_t}, obs, ans,
                                           [1.5]), ans)[0]
    print(f"reference C15 at {time.time() - t0:.0f} s", flush=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        traj = list(pool.map(trajectory_job, [(ref, 0), (ref, 1)]))
    fine = refine(times, N_SUB)
    windows = sorted({v[4] for v in VARIANTS.values()})
    defects = {(win, wt): trajectory_defect(fine, traj, strings, times, win, weight=(lambda s: s) if wt else None)
               for win in windows for wt in (False, True)}
    print(f"trajectory and defects at {time.time() - t0:.0f} s", flush=True)
    systems = build_systems(series, times, defects, obs, ans, windows)
    est = {v: fit_variant(v, systems, ans) for v in VARIANTS}
    print(f"point fits at {time.time() - t0:.0f} s:",
          {v: (round(e[1] * RAD_S, 2), round(e[2], 5)) for v, e in est.items()}, flush=True)
    rng = np.random.default_rng(2028)
    n_set = bases.shape[0]
    weights = [np.bincount(rng.integers(0, n_set, n_set), minlength=n_set).astype(float) for _ in range(N_BOOT)]
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init, initargs=(tables, times, defects)) as pool:
        boot = list(pool.map(replicate, weights))
    print(f"bootstrap at {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(STAGE1, traj0=traj[0], traj1=traj[1],
                        **{f"theta_{v}": e[0] for v, e in est.items()}, **{f"beta_{v}": e[1] for v, e in est.items()},
                        **{f"res_{v}": e[2] for v, e in est.items()},
                        **{f"boot_theta_{v}": np.array([b[v][0] for b in boot]) for v in BOOT},
                        **{f"boot_beta_{v}": np.array([b[v][1] for b in boot]) for v in BOOT})
    return est, boot


def hamiltonian_summary(theta, ans):
    k = {p: m for m, p in enumerate(ans.h_terms)}
    J = np.zeros((N, N))
    for i, j in PAIRS:
        J[i, j] = J[j, i] = theta[k[((i, "X"), (j, "X"))]] + theta[k[((i, "Y"), (j, "Y"))]]
    field = {a: np.array([theta[k[((i, a),)]] for i in range(N)]) for a in "XYZ"}
    jnn = float(np.mean([J[i, i + 1] for i in range(N - 1)]))
    d = np.arange(1, N)
    Jd = np.array([np.mean([J[i, i + dd] for i in range(N - dd)]) for dd in d])
    ok = Jd > 0
    alpha = float(-np.polyfit(np.log(d[ok]), np.log(Jd[ok]), 1)[0])
    return {"J_nn": jnn, "alpha": alpha, "Bz_sd": float(np.std(field["Z"])), "Bx": field["X"], "By": field["Y"]}


def analyse(times, est, boot, pred, data, sem):
    ans, obs = ansatz(), observables()
    k0, ke = len(ans.h_terms), len(ans.h_terms) + len(ans.deph_pairs)
    mask = times > T_VALID + 1e-9
    cls = charge_classes(obs)
    table = {}
    for name, coeffs in cls.items():
        d = combine(data, coeffs)[:, mask]
        row = {v: float(np.sqrt(np.mean((combine(p, coeffs)[:, mask] - d) ** 2))) for v, p in pred.items()}
        row["SEM"] = float(np.sqrt(np.mean(combine_sem(sem, coeffs)[:, mask] ** 2)))
        table[name] = row
    overall = {v: float(np.sqrt(np.mean((p[:, mask] - data[:, mask]) ** 2))) for v, p in pred.items()}
    params = {}
    for v, (theta, beta, res) in est.items():
        h = hamiltonian_summary(theta, ans)
        params[v] = {"J_nn (rad/s)": h["J_nn"] * RAD_S, "alpha": h["alpha"], "Bz spread (rad/s)": h["Bz_sd"] * RAD_S,
                     "gamma_c (rad/s)": float(theta[k0]) * RAD_S if VARIANTS[v][1] != "full" else None,
                     "beta (rad/s per ms)": beta * RAD_S, "emission mean (1/s)": float(theta[ke:].mean()) * RAD_S,
                     "in-window rms residual": res}
    boot_sum = {}
    for v in BOOT:
        th = np.array([b[v][0] for b in boot])
        be = np.array([b[v][1] for b in boot]) * RAD_S
        hs = [hamiltonian_summary(t, ans) for t in th]
        jnn = np.array([h["J_nn"] for h in hs]) * RAD_S
        gc = th[:, k0] * RAD_S
        bxy = np.array([np.r_[h["Bx"], h["By"]] for h in hs]) * RAD_S
        z_bxy = hamiltonian_summary(est[v][0], ans)
        z_bxy = np.r_[z_bxy["Bx"], z_bxy["By"]] * RAD_S / np.where(bxy.std(axis=0) > 0, bxy.std(axis=0), np.inf)
        boot_sum[v] = {"J_nn sd": float(jnn.std(ddof=1)), "gamma_c sd": float(gc.std(ddof=1)),
                       "gamma_c 2.5/97.5%": [float(x) for x in np.percentile(gc, [2.5, 97.5])],
                       "beta sd": float(be.std(ddof=1)), "beta 2.5/97.5%": [float(x) for x in np.percentile(be, [2.5, 97.5])],
                       "transverse fields |z| > 3": int(np.sum(np.abs(z_bxy) > 3)), "max |z| Bx, By": float(np.abs(z_bxy).max())}
    out_cert = {}
    t10 = 10.0
    bb = np.linspace(0.0, t10, 4001)

    def zeno_factor(chi):  # second-order leakage with the kernel chi, relative to no protection: int (t-b) chi / (t^2/2)
        return float(np.trapezoid((t10 - bb) * chi, bb) / (0.5 * t10**2))

    for v in ("Xb15", "Xb30"):
        gM, be = float(est[v][0][k0]), est[v][1]
        for q in (1, 2):
            w = np.maximum(4 * gM * bb + 2 * be * bb**2, 0.0)  # Var Phi for gamma_c(t) = gamma_M + beta t
            out_cert[f"{v} fitted mixture, |Q|={q}"] = zeno_factor(np.exp(-q**2 * w / 2))
    for v in ("X15", "X30"):
        gc = float(est[v][0][k0])
        T2 = 1.0 / (2 * gc)
        for q in (1, 2):
            out_cert[f"{v} white, |Q|={q}"] = zeno_factor(np.exp(-2 * q**2 * gc * bb))
            out_cert[f"{v} quasi-static with the same T2, |Q|={q}"] = zeno_factor(np.exp(-(q * bb / T2) ** 2))
    best = min(overall, key=overall.get)
    acceptance = {
        "collective-only beats the structured and full models out of sample": overall["C15"] < min(overall["S15"], overall["W15"]),
        "fixing the emission at the D5/2 rate does not hurt (C15e <= C15 + 0.002)": overall["C15e"] <= overall["C15"] + 0.002,
        "best variant": best,
    }
    return {"validation rms (t > 5 ms) by class": table, "validation rms, all 435 observables": overall,
            "parameters": params, "bootstrap": boot_sum,
            "Zeno factor at 10 ms (second-order leakage relative to no protection)": out_cert,
            "acceptance": acceptance}


def plot(times, pred, data, sem, out):
    plt = style()
    obs = observables()
    k = {p: m for m, p in enumerate(obs)}
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.6))
    ax = axes[0, 0]
    table = out["validation rms (t > 5 ms) by class"]
    names = list(table)
    show = [("W15", ORDINAL[0], "o"), ("S15", ORDINAL[1], "s"), ("C15", ORDINAL[2], "D"), ("X15", CAT[1], "^"),
            ("Y15", CAT[2], "v"), ("Xb15", ORDINAL[3], "*"), ("Xb30", MUTED, "P")]
    for j, (v, c, mk) in enumerate(show):
        ax.plot([table[n][v] for n in names], np.arange(len(names)) + 0.09 * (j - 3), mk, color=c,
                ms=8 if mk == "*" else 6, label=v, ls="none")
    ax.plot([table[n]["SEM"] for n in names], np.arange(len(names)), "|", color=INK, ms=14, mew=2,
            label="statistical error")
    ax.set_yticks(np.arange(len(names)), [n.replace("±", "±") for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("out-of-sample rms error, 5 < t ≤ 10 ms")
    ax.set_title("(a) validation by charge class")
    ax.legend(fontsize=7, loc="lower right")
    nn = [(i, i + 1) for i in range(N - 1)]

    def flipflop(v, s):
        return np.mean([0.5 * (v[s][:, k[((i, "X"), (j, "X"))]] + v[s][:, k[((i, "Y"), (j, "Y"))]]) for i, j in nn],
                       axis=0)

    def zmean(v, s):
        return np.mean([v[s][:, k[((i, "Z"),)]] for i in range(N)], axis=0)

    for ax, f, title in ((axes[0, 1], flipflop, "(b) nearest-neighbour flip-flop $(XX+YY)/2$ (q = 0)"),
                         (axes[1, 0], zmean, "(c) site-averaged $\\langle Z\\rangle$ (q = 0)")):
        for s, ls in ((0, "-"), (1, "--")):
            ax.plot(times, f(data, s), "o", color=INK, ms=3.5, label="data" if s == 0 else None)
            for v, c in (("W15", ORDINAL[0]), ("S15", ORDINAL[1]), ("X15", CAT[1])):
                ax.plot(times, f(pred[v], s), ls, color=c, label=v if s == 0 else None)
        ax.axvspan(0, 1.5, color=MUTED, alpha=0.08, lw=0)
        ax.set_xlabel("t (ms)")
        ax.set_title(title)
        ax.legend(fontsize=7)
    axes[0, 1].text(0.98, 0.04, "solid: state 1, dashed: state 2; shaded: learning window",
                    transform=axes[0, 1].transAxes, ha="right", fontsize=7, color=MUTED)
    ax = axes[1, 1]
    p, b = out["parameters"], out["bootstrap"]
    for v, c in (("Xb15", CAT[1]), ("Xb30", CAT[2])):
        win = VARIANTS[v][4]
        tt = np.linspace(0, win, 50)
        g, be = p[v]["gamma_c (rad/s)"], p[v]["beta (rad/s per ms)"]
        ax.plot(tt, g + be * tt, color=c, label=f"{v}: $\\gamma_c$ + $\\beta t$, $\\beta$ = {be:.1f} ± {b[v]['beta sd']:.1f}")
    for v, c in (("X15", CAT[1]), ("X30", CAT[2])):
        g, sd = p[v]["gamma_c (rad/s)"], b[v]["gamma_c sd"]
        ax.axhspan(g - sd, g + sd, color=c, alpha=0.15, lw=0)
        ax.axhline(g, color=c, ls=":", label=f"{v}: constant $\\gamma_c$ = {g:.1f} ± {sd:.1f}")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("collective rate (rad/s)")
    ax.set_title("(d) memory test: growing collective rate?")
    ax.legend(fontsize=7)
    fig.suptitle("E11: which generator do the BEQS 10-ion data support? (data: Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e11_model_selection")


if __name__ == "__main__":
    if "--replot" not in sys.argv:
        times, est, boot, pred, data, sem = run()
        np.savez_compressed(RESULTS / "e11_model_selection.npz", times=times, data=data, sem=sem,
                            **{f"theta_{v}": e[0] for v, e in est.items()},
                            **{f"beta_{v}": e[1] for v, e in est.items()},
                            **{f"res_{v}": e[2] for v, e in est.items()},
                            **{f"pred_{v}": p for v, p in pred.items()},
                            **{f"boot_theta_{v}": np.array([b[v][0] for b in boot]) for v in BOOT},
                            **{f"boot_beta_{v}": np.array([b[v][1] for b in boot]) for v in BOOT})
    z = np.load(RESULTS / "e11_model_selection.npz")
    times, data, sem = z["times"], z["data"], z["sem"]
    est = {v: (z[f"theta_{v}"], float(z[f"beta_{v}"]), float(z[f"res_{v}"])) for v in VARIANTS}
    pred = {v: z[f"pred_{v}"] for v in VARIANTS}
    boot = [{v: (z[f"boot_theta_{v}"][i], float(z[f"boot_beta_{v}"][i])) for v in BOOT}
            for i in range(z["boot_theta_X15"].shape[0])]
    out = analyse(times, est, boot, pred, data, sem)
    save_json(RESULTS / "e11_model_selection.json", out)
    print(plot(times, pred, data, sem, out))
    for key, val in out.items():
        print(f"{key}: {val}")
