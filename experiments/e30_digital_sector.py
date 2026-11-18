"""E30: strong-symmetry diagnostic on digital Fermi-Hubbard circuits of a trapped-ion quantum computer (preregistered)

Data: measured by Phasecraft and Quantinuum collaborators on Quantinuum H2-2 (arXiv:2510.26300), Zenodo
10.5281/zenodo.17881394 (CC BY 4.0), analysed here from the public raw shots (data/fetch_phasecraft2025.py). This is
an independent check of the diagnostic of E6b on another group's device and another platform (Q53, Q54).
Setting (paper Sec. "Results", App. C and D). Spinful Fermi-Hubbard model with flux on a doubly periodic 7 x 4 lattice,
Jordan-Wigner encoded on 56 qubits (28 modes per spin). Initial state at half filling with N_up = N_down = 14: S_z = 0
triplet dimers with one holon and one doublon. U = 0 and U = 4; t = 0.1, 0.2, ..., 2.0 (units 1/J). Every circuit is
k = 4 second-order Trotter steps of size t/4, so all 40 circuits share one gate structure and only the gate angles grow
with t. Each (U, t) has 160 shots from 16 Pauli pseudo-twirling instances. Measurement in the occupation basis.
Every gate conserves N_up and N_down (App. D.4), so an ideal run returns (14, 14) in every shot.
Known before this commit: the authors report (App. D.4, Fig. 28) that the Hamming weights are well described by a
Gaussian centred at (14, 14) and that 19% of shots have exactly (14, 14). Structure checks before this commit (no
statistics of N_up or N_down): per U, 20 times x 160 shots x 56 qubits; 28 up and 28 down modes; the shots are not
post-selected (more than one value of N occurs). The paper's two-qubit gate count is at most 2,415 per circuit.

Background (the manuscript's strong-symmetry proposition; test E6b). A channel whose Kraus operators all commute with
the charge keeps the charge distribution fixed, so leaving a sector needs charged jumps. A one-directional process
(every event lowers N by one, like relaxation 1 -> 0, or every event raises it) moves the mean and adds as much
variance as it moves the mean (Poisson limit; a binomial process adds slightly less). A balanced process (events in both
directions at equal rates, like Pauli X or Y errors at half filling) adds variance without moving the mean. E6b found
on the 51-ion quench that the variance grew six times faster than spontaneous decay accounts for.
Time is not exposure here: all circuits share the gate structure. The decomposition is therefore made per circuit run
(events per run), pooled over the 20 times of one U, and the dependence on t is reported separately (R3).

Estimators (fixed).
  For each U in {0, 4} and species s in {up, down}: N_s per shot; for each time t_k, d_k = mean(N_s) - 14 and v_k = the
  sample variance of N_s over the 160 shots.
  Pooled over the 20 circuits: D = mean_k d_k (mean shift), V = mean_k v_k, B = V - |D| (balanced excess: events per run
  beyond what a one-directional process gives) and the one-directional share r = |D| / V.
  Uncertainty: the 20 circuits are treated as replicates of one gate structure. Standard errors come from a bootstrap
  over circuits (20 circuits drawn with replacement, 4000 draws, seed 30), multiplied by sqrt(20/19) (the bootstrap SE
  of a mean over m replicates is sqrt((m-1)/m) of the usual one); 95% intervals of D, V and B are estimate +- t_19 SE,
  the interval of r is the bootstrap percentile interval. The between-circuit spread, including any trend with t, is
  part of the error. B includes balanced readout errors, which these data cannot separate from gate errors (no t = 0
  point).
  Joint sector: the share of shots with (N_up, N_down) = (14, 14), per (U, t) and pooled.
Rules fixed before the run.
  S0 (data handling; stop if failed): the pooled share of shots with (N_up, N_down) = (14, 14) over both U and all t lies
      in [0.17, 0.21] (the authors report 19%).
  R1 (balanced breaking detected): for each of the four (U, s), one-sided p of B / SE(B) from Student t with 19 degrees
      of freedom; Holm-Bonferroni at familywise 5% over the four. "Detected" for those that pass.
  R2 (character): for each (U, s), with the 95% bootstrap interval of r: "balanced-dominated" if its upper end is below
      0.5, "one-directional-dominated" if its lower end is above 0.5, otherwise "undetermined".
  R3 (dependence on the gate angles): for each (U, s), the OLS slope of v_k on t_k with its classical standard error;
      "grows with t" if the one-sided p (slope > 0, Student t with 18 degrees of freedom) passes Holm-Bonferroni at
      familywise 5% over the four.
  Post hoc and descriptive only (never change a verdict): the covariance of N_up and N_down per U, the share of shots
  in the sector per t, the d_k and v_k curves.
Choices made before this commit, on synthetic data only (`--synthetic`, results/e30_synthetic_calibration.json;
  nothing of it enters the data results). Each species: 14 of 28 modes occupied, then one-directional loss q per
  occupied mode, balanced flips p(t) per mode, balanced readout flips e per mode; 20 times x 160 shots; 300 worlds.
  Worlds: one-directional only (q = 0.03), balanced small (p = 0.004), mixed with a trend (q = 0.01, p = 0.01 + 0.01 t,
  e = 0.002), balanced near the published picture (q = 0.002, p = 0.03, e = 0.0015).
  - First design: percentile bootstrap intervals. Coverage of D, V, B was 0.84-0.93 (too narrow for 20 replicates), so
    the intervals were changed to the t_19 form above. With it: coverage 0.93-0.96 (D), 0.92-1.00 (V), 0.91-1.00 (B);
    the lowest B coverage (0.91) is the balanced-small world, where |D| near zero biases B low by about 0.005.
  - R1: rate 0.00 in the one-directional world (its B is -0.012, slightly negative as for a binomial loss), 1.00 in
    all three worlds with balanced events.
  - R2: one-directional-dominated in 100% of the one-directional world, balanced-dominated in 100% of the others.
  - R3: rate 0.02-0.04 in the worlds without a trend, 1.00 in the world with a trend.
  - B is unbiased within 0.005 in all worlds (-0.0124 vs -0.0122, 0.106 vs 0.111, 0.608 vs 0.603, 0.846 vs 0.847).
"""

import argparse
import sys

import numpy as np

from common import CAT, INK, MUTED, RESULTS, ROOT, save_figure, save_json, style

sys.path.insert(0, str(ROOT / "src"))
from qerr.fermidata import load_shots, species_masks  # noqa: E402

DATA = ROOT / "data" / "raw" / "phasecraft2025" / "fermionic_dynamics"
N0, MODES, N_BOOT, SEED = 14, 28, 4000, 30
TEXT_W = 510 / 72  # full text width of the manuscript (REVTeX reprint), inches


# ----------------------------------------------------------------------------------------------- estimators
def pooled(n: np.ndarray) -> dict:
    """n: (times, shots) particle numbers of one species. Returns per-circuit d, v and pooled D, V, B, r."""
    d = n.mean(axis=1) - N0
    v = n.var(axis=1, ddof=1)
    D, V = float(d.mean()), float(v.mean())
    return {"d": d, "v": v, "D": D, "V": V, "B": V - abs(D), "r": abs(D) / V}


def bootstrap(n: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT) -> dict:
    """Bootstrap over circuits: draws of D, V, B and r."""
    p = pooled(n)
    k = rng.integers(0, n.shape[0], size=(n_boot, n.shape[0]))
    D = p["d"][k].mean(axis=1)
    V = p["v"][k].mean(axis=1)
    return {"D": D, "V": V, "B": V - np.abs(D), "r": np.abs(D) / V}


def slope(t: np.ndarray, y: np.ndarray):
    """OLS slope of y on t and its classical standard error."""
    X = np.column_stack([np.ones_like(t), t])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = resid @ resid / (len(y) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    return float(coef[1]), float(np.sqrt(cov[1, 1]))


def t_sf(z, df):
    from scipy.stats import t as student_t
    return float(student_t.sf(z, df))


def t_quantile(q, df):
    from scipy.stats import t as student_t
    return float(student_t.ppf(q, df))


def holm(pvals, alpha=0.05):
    order = np.argsort(pvals)
    passed = np.zeros(len(pvals), dtype=bool)
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (len(pvals) - rank):
            passed[i] = True
        else:
            break
    return passed


def analyse(n_by_key: dict, t: np.ndarray, rng: np.random.Generator) -> dict:
    """n_by_key: {(U, s): (times, shots) array}. Applies the estimators and rules R1-R3."""
    keys = sorted(n_by_key)
    out = {}
    for key in keys:
        n = n_by_key[key]
        m = n.shape[0]
        p = pooled(n)
        b = bootstrap(n, rng)
        # bootstrap SE of a mean over m replicates is sqrt((m-1)/m) of the usual one; undo that and use t_{m-1}
        se = {q: float(b[q].std(ddof=1) * np.sqrt(m / (m - 1))) for q in ("D", "V", "B", "r")}
        tq = t_quantile(0.975, m - 1)
        ci = {q: [p[q] - tq * se[q], p[q] + tq * se[q]] for q in ("D", "V", "B")}
        ci["r"] = [float(np.quantile(b["r"], 0.025)), float(np.quantile(b["r"], 0.975))]
        sl, sl_se = slope(t, p["v"])
        out[key] = {"D": p["D"], "V": p["V"], "B": p["B"], "r": p["r"], "se": se, "ci95": ci,
                    "z_B": p["B"] / se["B"], "p_B": t_sf(p["B"] / se["B"], m - 1),
                    "slope_v": sl, "slope_v_se": sl_se, "p_slope": t_sf(sl / sl_se, m - 2),
                    "d_k": p["d"].tolist(), "v_k": p["v"].tolist()}
    r1 = holm(np.array([out[k]["p_B"] for k in keys]))
    r3 = holm(np.array([out[k]["p_slope"] for k in keys]))
    for key, a, c in zip(keys, r1, r3):
        lo, hi = out[key]["ci95"]["r"]
        out[key]["R1_detected"] = bool(a)
        out[key]["R2"] = ("balanced-dominated" if hi < 0.5 else
                          "one-directional-dominated" if lo > 0.5 else "undetermined")
        out[key]["R3_grows_with_t"] = bool(c)
    return out


# ----------------------------------------------------------------------------------------------- synthetic check
def synth_counts(rng, times, shots, q, p_of_t, e):
    """Particle numbers of one species: one-directional loss q per occupied mode, balanced flips p(t) per mode,
    balanced readout flips e per mode; 14 of 28 modes occupied initially."""
    out = np.empty((times.size, shots), dtype=np.int64)
    for k, t in enumerate(times):
        p = p_of_t(t)
        n = N0 - rng.binomial(N0, q, size=shots)
        n = n - rng.binomial(n, p) + rng.binomial(MODES - n, p)
        n = n - rng.binomial(n, e) + rng.binomial(MODES - n, e)
        out[k] = n
    return out


def truth(q, p, e):
    """Mean shift and variance of the synthetic process for one circuit (Monte Carlo, 500,000 runs)."""
    rng = np.random.default_rng(1)
    big = synth_counts(rng, np.array([0.0]), 500_000, q, lambda t: p, e)[0]
    return big.mean() - N0, big.var()


def synthetic(n_worlds: int = 300) -> dict:
    """Calibration of the estimators and rules on worlds with known processes (run before the commit)."""
    rng = np.random.default_rng(2030)
    times = np.round(np.arange(1, 21) * 0.1, 10)
    worlds = {
        "null_one_directional": dict(q=0.03, p=lambda t: 0.0, e=0.0),
        "balanced_small": dict(q=0.0, p=lambda t: 0.004, e=0.0),
        "mixed_trend": dict(q=0.01, p=lambda t: 0.01 + 0.01 * t, e=0.002),
        "balanced_like_paper": dict(q=0.002, p=lambda t: 0.03, e=0.0015),
    }
    summary = {}
    for name, w in worlds.items():
        D_true, V_true = np.mean([truth(w["q"], w["p"](t), w["e"]) for t in times], axis=0)
        cover = {q: 0 for q in ("D", "V", "B")}
        r1 = r2_bal = r2_one = r3 = 0
        Bs = []
        for _ in range(n_worlds):
            n_up = synth_counts(rng, times, 160, w["q"], w["p"], w["e"])
            n_dn = synth_counts(rng, times, 160, w["q"], w["p"], w["e"])
            res = analyse({("x", "up"): n_up, ("x", "down"): n_dn}, times, rng)["x", "up"]
            Bs.append(res["B"])
            for q, tv in (("D", D_true), ("V", V_true), ("B", V_true - abs(D_true))):
                lo, hi = res["ci95"][q]
                cover[q] += lo <= tv <= hi
            r1 += res["R1_detected"]
            r2_bal += res["R2"] == "balanced-dominated"
            r2_one += res["R2"] == "one-directional-dominated"
            r3 += res["R3_grows_with_t"]
        summary[name] = {"D_true": float(D_true), "V_true": float(V_true), "B_true": float(V_true - abs(D_true)),
                         "B_mean": float(np.mean(Bs)), "coverage": {q: cover[q] / n_worlds for q in cover},
                         "R1_rate": r1 / n_worlds, "R2_balanced_rate": r2_bal / n_worlds,
                         "R2_one_directional_rate": r2_one / n_worlds, "R3_rate": r3 / n_worlds}
        print(name, summary[name], flush=True)
    return summary


# ----------------------------------------------------------------------------------------------- data
def load(U: int):
    t, shots, meta = load_shots(DATA / f"U_{U}" / "shots.h5")
    up, dn = species_masks(meta["qubit_order"])
    return t, shots[..., up].sum(axis=-1), shots[..., dn].sum(axis=-1), meta


def plot(t, res, nn, joint):
    plt = style()
    fig, ax = plt.subplots(1, 3, figsize=(TEXT_W, 2.2), constrained_layout=True)
    # (a) joint histogram of (N_up, N_down), both U and all t
    lo, hi = 9, 19
    H = np.zeros((hi - lo + 1, hi - lo + 1))
    for (nu, nd) in nn:
        for a, b in zip(nu.ravel(), nd.ravel()):
            if lo <= a <= hi and lo <= b <= hi:
                H[b - lo, a - lo] += 1
    im = ax[0].imshow(H / H.sum(), origin="lower", extent=(lo - 0.5, hi + 0.5, lo - 0.5, hi + 0.5), cmap="Blues")
    ax[0].set_xlabel(r"$N_\uparrow$")
    ax[0].set_ylabel(r"$N_\downarrow$")
    ax[0].set_title(f"(a) shots in (14, 14): {joint:.0%}")
    ax[0].grid(False)
    fig.colorbar(im, ax=ax[0], shrink=0.8, label="share of shots")
    # (b) per-circuit mean shift and variance against t
    for (U, s), r in sorted(res.items()):
        c = CAT[0] if U == 0 else CAT[1]
        m = "o" if s == "up" else "s"
        ax[1].plot(t, r["v_k"], m, color=c, ms=3, label=f"$U={U}$, {s}")
        ax[1].plot(t, r["d_k"], m, color=c, ms=3, mfc="none")
    ax[1].axhline(0, color=MUTED, lw=0.6)
    ax[1].set_xlabel(r"$Jt$")
    ax[1].set_ylabel("events per run")
    ax[1].set_title("(b) variance (filled), mean shift (open)")
    ax[1].legend(fontsize=7, ncol=2, loc="center", bbox_to_anchor=(0.5, 0.315), handletextpad=0.2, columnspacing=0.8)
    # (c) pooled decomposition
    keys = sorted(res)
    x = np.arange(len(keys))
    B = [res[k]["B"] for k in keys]
    Dabs = [abs(res[k]["D"]) for k in keys]
    eb = [[res[k]["B"] - res[k]["ci95"]["B"][0] for k in keys], [res[k]["ci95"]["B"][1] - res[k]["B"] for k in keys]]
    ax[2].bar(x - 0.2, B, 0.4, color=CAT[2], yerr=eb, capsize=2, label="balanced excess $B$")
    ax[2].bar(x + 0.2, Dabs, 0.4, color=INK, label="|mean shift| $|D|$")
    ax[2].set_xticks(x, [f"U={U}\n{s}" for U, s in keys])
    ax[2].set_ylabel("events per run")
    ax[2].set_title("(c) decomposition")
    ax[2].set_ylim(0, 1.75)
    ax[2].legend(fontsize=7, loc="upper left", ncol=1)
    save_figure(fig, "e30_digital_sector")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="calibration on synthetic worlds (before the commit)")
    args = ap.parse_args()
    if args.synthetic:
        summary = synthetic()
        save_json(RESULTS / "e30_synthetic_calibration.json", summary)
        return
    rng = np.random.default_rng(SEED)
    n_by_key, nn, metas, t_ref = {}, [], {}, None
    for U in (0, 4):
        t, nu, nd, meta = load(U)
        if t_ref is None:
            t_ref = t
        assert np.allclose(t, t_ref)
        n_by_key[(U, "up")], n_by_key[(U, "down")] = nu, nd
        nn.append((nu, nd))
        metas[U] = {k: v for k, v in meta.items() if k != "qubit_order"}
    in_sector = np.concatenate([((nu == N0) & (nd == N0)).ravel() for nu, nd in nn])
    joint = float(in_sector.mean())
    s0 = 0.17 <= joint <= 0.21
    print(f"S0: pooled share in (14, 14) = {joint:.4f} -> {'pass' if s0 else 'FAIL (stop)'}")
    if not s0:
        save_json(RESULTS / "e30_digital_sector.json", {"S0_share_14_14": joint, "S0_pass": False})
        return
    res = analyse(n_by_key, t_ref, rng)
    descriptive = {}
    for U, (nu, nd) in zip((0, 4), nn):
        descriptive[f"U={U}"] = {
            "cov_up_down_pooled_within_circuit": float(np.mean([np.cov(a, b)[0, 1] for a, b in zip(nu, nd)])),
            "share_14_14_per_t": [float(np.mean((a == N0) & (b == N0))) for a, b in zip(nu, nd)],
        }
    for key in sorted(res):
        r = res[key]
        print(f"U={key[0]} {key[1]:>4}: D={r['D']:+.3f} [{r['ci95']['D'][0]:+.3f},{r['ci95']['D'][1]:+.3f}]  "
              f"V={r['V']:.3f}  B={r['B']:.3f} [{r['ci95']['B'][0]:.3f},{r['ci95']['B'][1]:.3f}] z={r['z_B']:.1f}  "
              f"r={r['r']:.3f} [{r['ci95']['r'][0]:.3f},{r['ci95']['r'][1]:.3f}]  R1={r['R1_detected']}  "
              f"R2={r['R2']}  slope_v={r['slope_v']:+.3f}+-{r['slope_v_se']:.3f} R3={r['R3_grows_with_t']}")
    for k, v in descriptive.items():
        print(k, "cov(N_up, N_down) =", round(v["cov_up_down_pooled_within_circuit"], 4))
    save_json(RESULTS / "e30_digital_sector.json", {
        "data": "Phasecraft and Quantinuum, arXiv:2510.26300; Zenodo 10.5281/zenodo.17881394 (CC BY 4.0)",
        "metadata": metas, "times": t_ref.tolist(), "S0_share_14_14": joint, "S0_pass": s0,
        "results": {f"U={U},{s}": r for (U, s), r in sorted(res.items())},
        "descriptive_post_hoc": descriptive,
    })
    plot(t_ref, res, nn, joint)


if __name__ == "__main__":
    main()
