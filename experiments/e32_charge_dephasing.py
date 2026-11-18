"""E32: the q^2 law of collective dephasing on the dephasing-only runs of the Mpemba experiment (preregistered)

Data: measured by Joshi et al. (PRL 133, 010402 (2024); twelve 40Ca+ ions, Innsbruck), Zenodo 10.5281/zenodo.10390690
(CC BY 4.0), analysed here from the public raw shots (data/fetch_mpemba2024.py). Same group as Kraft et al., but a
different run, ion number and protocol.
Runs used: "pure dephasing" (paper, Fig. 3 and its text: the spin-spin interaction is turned off and the tilted
ferromagnet evolves freely under the laboratory noise). Tilt angles 0.2 pi, 0.33 pi, 0.5 pi; times 0, 1, ..., 8 ms;
for each time 300 random single-qubit unitaries (CUE) x 30 shots. The unitaries are the first 300 of the published
500 (t = 0 check, research note Q56: correlation 0.96 between predicted and measured t = 0 bit frequencies at all
three angles; state measured u rho u^dag; initial Bloch vectors in the x-z plane, azimuth about 5 degrees).
Nothing beyond the t = 0 rows of the data had been looked at before this commit.

Question (manuscript: Lemma "collective damping"; Theorem 5). Collective dephasing multiplies the charge-q blocks of
the state by a factor whose exponent is proportional to q^2. Independent dephasing of each ion damps a coherence
according to the number of ions it involves instead (its Hamming weight). For local coherences:
  charge 1  sigma^+_i                       collective: q^2 = 1   independent: 1
  charge 2  sigma^+_i sigma^+_j             collective: 4         independent: 2
  charge 0  sigma^+_i sigma^-_j             collective: 0         independent: 2
  charge 3  sigma^+_i sigma^+_j sigma^+_k   collective: 9         independent: 3
A static field inhomogeneity only rotates these coherences and leaves their moduli unchanged. With a collective share
f (the exponent of charge 1 split as f collective and 1 - f independent), the exponents relative to charge 1 are
r_q = q^2 f + h_q (1 - f) with h = (2, 1, 2, 3) for q = (0, 1, 2, 3): r_0 = 2 - 2f, r_2 = 2 + 2f, r_3 = 3 + 6f. The
ratios hold at every time and do not depend on whether the noise is white or quasi-static (both give Gaussian phases;
the time dependence belongs to Theorem G and the PRX paper).

Estimators (fixed).
  Exact inversion (qerr.shadows.exact_duals): for every qubit set the least-squares inverse of the measurement channel
  of the actual unitary set, so the estimate of <O> is unbiased for this fixed set (the textbook snapshot
  3 u^dag|b><b|u - 1 is unbiased only on average over unitary draws; see the choices below).
  Classes over all 12 ions: q = 1 single ions (12), q = 2 (sigma^+ sigma^+) and q = 0 (sigma^+ sigma^-) pairs (66 each),
  q = 3 triples (220).
  Squared modulus, primary ("split"): the units are split into halves 1-150 and 151-300, each half is inverted exactly
  on its own, and |<O>|^2 is estimated by Re(o_A conj(o_B)). It does not assume that shots of one unitary, or of
  unitaries taken close in time, are independent. Alternative ("full", descriptive): all 300 units, |o|^2 minus the
  within-unit variance.
  Power P_q(t) = sum of |<O>|^2 over the class; decay D_q(t) = P_q(t) / P_q(0); clock kappa(t) = -ln D_1(t).
  Ratio r_q (q = 0, 2, 3) per angle: the value minimising sum_{t = 1..8 ms} (D_q(t) - exp(-r kappa(t)))^2 / s_q(t)^2,
  with s_q(t) the bootstrap standard deviation of D_q(t) (fixed from the full data), r in [-5, 40].
  Uncertainty: bootstrap over unitaries within each half (multiplicities, duals fixed), B = 2000, seed 32; every
  replicate recomputes all D_q(t) and refits all r_q with the same weights. SE = bootstrap sd.
  Collective share: generalised least squares of the nine ratios (3 angles x r_0, r_2, r_3) on r(f) with the
  bootstrap covariance of each angle (angles independent): f_hat, SE(f) and chi^2 with 8 degrees of freedom.
Rules fixed before the run.
  P0 (pipeline; stop if failed): for each angle, the largest correlation between predicted and measured t = 0 bit
      frequencies (scan of the Bloch azimuth and of the sign of its z component) is at least 0.9.
  R1 (collective component): for each angle, z = (r_2 - 2) / SE(r_2), one-sided normal p, Holm-Bonferroni at
      familywise 5% over the three angles. Independent dephasing alone gives r_2 = 2.
  R2 (protected charge-0 coherences): for each angle, z = (2 - r_0) / SE(r_0), one-sided, Holm at 5% over three.
  R3 (charge 3 beyond its Hamming weight): for each angle, z = (r_3 - 3) / SE(r_3), one-sided, Holm at 5% over three.
  R4 (two-component structure): chi^2 of the pooled fit with 8 degrees of freedom; "consistent" if p >= 0.01.
  R5 (character): with f_hat +- 1.96 SE(f): "collective-dominated" if the lower end exceeds 0.8, "independent-dominated"
      if the upper end is below 0.2, "mixed" if both ends lie in [0.2, 0.8], otherwise "undetermined".
  Descriptive only (no verdicts): D_q(t) curves; the ratios of the "full" estimator; the power-law exponent of kappa(t)
  (1 for white, 2 for quasi-static noise).
Choices made before this commit, on synthetic data only (`--synthetic`, results/e32_synthetic_calibration.json).
  Synthetic shots use the actual 300 unitaries, 30 shots each, the three tilts with per-ion tilt and azimuth errors of
  0.03 rad, static detunings of about 50 Hz, readout flips of 0.2%, and Gaussian phase noise: collective, independent
  per ion, or a gradient along the chain.
  - First design (textbook snapshots 3 u^dag|b><b|u - 1, squared moduli from pairs of shots with different unitaries).
    In a clean collective world the ratios came out (r_0, r_2, r_3) = (0.13-0.39, 3.70-3.81, 7.6-8.0) instead of
    (0, 4, 9), and f = 0.89 +- 0.02 instead of 1: the decay of charge 0 was 0.90 at 8 ms instead of 1. With a fresh
    random unitary for every shot (9,000 unitaries) the bias vanished (charge 0 stayed at 1.00-1.03). Cause: with one
    fixed set of 300 unitaries the measurement channel is not exactly depolarising, so the textbook inverse mixes
    Bloch components of different charge; the mixing is the same at every time and biases decay ratios.
  - Fix: exact (least-squares) inversion of the channel of the actual set for every qubit set (tests in
    tests/test_shadows.py: exact recovery of <O> for correlated states with a fixed set of 100 unitaries). With it the
    same clean worlds gave r = (0.0-0.12, 3.96-4.09, 8.5-9.5), f = 0.99 +- 0.01 (collective) and r = (2.0, 1.94-2.03,
    2.9-3.1), f = 0.01 +- 0.01 (independent).
  - Under exact inversion the expected contribution differs from unitary to unitary, so pairs of shots from different
    unitaries no longer give an unbiased squared modulus. The primary estimator is therefore the split-half product;
    the variance-subtracted full estimator is kept as a descriptive alternative (it assumes independent shots within
    a unitary, which slow drifts could violate).
  - Calibration of the final procedure (split estimator, 20 realisations per world, B = 300). Columns: mean
    (r_0, r_2, r_3); f mean (sd over runs; mean SE); rates of R1, R2, R3 (per angle, after Holm); R4 "consistent"
    rate; R5.
      collective white, truth (0, 4, 9):   (0.009, 3.999, 9.23); 0.995 (0.008; 0.026); 1, 1, 0.67; 1.00; collective 1.00
      collective quasi-static:             (-0.005, 4.03, 9.39); 1.002 (0.011; 0.027); 1, 1, 0.68; 1.00; collective 1.00
      independent white, truth (2, 2, 3):  (2.006, 2.000, 3.04); -0.002 (0.007; 0.009); 0, 0, 0; 1.00; independent 1.00
      mixed (collective quasi-static + independent white, f grows in time):
                                           (1.12, 2.83, 5.18); 0.44 (0.013; 0.018); 1, 1, 0.67; 0.95; mixed 1.00
      collective weak, kappa(8 ms) = 0.3:  (0.02, 3.99, 9.21); 0.97 (0.029; 0.091); 1, 1, 0.67; 1.00;
                                           collective 0.45, undetermined 0.55
      collective strong, kappa(8 ms) = 3:  (0.004, 4.006, 12.6); 0.999 (0.004; 0.008); 1, 1, 0.18; 1.00; collective 1.00
      collective + gradient along the chain (breaks the two-component structure):
                                           (0.73, 2.98, 6.22); 0.57 (0.017; 0.022); 1, 1, 0.67; 0.00; mixed 1.00
    Reading. r_0 and r_2 are unbiased in every world; r_3 is unbiased while charge 3 keeps a signal and becomes noisy
    when it decays away (strong world). The bootstrap SE (units resampled, duals fixed) is conservative, 2.5-3 times
    the spread over runs in the collective worlds, with coverage 1.00 for every ratio and for f: a resampled unit set
    with fixed duals re-creates part of the sector mixing that the exact inverse removes. The rules are therefore
    conservative: no false detection in the independent world, R4 rejects the gradient world in every run and the
    pure worlds in none (mixed: 5%), and R5 loses power only when the decay is weak.
  - Kept: the conservative bootstrap. A shot-level bootstrap would be accurate only if the 30 shots of a unitary are
    independent, which slow drifts could violate; recomputing the exact inverse in every unit-level replicate is too
    slow for this calibration.
"""

import argparse
import itertools
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, norm

from common import CAT, INK, RESULTS, ROOT, save_figure, save_json, style

sys.path.insert(0, str(ROOT / "src"))
from qerr.mpembadata import load_dephasing, load_unitaries  # noqa: E402
from qerr.shadows import bloch_rows, exact_duals, set_values, shot_paulis  # noqa: E402

RAW = ROOT / "data" / "raw" / "mpemba2024"
ANGLES = (("0p2", 0.2), ("0p33", 0.33), ("0p5", 0.5))
N_Q, N_U, N_SHOT = 12, 300, 30
HALF = N_U // 2
CLASSES = (1, 2, 0, 3)
SETS = {1: [(i,) for i in range(N_Q)], 2: list(itertools.combinations(range(N_Q), 2)),
        0: list(itertools.combinations(range(N_Q), 2)), 3: list(itertools.combinations(range(N_Q), 3))}
OPS = {1: "+", 2: "++", 0: "+-", 3: "+++"}
M0 = np.array([2.0, 2.0, 3.0])   # (r_0, r_2, r_3) at f = 0
G = np.array([-2.0, 2.0, 6.0])   # d r / d f
ESTIMATORS = ("split", "full")
TEXT_W = 510 / 72


# ----------------------------------------------------------------------------------------------- estimators
class Duals:
    """Exact-inverse dual vectors for the full unitary set and for its two halves (units 1-150 and 151-300)."""

    def __init__(self, U: np.ndarray):
        self.m = bloch_rows(U)
        self.full = {q: exact_duals(self.m, SETS[q], [OPS[q]] * len(SETS[q])) for q in CLASSES}
        self.A = {q: exact_duals(self.m[:HALF], SETS[q], [OPS[q]] * len(SETS[q])) for q in CLASSES}
        self.B = {q: exact_duals(self.m[HALF:], SETS[q], [OPS[q]] * len(SETS[q])) for q in CLASSES}


def unit_stats(duals: Duals, bits_t: np.ndarray, with_full: bool = True) -> dict:
    """Per-unit sums for one time: split halves (SA, SB) and, if requested, the full set (S, and V for the variance
    subtraction)."""
    p = shot_paulis(duals.m, bits_t)
    out = {}
    for q in CLASSES:
        xA = set_values(duals.A[q], p[:HALF], SETS[q], N_SHOT)
        xB = set_values(duals.B[q], p[HALF:], SETS[q], N_SHOT)
        out[q] = {"SA": xA.sum(axis=1), "SB": xB.sum(axis=1)}
        if with_full:
            x = set_values(duals.full[q], p, SETS[q], N_SHOT)
            out[q]["S"] = x.sum(axis=1)
            out[q]["V"] = N_SHOT / (N_SHOT - 1) * np.sum(np.abs(x - x.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return out


def class_power(st: dict, estimator: str, W=None) -> np.ndarray:
    """Sum over the class of the |<O>|^2 estimates; W (B, units) bootstrap multiplicities or None."""
    if estimator == "split":
        if W is None:
            return np.array(np.sum(np.real(st["SA"].sum(axis=0) * np.conj(st["SB"].sum(axis=0)))))
        oA, oB = W[:, :HALF] @ st["SA"], W[:, HALF:] @ st["SB"]
        return np.sum(np.real(oA * np.conj(oB)), axis=1)
    if W is None:
        return np.array(np.sum(np.abs(st["S"].sum(axis=0)) ** 2 - st["V"].sum(axis=0)))
    return np.sum(np.abs(W @ st["S"]) ** 2 - (W**2) @ st["V"], axis=1)


def boot_weights(rng, n_boot):
    """Multinomial multiplicities drawn separately within each half (each half keeps 150 units)."""
    WA = rng.multinomial(HALF, np.full(HALF, 1.0 / HALF), size=n_boot)
    WB = rng.multinomial(N_U - HALF, np.full(N_U - HALF, 1.0 / (N_U - HALF)), size=n_boot)
    return np.hstack([WA, WB]).astype(float)


def fit_ratio(D1: np.ndarray, Dq: np.ndarray, s: np.ndarray) -> float:
    kappa = -np.log(np.clip(D1, 1e-12, None))
    obj = lambda r: np.sum(((Dq - np.exp(-r * kappa)) / s) ** 2)  # noqa: E731
    return float(minimize_scalar(obj, bounds=(-5.0, 40.0), method="bounded", options={"xatol": 1e-6}).x)


def analyse_angle(duals: Duals, bits: np.ndarray, rng, n_boot: int, estimators=ESTIMATORS) -> dict:
    """Decays, ratios and bootstrap for one angle (same data and bootstrap draws for every estimator)."""
    stats = [unit_stats(duals, bits[t], with_full="full" in estimators) for t in range(bits.shape[0])]
    W = boot_weights(rng, n_boot)
    out = {}
    late = slice(1, None)
    for est in estimators:
        P = {q: np.array([class_power(st[q], est) for st in stats]) for q in CLASSES}
        Pb = {q: np.array([class_power(st[q], est, W) for st in stats]) for q in CLASSES}
        D = {q: P[q] / P[q][0] for q in CLASSES}
        Db = {q: Pb[q] / Pb[q][0][None, :] for q in CLASSES}
        s = {q: np.maximum(Db[q].std(axis=1, ddof=1), 1e-6) for q in CLASSES}
        r = {q: fit_ratio(D[1][late], D[q][late], s[q][late]) for q in (0, 2, 3)}
        rb = {q: np.array([fit_ratio(Db[1][late, b], Db[q][late, b], s[q][late]) for b in range(n_boot)])
              for q in (0, 2, 3)}
        Rb = np.stack([rb[0], rb[2], rb[3]], axis=1)
        out[est] = {"D": D, "D_sd": s, "r": r, "r_se": {q: float(rb[q].std(ddof=1)) for q in (0, 2, 3)},
                    "R": np.array([r[0], r[2], r[3]]), "cov": np.cov(Rb.T), "P0": {q: float(P[q][0]) for q in CLASSES}}
    return out


def holm(p, alpha=0.05):
    order = np.argsort(p)
    ok = np.zeros(len(p), dtype=bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (len(p) - rank):
            ok[i] = True
        else:
            break
    return ok


def verdicts(per_angle: dict) -> dict:
    """per_angle: {tag: analysis of one estimator}."""
    keys = list(per_angle)
    z1 = np.array([(per_angle[k]["r"][2] - 2) / per_angle[k]["r_se"][2] for k in keys])
    z2 = np.array([(2 - per_angle[k]["r"][0]) / per_angle[k]["r_se"][0] for k in keys])
    z3 = np.array([(per_angle[k]["r"][3] - 3) / per_angle[k]["r_se"][3] for k in keys])
    r1, r2, r3 = (holm(norm.sf(z)) for z in (z1, z2, z3))
    inv = {k: np.linalg.inv(per_angle[k]["cov"]) for k in keys}
    A = sum(G @ inv[k] @ G for k in keys)
    bvec = sum(G @ inv[k] @ (per_angle[k]["R"] - M0) for k in keys)
    f = float(bvec / A)
    se_f = float(1 / np.sqrt(A))
    c2 = float(sum((per_angle[k]["R"] - M0 - f * G) @ inv[k] @ (per_angle[k]["R"] - M0 - f * G) for k in keys))
    p_c2 = float(chi2.sf(c2, 3 * len(keys) - 1))
    lo, hi = f - 1.96 * se_f, f + 1.96 * se_f
    r5 = ("collective-dominated" if lo > 0.8 else "independent-dominated" if hi < 0.2
          else "mixed" if (0.2 <= lo and hi <= 0.8) else "undetermined")
    return {"z_R1": z1.tolist(), "R1": r1.tolist(), "z_R2": z2.tolist(), "R2": r2.tolist(),
            "z_R3": z3.tolist(), "R3": r3.tolist(), "f": f, "se_f": se_f, "chi2": c2, "p_chi2": p_c2,
            "R4_consistent": p_c2 >= 0.01, "R5": r5}


# ----------------------------------------------------------------------------------------------- synthetic worlds
def synth_bits(rng, m, theta, sign, var_c, var_l, var_g, times, detune, tilt_err=0.03, readout=0.002):
    """Shots of a tilted product state under Gaussian phase noise: collective (all ions), independent (per ion) and a
    gradient (proportional to the ion position); static detunings rotate each ion. m: Bloch rows of the unitaries.
    Returns bits (times, 300, 30, 12)."""
    x = (np.arange(N_Q) - 5.5) / 5.5
    th = theta * np.pi + tilt_err * rng.normal(size=N_Q)
    phi0 = np.deg2rad(5.0) + tilt_err * rng.normal(size=N_Q)
    r0 = np.stack([np.sin(th) * np.cos(phi0), np.sin(th) * np.sin(phi0), sign * np.cos(th)], axis=-1)
    out = np.empty((len(times), N_U, N_SHOT, N_Q), dtype=np.uint8)
    for k, t in enumerate(times):
        shape = (N_U, N_SHOT)
        alpha = (np.sqrt(var_c(t)) * rng.normal(size=shape)[..., None]
                 + np.sqrt(var_l(t)) * rng.normal(size=shape + (N_Q,))
                 + np.sqrt(var_g(t)) * rng.normal(size=shape)[..., None] * x
                 + detune * t)
        ca, sa = np.cos(alpha), np.sin(alpha)
        rx = r0[:, 0] * ca - r0[:, 1] * sa
        ry = r0[:, 0] * sa + r0[:, 1] * ca
        ez = m[:, None, :, 0] * rx + m[:, None, :, 1] * ry + m[:, None, :, 2] * r0[:, 2]
        bits = (rng.random(ez.shape) < (1 - ez) / 2).astype(np.uint8)
        out[k] = bits ^ (rng.random(bits.shape) < readout)
    return out


WORLDS = {
    "collective_white": dict(c=lambda t: t / 8, l=lambda t: 0.0, g=lambda t: 0.0, f=1.0),
    "collective_quasistatic": dict(c=lambda t: (t / 8) ** 2, l=lambda t: 0.0, g=lambda t: 0.0, f=1.0),
    "independent_white": dict(c=lambda t: 0.0, l=lambda t: t / 8, g=lambda t: 0.0, f=0.0),
    "mixed_qs_collective_white_independent": dict(c=lambda t: 0.5 * (t / 8) ** 2, l=lambda t: 0.5 * t / 8,
                                                  g=lambda t: 0.0, f=None),
    "collective_weak": dict(c=lambda t: 0.3 * t / 8, l=lambda t: 0.0, g=lambda t: 0.0, f=1.0),
    "collective_strong": dict(c=lambda t: 3.0 * t / 8, l=lambda t: 0.0, g=lambda t: 0.0, f=1.0),
    "collective_plus_gradient": dict(c=lambda t: 0.5 * t / 8, l=lambda t: 0.0, g=lambda t: 1.0 * t / 8, f=None),
}


def synthetic(n_real: int = 20, n_boot: int = 300, estimators=("split",)) -> dict:
    U = load_unitaries(RAW)[:N_U]
    duals = Duals(U)
    times = np.arange(9.0)
    rng = np.random.default_rng(2032)
    summary = {}
    for name, w in WORLDS.items():
        rows = {est: [] for est in estimators}
        for _ in range(n_real):
            detune = 2 * np.pi * 0.05 * rng.normal(size=N_Q)  # static detunings, rad per ms (about 50 Hz)
            per = {est: {} for est in estimators}
            for tag, theta in ANGLES:
                sign = 1.0 if theta == 0.5 else -1.0
                bits = synth_bits(rng, duals.m, theta, sign, w["c"], w["l"], w["g"], times, detune)
                res = analyse_angle(duals, bits, rng, n_boot, estimators)
                for est in estimators:
                    per[est][tag] = res[est]
            for est in estimators:
                v = verdicts(per[est])
                rows[est].append({"r0": [per[est][k]["r"][0] for k in per[est]],
                                  "r2": [per[est][k]["r"][2] for k in per[est]],
                                  "r3": [per[est][k]["r"][3] for k in per[est]],
                                  "se": [[per[est][k]["r_se"][q] for q in (0, 2, 3)] for k in per[est]],
                                  "f": v["f"], "se_f": v["se_f"], "R1": v["R1"], "R2": v["R2"], "R3": v["R3"],
                                  "R4": v["R4_consistent"], "R5": v["R5"]})
        summary[name] = {}
        for est in estimators:
            rr = rows[est]
            r_all = np.array([[x["r0"], x["r2"], x["r3"]] for x in rr])      # (real, 3 ratios, 3 angles)
            se_all = np.array([x["se"] for x in rr])                          # (real, 3 angles, 3 ratios)
            res = {"mean_r0_r2_r3": r_all.mean(axis=(0, 2)).round(4).tolist(),
                   "sd_r_over_runs": r_all.std(axis=0, ddof=1).mean(axis=1).round(4).tolist(),
                   "mean_se_r": se_all.mean(axis=(0, 1)).round(4).tolist(),
                   "mean_f": float(np.mean([x["f"] for x in rr])), "sd_f": float(np.std([x["f"] for x in rr], ddof=1)),
                   "mean_se_f": float(np.mean([x["se_f"] for x in rr])),
                   "R1_rate": float(np.mean([np.mean(x["R1"]) for x in rr])),
                   "R2_rate": float(np.mean([np.mean(x["R2"]) for x in rr])),
                   "R3_rate": float(np.mean([np.mean(x["R3"]) for x in rr])),
                   "R4_consistent_rate": float(np.mean([x["R4"] for x in rr])),
                   "R5": {c: float(np.mean([x["R5"] == c for x in rr])) for c in
                          ("collective-dominated", "independent-dominated", "mixed", "undetermined")}}
            if w["f"] is not None:
                truth = np.array([2 - 2 * w["f"], 2 + 2 * w["f"], 3 + 6 * w["f"]])
                cover = (np.abs(r_all - truth[None, :, None]) <= 1.96 * np.transpose(se_all, (0, 2, 1)))
                res["coverage_r0_r2_r3"] = cover.mean(axis=(0, 2)).round(3).tolist()
                res["coverage_f"] = float(np.mean([abs(x["f"] - w["f"]) <= 1.96 * x["se_f"] for x in rr]))
            summary[name][est] = res
        print(name, summary[name], flush=True)
    return summary


# ----------------------------------------------------------------------------------------------- data
def t0_check(m, bits0, theta):
    """Largest correlation of predicted and measured t = 0 bit-1 frequencies over Bloch azimuth and z sign."""
    f1 = bits0.mean(axis=1)  # (units, 12)
    best = -1.0
    for sign in (1.0, -1.0):
        for phi in np.linspace(0, 2 * np.pi, 145)[:-1]:
            rvec = np.array([np.sin(theta * np.pi) * np.cos(phi), np.sin(theta * np.pi) * np.sin(phi),
                             sign * np.cos(theta * np.pi)])
            pred = (1 - m @ rvec) / 2
            best = max(best, float(np.corrcoef(pred.ravel(), f1.ravel())[0, 1]))
    return best


def plot(times, per):
    plt = style()
    fig, ax = plt.subplots(1, 3, figsize=(TEXT_W, 2.3), constrained_layout=True)
    marks = {1: ("o", CAT[0], "charge 1"), 2: ("s", CAT[1], "charge 2"), 3: ("^", CAT[2], "charge 3"),
             0: ("D", INK, "charge 0")}
    tag = "0p5"
    for q in CLASSES:
        m, c, lab = marks[q]
        D = per[tag]["D"][q]
        ok = D > 0
        ax[0].errorbar(times[ok], D[ok], yerr=per[tag]["D_sd"][q][ok], fmt=m, color=c, ms=3, capsize=1.5, label=lab)
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-3, 2)
    ax[0].set_xlabel("time (ms)")
    ax[0].set_ylabel(r"$D_q(t)$")
    ax[0].set_title(r"(a) decay, tilt $0.5\pi$")
    ax[0].legend(fontsize=7, loc="lower left")
    kmax = 0.0
    for i, (tg, _) in enumerate(ANGLES):
        kappa = -np.log(np.clip(per[tg]["D"][1], 1e-12, None))
        kmax = max(kmax, float(kappa[1:].max()))
        for q in (0, 2, 3):
            m, c, lab = marks[q]
            Dq = per[tg]["D"][q]
            ok = Dq[1:] > 0
            ax[1].plot(kappa[1:][ok], -np.log(Dq[1:][ok]), m, color=c, ms=3, alpha=0.45 + 0.25 * i,
                       label=lab if i == 2 else None)
    kk = np.linspace(0, kmax * 1.05 if kmax > 0 else 1, 50)
    for q, sl in ((0, 0), (2, 4), (3, 9)):
        ax[1].plot(kk, sl * kk, "--", color=marks[q][1], lw=0.8)
    ax[1].set_xlabel(r"$-\ln D_1$")
    ax[1].set_ylabel(r"$-\ln D_q$")
    ax[1].set_title(r"(b) dashed: $q^2$ law")
    ax[1].legend(fontsize=7, loc="upper left")
    x = np.arange(3)
    for k, q in enumerate((0, 2, 3)):
        m, c, lab = marks[q]
        vals = [per[tg]["r"][q] for tg, _ in ANGLES]
        errs = [1.96 * per[tg]["r_se"][q] for tg, _ in ANGLES]
        ax[2].errorbar(x + 0.2 * (k - 1), vals, yerr=errs, fmt=m, color=c, ms=4, capsize=2, label=lab)
        ax[2].axhline({0: 0, 2: 4, 3: 9}[q], color=c, lw=0.8, ls="--")
        ax[2].axhline({0: 2, 2: 2, 3: 3}[q], color=c, lw=0.8, ls=":")
    ax[2].set_xticks(x, [r"$0.2\pi$", r"$0.33\pi$", r"$0.5\pi$"])
    ax[2].set_xlabel("tilt angle")
    ax[2].set_ylabel(r"exponent ratio $r_q$")
    ax[2].set_ylim(-1, 15)
    ax[2].set_title("(c) ratios to charge 1")
    save_figure(fig, "e32_charge_dephasing")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    if args.synthetic:
        save_json(RESULTS / "e32_synthetic_calibration.json", synthetic())
        return
    U = load_unitaries(RAW)[:N_U]
    duals = Duals(U)
    rng = np.random.default_rng(32)
    per, p0, desc = {}, {}, {}
    times = None
    for tag, theta in ANGLES:
        times, bits = load_dephasing(RAW, tag)
        p0[tag] = t0_check(duals.m, bits[0], theta)
        print(f"P0 {tag}: t = 0 correlation {p0[tag]:.3f}", flush=True)
        if p0[tag] < 0.9:
            save_json(RESULTS / "e32_charge_dephasing.json", {"P0": p0, "P0_pass": False})
            print("P0 failed: stop")
            return
        res = analyse_angle(duals, bits, rng, 2000)
        per[tag] = res["split"]
        full = res["full"]
        kappa = -np.log(np.clip(per[tag]["D"][1][1:], 1e-12, None))
        ok = kappa > 0
        desc[tag] = {"full_estimator_r": {str(q): full["r"][q] for q in (0, 2, 3)},
                     "full_estimator_r_se": {str(q): full["r_se"][q] for q in (0, 2, 3)},
                     "kappa_power_law_exponent": (float(np.polyfit(np.log(times[1:][ok]), np.log(kappa[ok]), 1)[0])
                                                  if ok.sum() >= 3 else None),
                     "P_at_t0": per[tag]["P0"]}
        r, se = per[tag]["r"], per[tag]["r_se"]
        print(f"  r0={r[0]:.3f}+-{se[0]:.3f}  r2={r[2]:.3f}+-{se[2]:.3f}  r3={r[3]:.3f}+-{se[3]:.3f}  "
              f"D1(8ms)={per[tag]['D'][1][-1]:.3f}  full: {desc[tag]['full_estimator_r']}  "
              f"kappa exponent {desc[tag]['kappa_power_law_exponent']}", flush=True)
    v = verdicts(per)
    print("verdicts", v)
    save_json(RESULTS / "e32_charge_dephasing.json", {
        "data": "Joshi et al., PRL 133, 010402 (2024); Zenodo 10.5281/zenodo.10390690 (CC BY 4.0)",
        "times_ms": times.tolist(), "P0_t0_correlation": p0, "P0_pass": True, "estimator": "split",
        "per_angle": {tag: {"D": {str(q): per[tag]["D"][q].tolist() for q in CLASSES},
                            "D_sd": {str(q): per[tag]["D_sd"][q].tolist() for q in CLASSES},
                            "r": {str(q): per[tag]["r"][q] for q in (0, 2, 3)},
                            "r_se": {str(q): per[tag]["r_se"][q] for q in (0, 2, 3)},
                            "cov_r0_r2_r3": per[tag]["cov"].tolist()} for tag, _ in ANGLES},
        "verdicts": v, "descriptive_post_hoc": desc})
    plot(np.asarray(times), per)


if __name__ == "__main__":
    main()
