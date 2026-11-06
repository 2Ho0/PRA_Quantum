"""E19: is the incoherent flip rate of the 10-ion data the same as the charge-changing rate of the 51-ion data?

Data: both BEQS data sets (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419,
CC BY 4.0), analysed here from the public raw shots.
- 51 ions (Neel state, z basis): the balanced sector-changing rate per ion is r = (d Var M/dt - |d<M>/dt|)/51
  (E6b, E9). Every flip changes M by +-1, so Var M grows by the total flip rate; the D5/2 decays are the |d<M>/dt|
  part. For isotropic flips (Kossakowski S_XX + S_YY = S) every ion flips at rate S, so r estimates S.
- 10 ions: the mean-Z balance of E17, d<Zbar>/dt = 2 h_x <Ybar> - 2 h_y <Xbar> - 2 S <Zbar> - e (1 + <Zbar>) + c, with
  the D5/2 emission e = 1/1.168 s fixed, uniform fields h, flip rate S and a constant pumping c; both initial
  states, integral form with spline quadrature, errors from 200 bootstrap replicates over the settings. Windows:
  t <= 1.5 ms (learning window; fields, S, c free), the whole record (fields, S, c free; E17), and t >= 3 ms (the
  transverse polarisation has decayed below 0.05; S and c free, fields fixed at the whole-record values or at 0).
Fixed before the run: the 10-ion late-window S is consistent with r if they differ by less than 2 combined standard
errors (bootstrap for 10 ions, slope errors for 51 ions). The other windows are reported for comparison, with E11e's
bias-corrected flip rate of the noise-block fit (t <= 1.5 ms).
"""

import sys

import numpy as np
from scipy.interpolate import CubicSpline

from common import CAT, INK, MUTED, RESULTS, save_figure, save_json, style
from e6_ion_learning import N, RAD_S, per_setting_tables, series_from
from e6c_validation import load

EMISSION = 1.0 / 1168.0  # 1/ms
N_BOOT, T_LATE = 200, 3.0
_SINGLES = [((i, a),) for i in range(N) for a in "XYZ"]


def means(series):
    """(state, time, 3) site means of X, Y, Z."""
    return np.stack([np.stack([np.mean([series[(s, ((i, a),))] for i in range(N)], axis=0) for a in "XYZ"], -1)
                     for s in (0, 1)])


def fit_window(t, m, lo, hi, fields=None):
    """Least squares for (h_x, h_y, S, c) (fields=None) or (S, c) with fixed fields, on lo <= t <= hi.

    Returns the parameters in rad/s and 1/s. Zbar(t) - Zbar(lo) = int [2 h_x Ybar - 2 h_y Xbar - 2 S Zbar
    - e (1 + Zbar) + c]. Equations are weighted equally (the bootstrap gives the errors).
    """
    sel = (t >= lo - 1e-9) & (t <= hi + 1e-9)
    tt = t[sel]
    rows, rhs = [], []
    for s in (0, 1):
        X, Y, Z = (m[s, sel, a] for a in range(3))
        cum = lambda f: CubicSpline(tt, f).antiderivative()(tt)
        target = Z - Z[0] + EMISSION * cum(1.0 + Z)
        regs = [cum(-2.0 * Z), cum(np.ones_like(Z))]
        if fields is None:
            regs = [cum(2.0 * Y), cum(-2.0 * X)] + regs
        else:
            target = target - cum(2.0 * fields[0] * Y - 2.0 * fields[1] * X)
        rows.append(np.column_stack(regs)[1:])
        rhs.append(target[1:])
    p, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)
    return p * RAD_S


def estimates(t, m):
    full = fit_window(t, m, 0.0, t[-1])
    learn = fit_window(t, m, 0.0, 1.5)
    late_f = fit_window(t, m, T_LATE, t[-1], fields=full[:2] / RAD_S)
    late_0 = fit_window(t, m, T_LATE, t[-1], fields=(0.0, 0.0))
    return np.r_[learn, full, late_f, late_0]  # 4 + 4 + 2 + 2


LABELS = (["learning window: h_x", "learning window: h_y", "learning window: S", "learning window: c"]
          + ["whole record: h_x", "whole record: h_y", "whole record: S", "whole record: c"]
          + ["t >= 3 ms, fields from the whole record: S", "t >= 3 ms, fields from the whole record: c"]
          + ["t >= 3 ms, no fields: S", "t >= 3 ms, no fields: c"])


def run():
    times, s1, s2, bases = load()
    tables = per_setting_tables((s1, s2), bases, _SINGLES)
    m = means(series_from(tables))
    rng = np.random.default_rng(2043)
    boot = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, 200, 200), minlength=200).astype(float)
        boot.append(estimates(times, means(series_from(tables, w))))
    return times, m, estimates(times, m), np.array(boot)


def analyse(times, m, est, boot):
    e6b = __import__("json").loads((RESULTS / "e6_symmetry.json").read_text(encoding="utf-8"))["acceptance"]
    ms, sms = e6b["51 ions: slope of M per ms (+- 1 se)"]
    vs, svs = e6b["51 ions: slope of varM per ms (+- 1 se)"]
    r51, sr51 = (vs - abs(ms)) / 51 * 1e3, np.hypot(svs, sms) / 51 * 1e3
    se = boot.std(axis=0, ddof=1)
    out = {"10 ions: estimate and bootstrap sd": {lab: [float(e), float(s)] for lab, e, s in zip(LABELS, est, se)},
           "51 ions: balanced sector-changing rate per ion (1/s) and se": [float(r51), float(sr51)]}
    e11e = __import__("json").loads((RESULTS / "e11e_noise_block_sbc.json").read_text(encoding="utf-8"))
    row = e11e["post hoc, not part of the rule"]["site means per combination, without memory (rad/s)"]["flip S=SXX+SYY"]
    corr = [row["measured"] - x for x in row["synthetic median per truth, oracle defect"]]
    out["10 ions, E11e: bias-corrected flip S of the noise block, t <= 1.5 ms (rad/s), over truths"] = [
        float(min(corr) + 0.43), float(max(corr) + 0.43)]  # the synthetic truths carry the D5/2 emission, S = 0.43
    iS = LABELS.index("t >= 3 ms, fields from the whole record: S")
    diff = est[iS] - r51
    z = diff / np.hypot(se[iS], sr51)
    out["late-window S minus the 51-ion rate, and z"] = [float(diff), float(z)]
    out["consistent (|z| < 2)"] = bool(abs(z) < 2)
    j = LABELS.index("t >= 3 ms, no fields: S")
    out["the same without fields, z"] = float((est[j] - r51) / np.hypot(se[j], sr51))
    out["transverse polarisation at 3 ms, site-mean |r_xy| per state"] = [
        float(np.hypot(m[s, times.searchsorted(3.0), 0], m[s, times.searchsorted(3.0), 1])) for s in (0, 1)]
    return out


def plot(out):
    plt = style()
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    est = out["10 ions: estimate and bootstrap sd"]
    rows = [("10 ions, t <= 1.5 ms (fields, S, c)", est["learning window: S"]),
            ("10 ions, whole record (fields, S, c)", est["whole record: S"]),
            ("10 ions, t >= 3 ms (fields fixed)", est["t >= 3 ms, fields from the whole record: S"]),
            ("10 ions, t >= 3 ms (no fields)", est["t >= 3 ms, no fields: S"])]
    lo, hi = out["10 ions, E11e: bias-corrected flip S of the noise block, t <= 1.5 ms (rad/s), over truths"]
    for k, (lab, (v, s)) in enumerate(rows):
        ax.errorbar(v, k, xerr=1.96 * s, fmt="o", color=CAT[0], ms=5, capsize=2)
    ax.plot([lo, hi], [len(rows)] * 2, color=CAT[1], lw=3)
    r, sr = out["51 ions: balanced sector-changing rate per ion (1/s) and se"]
    ax.errorbar(r, len(rows) + 1, xerr=1.96 * sr, fmt="s", color=INK, ms=5, capsize=2)
    ax.axvspan(r - 1.96 * sr, r + 1.96 * sr, color=MUTED, alpha=0.12, lw=0)
    ax.set_yticks(range(len(rows) + 2), [lab for lab, _ in rows] + ["10 ions, E11e noise block (range)",
                                                                   "51 ions, Var M (balanced rate)"])
    ax.axvline(0.43, color=MUTED, ls=":", lw=1)
    ax.set_xlabel("incoherent flip rate per ion, S (1/s); dotted: D5/2 decay, S = 0.43")
    ax.invert_yaxis()
    ax.set_title("E19: flip rate of the two BEQS data sets (95% intervals)", fontsize=9)
    fig.suptitle("E19: flip-rate cross-check (data: Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e19_flip_rate_crosscheck")


if __name__ == "__main__":
    if "--replot" not in sys.argv:
        times, m, est, boot = run()
        np.savez_compressed(RESULTS / "e19_flip_rate_crosscheck.npz", times=times, m=m, est=est, boot=boot)
    z = np.load(RESULTS / "e19_flip_rate_crosscheck.npz")
    out = analyse(z["times"], z["m"], z["est"], z["boot"])
    save_json(RESULTS / "e19_flip_rate_crosscheck.json", out)
    print(plot(out))
    for key, v in out.items():
        print(f"{key}: {v}")
