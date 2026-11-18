"""E32 figure for the manuscript (one column): drawn from results/e32_charge_dephasing.json, no recomputation.

(a) -ln D_q against -ln D_1 for every time and tilt, with the slopes of collective dephasing (q^2: 0, 4, 9; dashed,
    in the colour of the charge) and of independent dephasing (Hamming weight: 2, 2, 3; dotted, grey). Points whose
    D_q is below twice its bootstrap sd are left out, because their logarithm is dominated by noise.
(b) The fitted ratios r_q per tilt with 95% intervals (1.96 bootstrap SE, conservative by a factor of 2.4-2.8 as the
    calibration shows) and the same two predictions.
The analysis and its preregistered rules are in e32_charge_dephasing.py; this script only redraws.
Run:  python experiments/e32_paper_figure.py   ->  results/e32_paper.{png,pdf}
"""

import json

import numpy as np

from common import CAT, COL_W, INK, MUTED, RESULTS, save_figure, style

ANGLES = (("0p2", r"$0.2\pi$"), ("0p33", r"$0.33\pi$"), ("0p5", r"$0.5\pi$"))
MARKS = {0: ("D", INK), 2: ("s", CAT[1]), 3: ("^", CAT[2])}
COLLECTIVE = {0: 0, 2: 4, 3: 9}
INDEPENDENT = {0: 2, 2: 2, 3: 3}


def main() -> None:
    res = json.loads((RESULTS / "e32_charge_dephasing.json").read_text())["per_angle"]
    plt = style()
    fig, ax = plt.subplots(2, 1, figsize=(COL_W, 4.3), constrained_layout=True,
                           gridspec_kw={"height_ratios": [1.15, 1.0]})
    kmax = 0.0
    for tag, _ in ANGLES:
        D1 = np.array(res[tag]["D"]["1"][1:])
        kappa = -np.log(D1)
        kmax = max(kmax, float(kappa.max()))
        for q in (0, 2, 3):
            m, c = MARKS[q]
            Dq, sd = np.array(res[tag]["D"][str(q)][1:]), np.array(res[tag]["D_sd"][str(q)][1:])
            ok = Dq > 2 * sd
            ax[0].plot(kappa[ok], -np.log(Dq[ok]), m, color=c, ms=3.2, mfc="none" if q == 0 else c, mew=0.8,
                       alpha=0.85, label=f"charge {q}" if tag == "0p5" else None)
    kk = np.linspace(0, 1.05 * kmax, 20)
    for q in (0, 2, 3):
        ax[0].plot(kk, COLLECTIVE[q] * kk, "--", color=MARKS[q][1], lw=0.9)
    for slope in sorted(set(INDEPENDENT.values())):
        ax[0].plot(kk, slope * kk, ":", color=MUTED, lw=1.0)
    ax[0].set_xlim(0, 1.05 * kmax)
    ax[0].set_ylim(-0.3, 4.6)
    ax[0].set_xlabel(r"$-\ln D_1$")
    ax[0].set_ylabel(r"$-\ln D_q$")
    ax[0].set_title("(a)", loc="left")
    ax[0].legend(loc="upper left")

    x = np.arange(len(ANGLES))
    for k, q in enumerate((0, 2, 3)):
        m, c = MARKS[q]
        vals = [res[tag]["r"][str(q)] for tag, _ in ANGLES]
        errs = [1.96 * res[tag]["r_se"][str(q)] for tag, _ in ANGLES]
        ax[1].errorbar(x + 0.18 * (k - 1), vals, yerr=errs, fmt=m, color=c, ms=3.5, capsize=2, lw=0.9,
                       mfc="none" if q == 0 else c, mew=0.8)
        ax[1].axhline(COLLECTIVE[q], color=c, lw=0.9, ls="--")
    for level in sorted(set(INDEPENDENT.values())):
        ax[1].axhline(level, color=MUTED, lw=1.0, ls=":")
    ax[1].set_xticks(x, [lab for _, lab in ANGLES])
    ax[1].set_xlim(-0.5, len(ANGLES) - 0.5)
    ax[1].set_ylim(-1, 13)
    ax[1].set_xlabel(r"tilt angle $\theta$")
    ax[1].set_ylabel(r"exponent ratio $r_q$")
    ax[1].set_title("(b)", loc="left")
    fig.suptitle("E32: q^2 law on the Mpemba dephasing data")
    save_figure(fig, "e32_paper")


if __name__ == "__main__":
    main()
