"""E18b: the basis-sensitivity check of E18 under a familywise error rate (post hoc).

E18 removed the settings whose inferred measurement bases had a small margin and compared 9 key results with removing
the same number of random settings (40 draws, two thresholds), 18 comparisons in all. Its preregistered rule asked every
comparison to fall inside the 2.5-97.5% range of the random removals. That rule has no multiplicity correction: by
chance alone about 0.9 of 18 comparisons fall outside, and 2 did (|z| <= 2.0). The rule was the design defect.
Here the same stored results (results/e18_basis_sensitivity.npz) are judged with two familywise tests at level 5%:
  max statistic   the largest |z| over the 18 comparisons against its own null distribution, built from the random
                  removals (each draw standardised against the other draws, thresholds paired draw by draw)
  Holm            Holm-Bonferroni on two-sided Gaussian p-values of the 18 z values
Status: post hoc. The E18 numbers were known when these tests were chosen; this records how the failure of A1
reads under a rule with a multiplicity correction, it is not a new preregistered test.
"""

import numpy as np
from scipy.stats import norm

from common import RESULTS, save_json

N_RANDOM, ALPHA = 40, 0.05
NAMES = ("gamma_c (X15)", "J_nn (X15)", "beta (Xb15)", "beta (Xb30)", "K", "charged rms", "mean Z change, state 1",
         "mean ZZ change, state 2", "energy change, state 2")


def standardised(rand, values):
    return (values - rand.mean(axis=0)) / rand.std(axis=0, ddof=1)


def main():
    res = np.load(RESULTS / "e18_basis_sensitivity.npz")["res"]
    full = res[0]
    z_obs, null = [], []
    for k in range(2):
        drop = res[1 + k] - full
        rand = res[3 + N_RANDOM * k:3 + N_RANDOM * (k + 1)] - full
        z_obs.append(standardised(rand, drop))
        null.append(np.array([standardised(np.delete(rand, r, axis=0), rand[r]) for r in range(N_RANDOM)]))
    z_obs, null = np.array(z_obs), np.array(null)            # (2, 9), (2, 40, 9)
    max_obs = float(np.abs(z_obs).max())
    max_null = np.abs(null).max(axis=(0, 2))                 # draw r of both thresholds paired: max over 18
    p_max = float((1 + np.sum(max_null >= max_obs)) / (1 + N_RANDOM))
    p = 2 * norm.sf(np.abs(z_obs.ravel()))
    order = np.argsort(p)
    holm_reject = []
    for rank, i in enumerate(order):
        if p[i] <= ALPHA / (len(p) - rank):
            holm_reject.append(int(i))
        else:
            break
    labels = [f"threshold {th}: {n}" for th in (4, 5) for n in NAMES]
    out = {"z of the 18 comparisons": {lab: float(v) for lab, v in zip(labels, z_obs.ravel())},
           "comparisons with |z| > 1.96 (the per-comparison rule of E18)": int(np.sum(np.abs(z_obs) > 1.96)),
           "expected by chance under the per-comparison rule": 18 * ALPHA,
           "max statistic: observed max |z|, 95% of the null max, p": [max_obs, float(np.quantile(max_null, 0.95)), p_max],
           "Holm: smallest p, its threshold, rejected comparisons": [float(p[order[0]]), ALPHA / len(p),
                                                                    [labels[i] for i in holm_reject]],
           "verdict (post hoc): bases harmless under a familywise 5% rule": bool(p_max > ALPHA and not holm_reject)}
    save_json(RESULTS / "e18b_familywise.json", out)
    for key, v in out.items():
        print(f"{key}: {v}")


if __name__ == "__main__":
    main()
