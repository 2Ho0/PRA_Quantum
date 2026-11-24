"""Bootstrap uncertainty of the mean nearest-neighbour coupling J_1 of the re-learned ten-ion generator (E6a).

Added on 2026-11-24 to fill the uncertainty of J_1 that the manuscript quotes in Sec. VI A. No new fit is made: the
script reads the point estimate and the 100 bootstrap re-learnings stored by e6_ion_learning.py
(results/e6_ion_learning.npz) and evaluates J_1 with the same function `derived` that gives the other quoted
uncertainties of that paragraph (dominant dephasing eigenvalue, mean off-diagonal entry). It is a descriptive number
with no decision rule. If the bootstrap re-runs of test E33 (results/e33_parts/C/C_learn_k*, not in git) are present,
the same quantities are reported for bootstrap seeds 1 and 2.

Run: python e6b_j1_uncertainty.py   (a few seconds)
Output: results/e6b_j1_uncertainty.json
"""

import numpy as np

from common import RESULTS, save_json
from e6_ion_learning import RAD_S, ansatz, derived


def summary(npz, ans):
    z = np.load(npz)
    d = derived(z["theta"], ans)
    db = [derived(t, ans) for t in z["boot"]]
    sd = lambda key, f=lambda x: x: float(np.std([f(x[key]) for x in db], ddof=1))
    return {
        "bootstrap re-learnings": int(len(db)),
        "J_1, mean nearest-neighbour coupling (rad/s)": float(d["J_d"][0] * RAD_S),
        "J_1 bootstrap sd (rad/s)": sd("J_d", lambda v: v[0] * RAD_S),
        "Gamma dominant eigenvalue (rad/s)": float(d["theta0"] * RAD_S),
        "Gamma dominant eigenvalue bootstrap sd (rad/s)": sd("theta0", lambda v: v * RAD_S),
        "mean off-diagonal Gamma (rad/s)": float(d["gamma_c_struct"] * RAD_S),
        "mean off-diagonal Gamma bootstrap sd (rad/s)": sd("gamma_c_struct", lambda v: v * RAD_S),
    }


def main():
    ans = ansatz()
    out = {"seed 0 (results/e6_ion_learning.npz)": summary(RESULTS / "e6_ion_learning.npz", ans)}
    for k in (1, 2):
        p = RESULTS / "e33_parts" / "C" / f"C_learn_k{k}" / "e6_ion_learning.npz"
        if p.exists():
            out[f"seed {k} (test E33, {p.relative_to(RESULTS.parent).as_posix()})"] = summary(p, ans)
    key = "J_1 bootstrap sd (rad/s)"
    sds = [v[key] for v in out.values()]
    out["J_1 bootstrap sd: largest relative change from seed 0"] = float(max(abs(x / sds[0] - 1) for x in sds))
    save_json(RESULTS / "e6b_j1_uncertainty.json", out)
    for name, v in out.items():
        print(name, v)


if __name__ == "__main__":
    main()
