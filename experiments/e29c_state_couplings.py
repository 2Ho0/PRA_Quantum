"""E29c: uncertainty of the single-state couplings of E29b (descriptive, post hoc)

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0), analysed
here from the public raw shots.
Question: E29b refitted estimator T on each initial state alone and found mean nearest-neighbour couplings of 633 and
731 rad/s, without uncertainties. This script repeats the two refits of E29b (same estimator, offset, prior scale and
window) and reports the mean nearest-neighbour coupling with its Laplace posterior standard deviation, which includes
the Birge factor as in E29, and the difference of the two states with its standard deviation (the two fits use
disjoint data). No band, no decision rule; nothing here changes a verdict.
Run: `python e29c_state_couplings.py` (two refits in parallel, a few minutes).
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from common import RESULTS, save_json  # noqa: E402
from e24_eiv_correction import N_WIN  # noqa: E402
from e29_neutral_trajectory import NN, Neutral, current_prior, fit_t, load  # noqa: E402
from e29b_window_diagnostic import VARIANTS  # noqa: E402

STATES = ("state 1 only", "state 2 only")


def refit(name):
    ti, st = VARIANTS[name]
    a = np.load(RESULTS / "e29_stepA.npz")
    times = load()[0]
    model = Neutral(a["bloch"])
    y = (a["y"] - a["delta"][:, :N_WIN])[list(st)][:, list(ti)]
    sem = a["sem"][list(st)][:, list(ti)]
    x_hat, sig, _, info = fit_t(lambda x, tg: model(x, tg)[list(st)], y, sem, a["x_ls"], current_prior()[1],
                                times[list(ti)])
    w = np.zeros(len(x_hat))
    w[NN] = 1.0 / len(NN)
    return name, float(w @ x_hat) * 1e3, float(np.sqrt(w @ sig @ w)) * 1e3, info


def main():
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=2) as pool:
        res = {r[0]: r[1:] for r in pool.map(refit, STATES)}
    out = {}
    for name in STATES:
        jnn, sd, info = res[name]
        out[name] = {"J_nn mean (rad/s)": jnn, "posterior sd (rad/s)": sd,
                     "chi^2 / nu, Birge factor": [float(info["chi2"] / info["nu"]), float(info["birge"])]}
    (j1, s1, _), (j2, s2, _) = res[STATES[0]], res[STATES[1]]
    diff, sdd = j2 - j1, float(np.hypot(s1, s2))
    out["state 2 minus state 1: difference (rad/s), sd (rad/s), z, relative to the mean of the two"] = [
        diff, sdd, diff / sdd, diff / (0.5 * (j1 + j2))]
    out["E29b values for comparison (rad/s)"] = [633.3249203859945, 730.5007855679045]
    save_json(RESULTS / "e29c_state_couplings.json", out)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
