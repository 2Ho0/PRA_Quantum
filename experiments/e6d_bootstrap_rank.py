"""E6d: unmeasured Pauli strings in the bootstrap of the re-learned ten-ion generator (E6a), post hoc

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0), analysed
here from the public raw shots.
Question: added on 2026-11-28. The W15 refits of e11i_rms_uncertainty.py diverge in 8 of 12 resamples. A resample of
the 200 settings leaves some Pauli strings unmeasured, every equation of the integral method that contains one of them
drops out (qerr.learning.integral_system), and the design matrix of the full ansatz can lose rank. E6a fits the same
ansatz to the same equations with np.linalg.lstsq, the least-squares solution of minimum norm, so its bootstrap stays
finite. For the 100 E6a resamples (seed 2026) this script reports how often the design matrix loses rank and where its
null directions lie. It also compares the quoted statistics over the resamples without null directions with random
subsets of the same size. If every null direction lies inside the dephasing matrix, all least-squares solutions share
their Hamiltonian couplings. The minimum-norm choice of E6a then leaves the couplings and the z-scores of the
U(1)-breaking couplings unchanged.
Method: no new fit. The bootstrap re-learnings are read from results/e6_ion_learning.npz, and the rank is that of the
E11 design matrix of the full ansatz (e11i_rms_uncertainty.w15_rank). For the same resample it keeps the same rows as
the E6a system, because a row drops whenever it contains an unmeasured string, whatever the defect correction.
Descriptive, no decision rule.

Run: python e6d_bootstrap_rank.py   (about 4 min on one core)
Output: results/e6d_bootstrap_rank.json
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from common import RESULTS, save_json  # noqa: E402
from e11i_rms_uncertainty import w15_rank  # noqa: E402
from e6_ion_learning import N, RAD_S, ansatz, derived  # noqa: E402
from e6c_validation import load  # noqa: E402

N_RANDOM = 2000


def breaking(theta, ans):
    """The 110 U(1)-breaking combinations of Sec. VI A: (J^xx - J^yy)/2, (J^xy + J^yx)/2 and the transverse fields."""
    k = {p: m for m, p in enumerate(ans.h_terms)}
    out = []
    for i in range(N):
        for j in range(i + 1, N):
            xx, yy, xy, yx = (k[((i, a), (j, b))] for a, b in ("XX", "YY", "XY", "YX"))
            out += [0.5 * (theta[xx] - theta[yy]), 0.5 * (theta[xy] + theta[yx])]
        out += [theta[k[((i, "X"),)]], theta[k[((i, "Y"),)]]]
    return np.array(out)


def main():
    ans = ansatz()
    z = np.load(RESULTS / "e6_ion_learning.npz")
    theta, boot = z["theta"], z["boot"]
    n_set = load()[3].shape[0]
    rng = np.random.default_rng(2026)  # the E6a bootstrap sequence
    weights = [np.bincount(rng.integers(0, n_set, n_set), minlength=n_set).astype(float) for _ in range(len(boot))]
    ranks = [w15_rank(w) for w in weights]
    missing = np.array([m for m, _ in ranks])
    null = np.array([len(s) for _, s in ranks])
    full = np.flatnonzero(null == 0)
    b0, bb = breaking(theta, ans), np.array([breaking(t, ans) for t in boot])
    dom = np.array([derived(t, ans)["theta0"] * RAD_S for t in boot])
    max_z = lambda idx: float(np.max(np.abs(b0) / bb[idx].std(axis=0, ddof=1)))
    sd_dom = lambda idx: float(dom[idx].std(ddof=1))
    sub = np.random.default_rng(1)
    rand = [sub.choice(len(boot), full.size, replace=False) for _ in range(N_RANDOM)]
    mz, sdd = np.array([max_z(r) for r in rand]), np.array([sd_dom(r) for r in rand])
    everything = np.arange(len(boot))
    out = {
        "bootstrap re-learnings": int(len(boot)),
        "resamples whose design matrix has null directions": int((null > 0).sum()),
        "null directions per resample (singular value < 1e-10 of the largest)": null.tolist(),
        "smallest share of a null direction inside the dephasing matrix": float(min(x for _, s in ranks for x in s)),
        "unmeasured Pauli strings per resample: min, median, max": [int(missing.min()), float(np.median(missing)),
                                                                     int(missing.max())],
        "largest |z| of the 110 U(1)-breaking couplings": {
            "all resamples (as quoted in Sec. VI A)": max_z(everything),
            "resamples without null directions": max_z(full),
            f"random subsets of the same size ({N_RANDOM}): median, 5%, 95%": np.percentile(mz, [50, 5, 95]).tolist(),
            "share of random subsets at or above the value without null directions": float((mz >= max_z(full)).mean())},
        "dominant eigenvalue of Gamma, bootstrap sd (rad/s)": {
            "all resamples (as quoted in Sec. VI A)": sd_dom(everything),
            "resamples without null directions": sd_dom(full),
            f"random subsets of the same size ({N_RANDOM}): median, 5%, 95%": np.percentile(sdd, [50, 5, 95]).tolist()},
        "dominant eigenvalue of Gamma, bootstrap mean (rad/s): all, without null directions, with null directions": [
            float(dom.mean()), float(dom[full].mean()), float(dom[null > 0].mean())],
        "dominant eigenvalue of Gamma, point estimate (rad/s)": float(derived(theta, ans)["theta0"] * RAD_S),
    }
    save_json(RESULTS / "e6d_bootstrap_rank.json", out)
    for name, v in out.items():
        print(name, v if not isinstance(v, list) or len(v) < 8 else f"[{len(v)} values]", flush=True)


if __name__ == "__main__":
    main()
