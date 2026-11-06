"""E18: do the main data results depend on the inferred measurement bases?

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0), analysed
here from the public raw shots. The bases of the 200 settings are not stored; E6a inferred them from the t = 0 means of
the two known initial states (qerr.iondata.infer_bases). The margin of a (setting, qubit) is the distance to the
deciding threshold in units of its standard error; a setting's margin is the smallest over its ten qubits. We did not
ask the authors to confirm the bases, so this check stands in for that confirmation.
Every key result is recomputed with the low-margin settings removed (margin < 4: 13 settings; margin < 5: 77
settings) and compared with removing the same number of random settings (40 draws each):
  gamma_c and J_nn of X15, the memory slope beta of Xb15 and Xb30 (E11, E14), the site-mean charge-2 imbalance K and
  the charged rms of the noise-block fit (E11c), and the conserved-quantity changes of E17 (mean Z of state 1, mean ZZ
  and energy of state 2; mean over t >= 2 ms minus t = 0).
Fixed before the run: a result is robust if the change from removing the low-margin settings lies inside the central
95% of the changes from removing random settings of the same number; the bases are judged harmless if every result
is robust for both thresholds. Also reported: the expected number of misassigned (setting, qubit) cells if the t = 0
means are Gaussian, sum Phi(-margin).
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

from common import RESULTS, save_json  # noqa: E402
from e6_ion_learning import N, PAIRS, RAD_S, RAW, ansatz, observables, per_setting_tables, series_from  # noqa: E402
from e9_symmetry_data import COMBOS, extended_ansatz  # noqa: E402
from e11_model_selection import N_SUB, build_systems, fit_variant  # noqa: E402
from e11c_noise_ansatz import fit_noise_block  # noqa: E402
from e11e_noise_block_sbc import block_stats  # noqa: E402
from e17_neutral_diagnosis import as_array, conserved, energy_weights, indices  # noqa: E402
from qerr.iondata import infer_bases, load_10ion  # noqa: E402
from qerr.learning import needed_strings, refine, trajectory_defect  # noqa: E402

THRESHOLDS, N_RANDOM = (4.0, 5.0), 40
NAMES = ["gamma_c of X15 (rad/s)", "J_nn of X15 (rad/s)", "beta of Xb15 (rad/s per ms)", "beta of Xb30 (rad/s per ms)",
         "site-mean K (rad/s)", "charged rms (rad/s)", "mean Z change, state 1", "mean ZZ change, state 2",
         "energy change, state 2 (rad/s)"]
_W = {}


def load_with_margins():
    times, s1 = load_10ion(RAW, "state1")
    _, s2 = load_10ion(RAW, "state2")
    stats = lambda s: (s[0].mean(axis=1).astype(float), s[0].std(axis=1, ddof=1) / np.sqrt(s.shape[2]))
    (m1, e1), (m2, e2) = stats(s1), stats(s2)
    bases, margin = infer_bases(m1, m2, e1, e2)
    return times, s1, s2, bases, margin


def jnn(theta, ans):
    k = {p: m for m, p in enumerate(ans.h_terms)}
    return float(np.mean([theta[k[((i, "X"), (i + 1, "X"))]] + theta[k[((i, "Y"), (i + 1, "Y"))]]
                          for i in range(N - 1)]))


def _init(tables, times, defects, full_values):
    _W.update(tables=tables, times=times, defects=defects, full_values=full_values)


def results(w):
    """The nine key results for one weight vector over the settings."""
    ans, obs = ansatz(), observables()
    times, defects = _W["times"], _W["defects"]
    series = series_from(_W["tables"], w)
    systems = build_systems(series, times, defects, obs, ans, [1.5, 3.0])
    x15 = fit_variant("X15", systems, ans)[0]
    b15 = fit_variant("Xb15", systems, ans)[1]
    b30 = fit_variant("Xb30", systems, ans)[1]
    P = extended_ansatz().n_params
    st = block_stats(fit_noise_block(series, times, defects[(1.5, False)], defects[(1.5, True)])[:P])
    iK = 3 + list(COMBOS).index("SXX-SYY")
    ix = indices()
    ecols, ew = energy_weights(np.load(RESULTS / "e11_model_selection.npz")["theta_Y15"])
    vals = np.stack([as_array(series, s, fill=_W["full_values"][s]) for s in (0, 1)])
    cq = conserved(vals, ix, ecols, ew)
    late = times >= 2.0 - 1e-9
    ch = cq[:, late].mean(axis=1) - cq[:, 0]
    return np.array([x15[len(ans.h_terms)] * RAD_S, jnn(x15, ans) * RAD_S, b15 * RAD_S, b30 * RAD_S,
                     st[iK], st[0], ch[0, 0], ch[1, 1], ch[1, 2]])


def run():
    times, s1, s2, bases, margin = load_with_margins()
    obs = observables()
    strings = sorted(set(obs) | set(needed_strings(obs, ansatz())))
    tables = per_setting_tables((s1, s2), bases, strings)
    z = np.load(RESULTS / "e11_stage1.npz")
    traj_strings = needed_strings(obs, ansatz())
    fine = refine(times, N_SUB)
    defects = {(win, wt): trajectory_defect(fine, [z["traj0"], z["traj1"]], traj_strings, times, win,
                                            weight=(lambda s: s) if wt else None)
               for win in (1.5, 3.0) for wt in (False, True)}
    smin = margin.min(axis=1)
    rng = np.random.default_rng(2041)
    weights = [np.ones(200)]
    for th in THRESHOLDS:
        weights.append((smin >= th).astype(float))
    for th in THRESHOLDS:
        n_drop = int((smin < th).sum())
        for _ in range(N_RANDOM):
            w = np.ones(200)
            w[rng.choice(200, n_drop, replace=False)] = 0.0
            weights.append(w)
    full_values = np.stack([as_array(series_from(tables), s) for s in (0, 1)])  # fills strings a subset lacks
    with ProcessPoolExecutor(max_workers=8, initializer=_init,
                             initargs=(tables, times, defects, full_values)) as pool:
        res = np.array(list(pool.map(results, weights)))
    expected_wrong = float(norm.sf(margin).sum())
    return res, smin, expected_wrong


def analyse(res, smin, expected_wrong):
    full = res[0]
    out = {"settings below the margin threshold (4, 5)": [int((smin < th).sum()) for th in THRESHOLDS],
           "expected number of misassigned (setting, qubit) cells, sum Phi(-margin)": expected_wrong,
           "full data": dict(zip(NAMES, full.tolist()))}
    verdict = True
    for k, th in enumerate(THRESHOLDS):
        drop = res[1 + k] - full
        rand = res[3 + N_RANDOM * k: 3 + N_RANDOM * (k + 1)] - full
        lo, hi = np.percentile(rand, [2.5, 97.5], axis=0)
        ok = (drop >= lo) & (drop <= hi)
        verdict &= bool(ok.all())
        out[f"margin < {th:g}: change when the low-margin settings are removed"] = {
            n: {"change": float(drop[i]), "random-removal 2.5%, 97.5%": [float(lo[i]), float(hi[i])],
                "z against random removal": float((drop[i] - rand[:, i].mean()) / rand[:, i].std(ddof=1)),
                "inside": bool(ok[i])} for i, n in enumerate(NAMES)}
    out["verdict: every result robust for both thresholds"] = verdict
    return out


if __name__ == "__main__":
    if "--replot" not in sys.argv:
        res, smin, expected_wrong = run()
        np.savez_compressed(RESULTS / "e18_basis_sensitivity.npz", res=res, smin=smin, expected_wrong=expected_wrong)
    z = np.load(RESULTS / "e18_basis_sensitivity.npz")
    out = analyse(z["res"], z["smin"], float(z["expected_wrong"]))
    save_json(RESULTS / "e18_basis_sensitivity.json", out)
    for key, v in out.items():
        print(f"{key}: {v}")
