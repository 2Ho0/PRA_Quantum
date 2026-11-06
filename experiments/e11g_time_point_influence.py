"""E11g (post hoc): does the charge-2 imbalance of the data rest on one time point of the learning window?

E17 found time-point jitter in the conserved quantities of the BEQS data (measured by Kraft et al., PRX 16, 031037
(2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0), with several quantities jumping together at single time points
(initial state 2 at 0.5 ms). The noise-block fit of E11c uses only the four times 0, 0.5, 1 and 1.5 ms, and shot-level
calibration (E11e, E11f) does not contain such run-level jitter. Here the fit is repeated with one interior time point
removed (0.5 or 1.0 ms; t = 0 and the window end stay), with the quadrature defect recomputed for the reduced grid from
the same reference trajectory as E11c. The change of the site-mean K = S_XX - S_YY is compared with its distribution
on 20 synthetic data sets of the E11f control truth fitted with the same designs (oracle defect; shot noise only).
Added after E17 and before E11f was analysed; no rule was fixed in advance, the result is reported as a z-score.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from common import RESULTS, save_json  # noqa: E402
from e6_ion_learning import ansatz, observables, per_setting_tables, series_from  # noqa: E402
from e6c_validation import load  # noqa: E402
from e9_symmetry_data import COMBOS, extended_ansatz  # noqa: E402
from e11_model_selection import N_SUB  # noqa: E402
from e11c_noise_ansatz import fit_noise_block  # noqa: E402
from e11e_noise_block_sbc import block_stats  # noqa: E402
from e11f_nonmarkov_truths import grids, load_truth  # noqa: E402
from e14_shot_calibration import synthetic_series  # noqa: E402
from qerr.learning import needed_strings, refine, trajectory_defect  # noqa: E402

DESIGNS = {"all four times": None, "without 0.5 ms": 0.5, "without 1.0 ms": 1.0}
T_LEARN, N_REAL = 1.5, 20
IK = 3 + list(COMBOS).index("SXX-SYY")
_W = {}


def reduced(series, times, drop):
    if drop is None:
        return series, times
    j = int(np.argmin(np.abs(times - drop)))
    return {k: np.delete(v, j) for k, v in series.items()}, np.delete(times, j)


def fit_designs(series, times, fine, traj, strings):
    """Site-mean statistics (no memory) of the noise-block fit for every design."""
    P = extended_ansatz().n_params
    out = []
    for drop in DESIGNS.values():
        s, t = reduced(series, times, drop)
        d = trajectory_defect(fine, traj, strings, t, T_LEARN)
        dt = trajectory_defect(fine, traj, strings, t, T_LEARN, weight=lambda u: u)
        out.append(block_stats(fit_noise_block(s, t, d, dt)[:P]))
    return np.array(out)


def _init(prob, bases, strings, times, fine, traj):
    _W.update(prob=prob, bases=bases, strings=strings, times=times, fine=fine, traj=traj)


def synthetic(seed):
    series = synthetic_series(_W["prob"], _W["bases"], _W["strings"], np.random.default_rng(seed))
    return fit_designs(series, _W["times"], _W["fine"], _W["traj"], _W["strings"])


def run():
    times, s1, s2, bases = load()
    obs = observables()
    strings = sorted(set(obs) | set(needed_strings(obs, ansatz())))
    z = np.load(RESULTS / "e11_stage1.npz")
    traj_strings = needed_strings(obs, ansatz())
    data = fit_designs(series_from(per_setting_tables((s1, s2), bases, strings)), times, refine(times, N_SUB),
                       [z["traj0"], z["traj1"]], traj_strings)
    tr = load_truth("control")
    fine, _ = grids(times)
    with ProcessPoolExecutor(max_workers=8, initializer=_init,
                             initargs=(tr["prob"], bases, strings, times, fine, tr["fine"])) as pool:
        syn = np.array(list(pool.map(synthetic, [7727 * i + 3 for i in range(N_REAL)])))
    return data, syn


def analyse(data, syn):
    names = list(DESIGNS)
    out = {"data: site-mean K per design (rad/s)": {n: float(data[i, IK]) for i, n in enumerate(names)},
           "data: charged rms per design (rad/s)": {n: float(data[i, 0]) for i, n in enumerate(names)}}
    rows = {}
    for i, n in enumerate(names[1:], start=1):
        dK = syn[:, i, IK] - syn[:, 0, IK]
        d = data[i, IK] - data[0, IK]
        rows[n] = {"data change of K (rad/s)": float(d),
                   "synthetic change: mean, sd": [float(dK.mean()), float(dK.std(ddof=1))],
                   "z of the data change": float((d - dK.mean()) / dK.std(ddof=1))}
    out["change of K when one time point is removed"] = rows
    out["synthetic K per design: median"] = {n: float(np.median(syn[:, i, IK])) for i, n in enumerate(names)}
    return out


if __name__ == "__main__":
    if "--replot" not in sys.argv:
        data, syn = run()
        np.savez_compressed(RESULTS / "e11g_time_point_influence.npz", data=data, syn=syn)
    z = np.load(RESULTS / "e11g_time_point_influence.npz")
    out = analyse(z["data"], z["syn"])
    save_json(RESULTS / "e11g_time_point_influence.json", out)
    for key, v in out.items():
        print(f"{key}: {v}")
