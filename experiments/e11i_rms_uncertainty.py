"""E11i: bootstrap spread of the out-of-sample rms errors and of the 10-ms Zeno factors of E11 (descriptive, post hoc)

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0), analysed
here from the public raw shots.
Question: the manuscript quotes the out-of-sample rms errors of E11 (W15 0.118, X15 0.052, Xb15 0.040) and the Zeno
factors at 10 ms (X15 white, Xb15 fitted mixture) without uncertainties. This script gives their spread over bootstrap
re-learnings of the 200 settings. The point estimates are not refitted, and there is no decision rule.
  rms: X15 and Xb15 from the 12 bootstrap re-learnings that E15 simulated exactly (results/e15_coverage.npz, the first
       12 resamples of the E11 bootstrap); W15 from the same 12 resamples (E11 seed 2028), fitted as in E11 with its
       stored defects and simulated with the E11 validation code. The rms is that of E11: all 435 observables, both
       initial states, 5 < t <= 10 ms.
  W15 refits that diverge: E11 bootstrapped only the constrained variants. Refitted on a resample, the full W15
       dephasing matrix can run off along a direction the resampled settings do not fix (a pair of eigenvalues of
       opposite sign and size beyond 1e14 rad/s); such a model cannot be integrated. A refit whose dephasing matrix has
       an eigenvalue beyond 1e6 rad/s (threshold set after the first runs; bounded refits stay below 600 rad/s) is
       counted as diverged and not simulated, and the W15 rms statistics are over the bounded refits only. The cause
       is listed per resample: a resample leaves some of the Pauli strings of the integral method unmeasured, and the
       design matrix of the full ansatz can then have null directions, which all lie inside the dephasing matrix.
  Zeno factors: the 100 E11 bootstrap values of gamma_c (X15) and of (gamma_c, beta) (Xb15), with the E11 formula.
  For Xb15 the file also lists, per resample, the learned growing rate beta beside the rms, and their correlation.
Run: `python e11i_rms_uncertainty.py` (12 W15 refits of a few seconds each, then two simulations of the ten-ion model
per bounded refit, each 5-35 min on one core depending on the load; resumable: each simulation is kept in
results/e11i_parts/).
`--quick` writes the result file without the W15 refits and simulations, with the W15 entry marked as pending.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from functools import lru_cache  # noqa: E402

import numpy as np  # noqa: E402

from common import RESULTS, save_json  # noqa: E402
from e11_model_selection import (N_BOOT, N_SUB, STAGE1, T_VALID, VARIANTS, build_systems, fit_variant,  # noqa: E402
                                 validate_job, variant_map)
from e6_ion_learning import RAD_S, ansatz, gamma_matrix, observables, per_setting_tables, series_from  # noqa: E402
from e6c_validation import load  # noqa: E402
from qerr.learning import needed_strings, refine, trajectory_defect  # noqa: E402

N_REP, WIN = 12, 1.5
DIVERGED = 1.0e3  # rad/ms (1e6 rad/s): a refitted dephasing eigenvalue beyond this marks a diverged W15 refit
WORKERS = int(os.environ.get("E11I_WORKERS", "12"))
PARTS = RESULTS / "e11i_parts"
T10 = 10.0  # ms
BB = np.linspace(0.0, T10, 4001)
GAMMA_BEQS = 0.460 / 10  # rad/ms: dominant eigenvalue of Gamma in Kraft et al., divided by N = 10


def zeno_factor(chi):
    """Second-order leakage with the kernel chi, relative to no protection (the E11 formula)."""
    return float(np.trapezoid((T10 - BB) * chi, BB) / (0.5 * T10**2))


def white(gc, q):
    return zeno_factor(np.exp(-2 * q**2 * gc * BB))


def mixture(gc, beta, q):
    return zeno_factor(np.exp(-q**2 * np.maximum(4 * gc * BB + 2 * beta * BB**2, 0.0) / 2))


@lru_cache(maxsize=1)
def w15_setup():
    """Data tables and the stored E11 defects, shared by all W15 refits."""
    times, s1, s2, bases = load()
    ans, obs = ansatz(), observables()
    strings = needed_strings(obs, ans)
    tables = per_setting_tables((s1, s2), bases, sorted(set(obs) | set(strings)))
    z = np.load(STAGE1)
    defects = {(WIN, wt): trajectory_defect(refine(times, N_SUB), [z["traj0"], z["traj1"]], strings, times, WIN,
                                            weight=(lambda s: s) if wt else None) for wt in (False, True)}
    return times, ans, obs, tables, defects


def w15_systems(weights):
    times, ans, obs, tables, defects = w15_setup()
    series = series_from(tables, weights)
    return series, build_systems(series, times, defects, obs, ans, [WIN])


def w15_theta(weights):
    """W15 refitted on one resample of the settings (weights None: the E11 point fit)."""
    return fit_variant("W15", w15_systems(weights)[1], w15_setup()[1])[0]


def w15_rank(weights):
    """Pauli strings a resample leaves unmeasured, and the null directions of its W15 design matrix: their number and
    the share of each that lies in the dephasing matrix."""
    _, ans, _, tables, _ = w15_setup()
    series, systems = w15_systems(weights)
    missing = len({p for (s, p) in series_from(tables) if (s, p) not in series})
    T = variant_map(ans, *VARIANTS["W15"][:3])[0]
    _, sv, vt = np.linalg.svd(systems[WIN][0] @ T, full_matrices=False)
    nh, nd = len(ans.h_terms), len(ans.deph_pairs)
    share = [float(np.linalg.norm((T @ v)[nh:nh + nd]) / np.linalg.norm(T @ v)) for v in vt[sv / sv[0] < 1e-10]]
    return missing, share


def part_path(k, s):
    return PARTS / f"w15_k{k:02d}_s{s}.npy"


def job(spec):
    k, theta, s = spec
    pred = validate_job(("W15", theta, 0.0, s))
    np.save(part_path(k, s), pred)
    return k, s


def rms(pred, data, mask):
    return float(np.sqrt(np.mean((pred[:, mask] - data[:, mask]) ** 2)))


def summary(values, central):
    v = np.asarray(values)
    return {"central (E11)": central, "bootstrap mean": float(v.mean()), "bootstrap sd": float(v.std(ddof=1)),
            "bootstrap min, max": [float(v.min()), float(v.max())], "n": int(v.size)}


def simulate_w15():
    t0 = time.time()
    check = np.max(np.abs(w15_theta(None) - np.load(RESULTS / "e11_model_selection.npz")["theta_W15"]))
    print(f"W15 point fit reproduced to {check:.2e} rad/ms ({time.time() - t0:.0f} s)", flush=True)
    bases = load()[3]
    rng = np.random.default_rng(2028)  # the E11 bootstrap sequence
    n_set = bases.shape[0]
    weights = [np.bincount(rng.integers(0, n_set, n_set), minlength=n_set).astype(float) for _ in range(N_BOOT)]
    ans = ansatz()
    thetas = [w15_theta(weights[k]) for k in range(N_REP)]
    eig = [np.linalg.eigvalsh(gamma_matrix(th, ans))[[0, -1]] for th in thetas]
    bounded = [k for k in range(N_REP) if np.abs(eig[k]).max() < DIVERGED]
    ranks = [w15_rank(weights[k]) for k in range(N_REP)]
    print(f"W15 refits: {N_REP - len(bounded)} of {N_REP} diverged; bounded: {bounded} ({time.time() - t0:.0f} s)",
          flush=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    specs = [(k, thetas[k], s) for k in bounded for s in (0, 1) if not part_path(k, s).exists()]
    print(f"{len(specs)} of {2 * len(bounded)} W15 simulations to run on {min(WORKERS, max(len(specs), 1))} workers",
          flush=True)
    with ProcessPoolExecutor(max_workers=min(WORKERS, max(len(specs), 1))) as pool:
        for k, s in pool.map(job, specs):
            print(f"  W15 replicate {k}, state {s + 1} done ({time.time() - t0:.0f} s)", flush=True)
    return float(check), [[float(lo * RAD_S), float(hi * RAD_S)] for lo, hi in eig], bounded, ranks


def main(quick=False):
    check, eig, bounded, ranks = (None, None, None, None) if quick else simulate_w15()
    z11 = np.load(RESULTS / "e11_model_selection.npz")
    times, data = z11["times"], z11["data"]
    mask = times > T_VALID + 1e-9
    z15 = np.load(RESULTS / "e15_coverage.npz")
    out_rms = {}
    if bounded is not None:
        pred_w = [np.array([np.load(part_path(k, s)) for s in (0, 1)]) for k in bounded]
        rms_w = [rms(p, data, mask) for p in pred_w]
        out_rms["W15"] = summary(rms_w, rms(z11["pred_W15"], data, mask)) | {
            "note": "bootstrap statistics over the bounded refits only; the other refits diverged and have no rms",
            "bounded resamples and their rms": dict(zip([str(k) for k in bounded], rms_w)),
            "diverged resamples": [k for k in range(N_REP) if k not in bounded],
            "divergence threshold on the dephasing eigenvalues (rad/s)": DIVERGED * RAD_S,
            "smallest and largest dephasing eigenvalue per resample (rad/s)": eig,
            "Pauli strings unmeasured in the resample (of those the point fit uses), per resample": [m for m, _ in ranks],
            "null directions of the W15 design matrix per resample (singular value < 1e-10 of the largest)":
                [len(s) for _, s in ranks],
            "share of each null direction inside the dephasing matrix, per resample": [s for _, s in ranks]}
    else:
        out_rms["W15"] = {"central (E11)": rms(z11["pred_W15"], data, mask), "bootstrap sd": None,
                          "note": "pending: run without --quick"}
    out_rms["X15"] = summary([rms(p, data, mask) for p in z15["band_X15"]], rms(z11["pred_X15"], data, mask))
    rms_xb = [rms(p, data, mask) for p in z15["band_Xb15"]]
    out_rms["Xb15"] = summary(rms_xb, rms(z11["pred_Xb15"], data, mask))
    beta12 = z11["boot_beta_Xb15"][:N_REP] * RAD_S
    out = {"rms over all 435 observables, both states, 5 < t <= 10 ms": out_rms,
           "W15 point fit reproduced (max abs difference, rad/ms)": check,
           "Xb15 per resample: beta (rad/s per ms), rms, and their correlation": {
               "beta": [float(b) for b in beta12], "rms": rms_xb,
               "correlation": float(np.corrcoef(rms_xb, beta12)[0, 1])}}
    k0 = len(ansatz().h_terms)
    zf = {}
    for q in (1, 2):
        gx = z11["boot_theta_X15"][:, k0]
        zf[f"X15 white, |Q|={q}"] = summary([white(g, q) for g in gx], white(float(z11["theta_X15"][k0]), q))
        gb, bt = z11["boot_theta_Xb15"][:, k0], z11["boot_beta_Xb15"]
        zf[f"Xb15 fitted mixture, |Q|={q}"] = summary([mixture(g, b, q) for g, b in zip(gb, bt)],
                                                      mixture(float(z11["theta_Xb15"][k0]), float(z11["beta_Xb15"]), q))
        zf[f"white at the rate of Kraft et al. (460 rad/s / 10), |Q|={q}"] = white(GAMMA_BEQS, q)
    out["Zeno factor at 10 ms (second-order leakage relative to no protection)"] = zf
    out["gamma_c and beta of the bootstrap (rad/s, rad/s per ms): X15 gamma_c mean, sd; Xb15 gamma_c, beta means"] = [
        float(z11["boot_theta_X15"][:, k0].mean() * RAD_S), float(z11["boot_theta_X15"][:, k0].std(ddof=1) * RAD_S),
        float(z11["boot_theta_Xb15"][:, k0].mean() * RAD_S), float(z11["boot_beta_Xb15"].mean() * RAD_S)]
    save_json(RESULTS / "e11i_rms_uncertainty.json", out)
    print("written results/e11i_rms_uncertainty.json", "(W15 pending)" if bounded is None else "", flush=True)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
