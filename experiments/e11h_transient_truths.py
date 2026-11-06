"""E11h: does a transient in the first millisecond produce the charge-2 imbalance of the BEQS data?

E11f excluded non-Markovian candidates (data measured by Kraft et al., PRX 16, 031037 (2026); Zenodo
10.5281/zenodo.17750419, CC BY 4.0; analysed here from the public raw shots). E19 then found that the apparent flip
rate of the 10-ion data is about 19 /s on t <= 1.5 ms but 2-3 /s later, as in the 51-ion data: something acts mainly
in the first millisecond, which a time-local ansatz cannot represent.
(A) Calibration on the mean-Z balance of both initial states (E17, E19; whole record, weights from the bootstrap
    errors of E17), each with an exponential transient of time constant tau scanned on a grid:
    flip burst:      uniform static fields, a constant c, and an isotropic flip rate S(t) = S_inf + S_1 e^{-t/tau}
    field transient: S fixed at the late-window value of E19 (3.3 /s), c, and uniform fields
                     h(t) = h_static + h_trans e^{-t/tau}
    (a free S makes the field fit unphysical: S < 0 and a static field of 220 rad/s).
(B) Shot-level SBC exactly as E11f with two truths built from (A); both use the Y15 couplings and z fields, the
    measured product initial states, quasi-static collective noise (f = 1) and no emission:
    flip burst:      S(t) on every ion (covariant, so the collective factor of Lemma G.0 applies); the static fields of
                     the calibration are left out because the E11c ansatz contains static fields
    field transient: h(t) on every ion, the collective noise as an explicit quasi-static detuning (12 Gauss-Hermite
                     nodes), exponential-midpoint steps of 0.0625 ms
    Only t <= 1.5 ms enters the fit; later outcome distributions are copied from the E11f control and are not used.
Fixed before the run (fit without memory, site-mean K = S_XX - S_YY), as in E11f: a truth reproduces K if the measured
-94.1 lies inside its 5-95% range for both defects; it gives the sign and half the excess if its median minus the
control median is <= -37 rad/s for both defects. The other combinations and the mean-Z change of the truths are
reported, not judged.
"""

import os
import sys

_THREADS = "4" if "--truths" in sys.argv else "1"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from scipy.linalg import expm  # noqa: E402

from common import CAT, INK, MUTED, RESULTS, save_figure, save_json, style  # noqa: E402
from e6_ion_learning import N, RAD_S, ansatz, observables, per_setting_tables, product_state, series_from  # noqa: E402
from e6c_validation import load  # noqa: E402
from e9_symmetry_data import COMBOS  # noqa: E402
from e11e_noise_block_sbc import measured  # noqa: E402
from e11f_nonmarkov_truths import (PARTS, defects, grids, hamiltonian, load_truth, realisation,  # noqa: E402
                                   _init as sbc_init)
from e14_shot_calibration import chi_exponent, truth_theta  # noqa: E402
from e19_flip_rate_crosscheck import EMISSION, _SINGLES, means  # noqa: E402
from qerr import spins  # noqa: E402
from qerr.learning import needed_strings  # noqa: E402
from qerr.lindblad import OpenModel, lindblad_rhs  # noqa: E402
from qerr.pauli import expectation_tables, expectations  # noqa: E402
from qerr.shots import pauli_tensor, setting_probabilities  # noqa: E402

TAUS = np.geomspace(0.05, 8.0, 120)
S_LATE = 3.3  # 1/s, E19 late window with fields
N_GH, DT, N_REAL = 12, 0.0625, 20
TRUTHS = ("control", "flip burst", "field transient")
OUT_PARTS = RESULTS / "e11h_parts"


# ---------------- (A) calibration ----------------
def balance_rows(m, se, times, tau, kind):
    rows, rhs = [], []
    for s in (0, 1):
        X, Y, Z = (m[s, :, a] for a in range(3))
        cum = lambda f: CubicSpline(times, f).antiderivative()(times)
        target = Z - Z[0] + EMISSION * cum(1.0 + Z)
        e = np.exp(-times / tau)
        if kind == "flip burst":
            regs = [cum(2 * Y), cum(-2 * X), cum(-2 * Z), cum(-2 * Z * e), cum(np.ones_like(Z))]
        else:
            target = target + (S_LATE / RAD_S) * cum(2 * Z)
            regs = [cum(2 * Y), cum(2 * Y * e), cum(-2 * X), cum(-2 * X * e), cum(np.ones_like(Z))]
        w = 1.0 / np.hypot(se[s], se[s, 0])
        rows.append(np.column_stack(regs)[1:] * w[1:, None])
        rhs.append(target[1:] * w[1:])
    return np.vstack(rows), np.concatenate(rhs)


def calibrate():
    times, s1, s2, bases = load()
    m = means(series_from(per_setting_tables((s1, s2), bases, _SINGLES)))
    se = np.load(RESULTS / "e17_neutral_diagnosis.npz")["se"][..., 0]
    out = {}
    for kind in ("flip burst", "field transient"):
        fits = []
        for tau in TAUS:
            A, y = balance_rows(m, se, times, tau, kind)
            p, *_ = np.linalg.lstsq(A, y, rcond=None)
            fits.append((float(((A @ p - y) ** 2).sum()), tau, p * RAD_S))
        chi2, tau, p = min(fits, key=lambda f: f[0])
        out[kind] = {"tau (ms)": float(tau), "chi2": chi2, "dof": 40 - len(p), "params (rad/s, 1/s)": p.tolist()}
    A, y = balance_rows(m, se, times, 1.0, "flip burst")
    stat = A[:, [0, 1, 2, 4]]
    p, *_ = np.linalg.lstsq(stat, y, rcond=None)
    out["stationary (fields, S, c)"] = {"chi2": float(((stat @ p - y) ** 2).sum()), "dof": 36,
                                        "params (rad/s, 1/s)": (p * RAD_S).tolist()}
    return out


# ---------------- (B) truths ----------------
def setup(s):
    times, s1, s2, bases = load()
    obs = observables()
    strings = sorted(set(obs) | set(needed_strings(obs, ansatz())))
    rho0 = product_state(series_from(per_setting_tables((s1, s2), bases, _SINGLES)), s)
    fine, _ = grids(times)
    window = times[times <= 1.5 + 1e-9]
    allt = np.union1d(fine, window)
    return times, bases, strings, rho0, fine, window, allt


def flip_burst_states(rho0, allt, cal):
    """Y15 Hamiltonian plus isotropic flips S(t) on every ion, exact to rtol 1e-9; collective noise applied after."""
    th = truth_theta()
    S_inf, S_1 = cal["params (rad/s, 1/s)"][2:4]
    tau = cal["tau (ms)"]
    D = 2**N
    hm = OpenModel(H=hamiltonian(th), Gamma=np.zeros((N, N)), emission=np.zeros(N))
    fm = OpenModel(H=sp.csr_matrix((D, D)), Gamma=np.zeros((N, N)), emission=np.ones(N), absorption=np.ones(N))
    rate = lambda t: (S_inf + S_1 * np.exp(-t / tau)) / RAD_S
    rhs = lambda t, y: (lindblad_rhs(hm, y.reshape(D, D)) + rate(t) * lindblad_rhs(fm, y.reshape(D, D))).ravel()
    sol = solve_ivp(rhs, (0.0, allt[-1]), rho0.astype(complex).ravel(), t_eval=allt, method="DOP853",
                    rtol=1e-9, atol=1e-11)
    if not sol.success:
        raise RuntimeError(sol.message)
    return [sol.y[:, k].reshape(D, D) for k in range(len(allt))], True


def field_transient_states(rho0, allt, cal):
    """Y15 Hamiltonian plus uniform fields h(t), explicit quasi-static collective detuning, midpoint steps."""
    th = truth_theta()
    gamma_c = th[len(ansatz().h_terms)]
    sigma = np.sqrt((2.0 * gamma_c / 1.5) / 2.0)
    hx_s, hx_t, hy_s, hy_t = (v / RAD_S for v in cal["params (rad/s, 1/s)"][:4])
    tau = cal["tau (ms)"]
    H0 = hamiltonian(th).toarray()
    X = sum(spins.pauli_op(N, i, "x") for i in range(N))
    Y = sum(spins.pauli_op(N, i, "y") for i in range(N))
    Ztot = np.diag(spins.zvals(N).sum(axis=1).astype(float))
    x, w = np.polynomial.hermite.hermgauss(N_GH)
    steps = np.round(allt / DT).astype(int)
    assert np.allclose(steps * DT, allt), "the output times must lie on the step grid"
    avg = [np.zeros_like(rho0, dtype=complex) for _ in allt]
    for xk, wk in zip(x, w):
        Hk = H0 + np.sqrt(2.0) * sigma * xk * Ztot
        rho = rho0.astype(complex)
        k_out = 0
        for n in range(steps[-1] + 1):
            while k_out < len(steps) and steps[k_out] == n:
                avg[k_out] += (wk / np.sqrt(np.pi)) * rho
                k_out += 1
            if n == steps[-1]:
                break
            tm = (n + 0.5) * DT
            e = np.exp(-tm / tau)
            U = expm(-1j * DT * (Hk + (hx_s + hx_t * e) * X + (hy_s + hy_t * e) * Y))
            rho = U @ rho @ U.conj().T
    return avg, False


def truth_job(args):
    name, s, cal = args
    OUT_PARTS.mkdir(exist_ok=True)
    path = OUT_PARTS / f"{name.replace(' ', '_')}_{s}.npz"
    if path.exists():
        return path
    t0 = time.time()
    times, bases, strings, rho0, fine, window, allt = setup(s)
    states, factor = (flip_burst_states if name == "flip burst" else field_transient_states)(rho0, allt, cal)
    gamma_c = truth_theta()[len(ansatz().h_terms)]
    m = spins.total_sz(N)
    q2 = (m[:, None] - m[None, :]) ** 2
    idx, ph = expectation_tables(strings, N)
    vals, rhos = [], {}
    for t, rho in zip(allt, states):
        rf = rho * np.exp(-q2 * chi_exponent(1.0, gamma_c, t)) if factor else rho
        vals.append(expectations(rf, idx, ph))
        if np.any(np.isclose(window, t)):
            rhos[float(t)] = rf
    vals = np.array(vals)
    control = load_truth("control")
    prob = control["prob"][s].copy()  # t > 1.5 ms is not fitted; the window is replaced below
    coarse = control["coarse"][s].copy()
    for j, t in enumerate(window):
        prob[j] = np.array([setting_probabilities(E, b) for E in [pauli_tensor(rhos[float(t)], N)] for b in bases],
                           np.float32)
        coarse[j] = vals[np.searchsorted(allt, t - 1e-12)]
    fine_at = np.searchsorted(allt, fine - 1e-12)
    np.savez_compressed(path, fine=vals[fine_at], fine_ref=vals[fine_at], coarse=coarse, prob=prob,
                        seconds=time.time() - t0)
    print(f"{path.stem}: {time.time() - t0:.0f} s", flush=True)
    return path


def load_h(name):
    if name == "control":
        return load_truth("control")
    z = [np.load(OUT_PARTS / f"{name.replace(' ', '_')}_{s}.npz") for s in (0, 1)]
    return {k: [zz[k] for zz in z] for k in ("fine", "coarse", "prob")}


def run_sbc():
    times, s1, s2, bases = load()
    obs = observables()
    strings = sorted(set(obs) | set(needed_strings(obs, ansatz())))
    fine, _ = grids(times)
    tr = {n: load_h(n) for n in TRUTHS}
    ref = defects(fine, load_truth("control")["fine_ref"], strings, times)
    dfs = {n: {"oracle": defects(fine, tr[n]["fine"], strings, times), "reference": ref} for n in TRUTHS}
    prob = {n: tr[n]["prob"] for n in TRUTHS}
    jobs = [(n, 130363 * i + 1000 * k + 29) for k, n in enumerate(TRUTHS) for i in range(N_REAL)]
    with ProcessPoolExecutor(max_workers=8, initializer=sbc_init, initargs=(prob, dfs, bases, strings, times)) as pool:
        return np.array(list(pool.map(realisation, jobs))).reshape(len(TRUTHS), N_REAL, 2, 2, -1)


def analyse(stats, cal):
    meas = measured()[0]
    iK = 3 + list(COMBOS).index("SXX-SYY")
    K = stats[:, :, :, 0, iK]
    med = np.median(K, axis=1)
    lo, hi = np.percentile(K, [5, 95], axis=1)
    out = {"calibration (A)": cal, "measured site-mean K (rad/s)": meas[iK]}
    rows, repro, half = {}, [], []
    for k, n in enumerate(TRUTHS):
        row = {}
        for j, dname in enumerate(("oracle", "reference")):
            s = stats[k, :, j, 0]
            row[f"{dname} defect"] = {
                "K median, 5th, 95th percentile": [float(med[k, j]), float(lo[k, j]), float(hi[k, j])],
                "K median minus control median": float(med[k, j] - med[0, j]),
                "charged rms median": float(np.median(s[:, 0])),
                "site means per combination, median": {c: float(np.median(s[:, 3 + m])) for m, c in enumerate(COMBOS)}}
        if n != "control":
            r_ok = all(lo[k, j] <= meas[iK] <= hi[k, j] for j in (0, 1))
            h_ok = all(med[k, j] - med[0, j] <= -37.0 for j in (0, 1))
            row["reproduces K"], row["sign and half the excess"] = bool(r_ok), bool(h_ok)
            repro += [n] if r_ok else []
            half += [n] if h_ok else []
            c = load_h(n)["coarse"]
            strings = sorted(set(observables()) | set(needed_strings(observables(), ansatz())))
            iz = [strings.index(((i, "Z"),)) for i in range(N)]
            row["truth: mean Z change at 0.5 / 1 / 1.5 ms, state 1 and 2"] = [
                [float(c[s][j, iz].mean() - c[s][0, iz].mean()) for j in (1, 2, 3)] for s in (0, 1)]
        rows[n] = row
    out["truths"] = rows
    out["verdict"] = {"reproduce K": repro, "give the sign and half the excess": half}
    return out


def plot(stats, out):
    plt = style()
    iK = 3 + list(COMBOS).index("SXX-SYY")
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    for k, n in enumerate(TRUTHS):
        for j, (mk, c) in enumerate((("o", CAT[0]), ("s", CAT[1]))):
            v = stats[k, :, j, 0, iK]
            ax.plot(v, np.full(v.size, k + 0.18 * (j - 0.5)) + np.linspace(-0.05, 0.05, v.size), mk, color=c, ms=3.2,
                    alpha=0.75, label=("oracle defect" if j == 0 else "reference defect") if k == 0 else None)
    ax.axvline(out["measured site-mean K (rad/s)"], color=INK, lw=1.6, label="BEQS data (E11c)")
    ax.axvline(0.0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(TRUTHS)), TRUTHS)
    ax.invert_yaxis()
    ax.set_xlabel("site-mean charge-2 imbalance $S_{XX} - S_{YY}$ of the fit (rad/s)")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("E11h: synthetic fits with a first-millisecond transient", fontsize=9)
    fig.suptitle("E11h: charge-2 imbalance under transient truths (data: Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e11h_transient_truths")


if __name__ == "__main__":
    cal = calibrate()
    if "--truths" in sys.argv:
        jobs = [(n, s, cal[n]) for n in TRUTHS[1:] for s in (0, 1)]
        with ProcessPoolExecutor(max_workers=2) as pool:
            list(pool.map(truth_job, jobs))
    elif "--sbc" in sys.argv:
        np.savez_compressed(RESULTS / "e11h_transient_truths.npz", stats=run_sbc())
    if "--truths" not in sys.argv:
        stats = np.load(RESULTS / "e11h_transient_truths.npz")["stats"]
        out = analyse(stats, cal)
        save_json(RESULTS / "e11h_transient_truths.json", out)
        print(plot(stats, out))
        for key, v in out.items():
            print(f"{key}: {v}")
