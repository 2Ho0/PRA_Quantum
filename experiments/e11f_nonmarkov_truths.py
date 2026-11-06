"""E11f: can non-Markovian single-ion or control noise produce the charge-2 imbalance of the BEQS data?

E11e calibrated the noise-block fit of E11c with truths that contain no single-ion noise (E14: Y15 couplings and z
fields, D5/2 emission, collective noise). The data (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo
10.5281/zenodo.17750419, CC BY 4.0; analysed here from the public raw shots) keep a site-mean charge-2 imbalance
S_XX - S_YY = -94 rad/s against a synthetic median of -18 to -22, an excess of about -74 rad/s. Here the same
shot-level SBC is run with truths that add one non-Markovian process each. All truths use the measured product
initial states, the Y15 couplings and z fields, no emission, and quasi-static collective noise (f = 1 of E14, the
value the 3 ms window supports) unless stated:
  control         no extra noise (checks the propagation route against E11e)
  local qs 50     quasi-static single-ion z detuning, independent Gaussian per ion and shot, rms 50 rad/s
  local qs 150    the same, rms 150 rad/s
  global J 10%    quasi-static common scale of all couplings, J -> (1 + e) J with e ~ N(0, 0.1^2)
  X15 fields + qs detuning   the X15 Hamiltonian (U(1)-symmetric part and its single-site transverse fields, 8-64
                  rad/s) with the collective noise written as a quasi-static detuning d sum_i Z_i, d ~ N(0, beta_s/2),
                  so that the fields are dephased by it; no free parameter
  local white 5   Markovian single-ion z dephasing at 5 rad/s (not in the E11c ansatz, which has collective
                  dephasing only)
Covariant truths are propagated exactly in the magnetisation sectors and the collective factor of Lemma G.0 is applied
afterwards; the field truth is propagated in the full space for every node of a 20-point Gauss-Hermite rule in d.
Each truth is sampled like the data (200 settings, 200 shots, both initial states; 20 realisations) and fitted as in
E11e, with the truth's own defect (oracle) and with the white-collective control trajectory (reference).
Fixed before the run (fit without memory, site-mean K = S_XX - S_YY):
  route check: the control's median K is within 10 rad/s of E11e's f = 1 oracle median (-22.2 rad/s);
  a truth "reproduces K" if the measured -94.1 lies inside its 5-95% range for both defects;
  a truth "gives the sign and half the excess" if its median K minus the control's is <= -37 rad/s for both defects;
  the verdict lists the truths that do either; the other seven combinations are reported, not judged.
Also reported (diagnostics, no rule): the noise-free fit of every truth, its conserved quantities (E17) next to the
data, and the fraction of shots with all ten spins equal (the collision check of E17 against a model).
Run in two steps: `--truths` (2 processes x 4 BLAS threads) and then `--sbc` (8 processes x 1 thread).
"""

import os
import sys

_THREADS = "4" if "--truths" in sys.argv else "1"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from common import CAT, INK, MUTED, ORDINAL, RESULTS, save_figure, save_json, style  # noqa: E402
from e6_ion_learning import (N, PAIRS, RAD_S, ansatz, observables, per_setting_tables, product_state,  # noqa: E402
                             series_from)
from e6c_validation import load  # noqa: E402
from e9_symmetry_data import COMBOS, extended_ansatz  # noqa: E402
from e11c_noise_ansatz import fit_noise_block  # noqa: E402
from e11e_noise_block_sbc import block_stats, measured  # noqa: E402
from e14_shot_calibration import chi_exponent, synthetic_series, truth_theta  # noqa: E402
from qerr import spins  # noqa: E402
from qerr.learning import needed_strings, refine, to_openmodel, trajectory_defect  # noqa: E402
from qerr.lindblad import OpenModel, evolve  # noqa: E402
from qerr.pauli import expectation_tables, expectations  # noqa: E402
from qerr.shots import pauli_tensor, setting_probabilities  # noqa: E402

TRUTHS = ("control", "local qs 50", "local qs 150", "global J 10%", "X15 fields + qs detuning", "local white 5")
N_REAL, N_QS, N_GH_J, N_GH_D, T_LEARN = 20, 32, 10, 20, 1.5
PARTS = RESULTS / "e11f_parts"
K_E11E_F1 = -22.2  # E11e: median site-mean K of the f = 1 truth, oracle defect, fit without memory (rad/s)
_W = {}


def grids(times):
    """The fine grid of the learning window (defect) and the union with the 21 measured times."""
    fine = refine(times[times <= T_LEARN + 1e-9], 4)
    return fine, np.union1d(fine, times)


def hamiltonian(theta, keep=lambda p: True):
    a = ansatz()
    th = np.array([v if (m < len(a.h_terms) and keep(a.h_terms[m])) else 0.0 for m, v in enumerate(theta)])
    return to_openmodel(th, a, sparse=True).H


def sectors():
    m = spins.total_sz(N)
    perm = np.argsort(m, kind="stable")
    edges = np.r_[0, np.flatnonzero(np.diff(m[perm])) + 1, len(m)]
    return perm, [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]


class Propagator:
    """exp(-iHt) rho0 exp(iHt) for a static H, block-diagonal in the magnetisation sectors or in the full space."""

    def __init__(self, H, rho0, blocks=None):
        H = H.toarray() if hasattr(H, "toarray") else np.asarray(H)
        if blocks is None:
            self.E, V = np.linalg.eigh(H)
            self.V, self.perm = [V], np.arange(H.shape[0])
            self.slices = [slice(0, H.shape[0])]
        else:
            self.perm, self.slices = blocks
            Hp = H[np.ix_(self.perm, self.perm)]
            self.E = np.empty(H.shape[0])
            self.V = []
            for sl in self.slices:
                w, v = np.linalg.eigh(Hp[sl, sl])
                self.E[sl] = w
                self.V.append(v)
        self.inv = np.argsort(self.perm)
        self.tilde = self._right(self._left(rho0[np.ix_(self.perm, self.perm)], dagger=True), dagger=False)

    def _left(self, X, dagger):
        Y = np.empty_like(X)
        for sl, V in zip(self.slices, self.V):
            Y[sl] = (V.conj().T if dagger else V) @ X[sl]
        return Y

    def _right(self, X, dagger):
        Y = np.empty_like(X)
        for sl, V in zip(self.slices, self.V):
            Y[:, sl] = X[:, sl] @ (V.conj().T if dagger else V)
        return Y

    def __call__(self, t):
        ph = np.exp(-1j * self.E * t)
        Xp = self._right(self._left(ph[:, None] * self.tilde * ph.conj()[None, :], dagger=False), dagger=True)
        return Xp[np.ix_(self.inv, self.inv)]


def realisations(name, theta, theta_x15, rng):
    """Yield (weight, dense Hamiltonian) of a truth; the collective factor is applied afterwards except for the field
    truth, whose collective noise is the explicit detuning."""
    H = hamiltonian(theta).toarray()
    z = spins.zvals(N).astype(float)
    if name in ("control", "local white 5"):
        yield 1.0, H
    elif name.startswith("local qs"):
        sigma = float(name.split()[-1]) / RAD_S
        for d in [rng.normal(0.0, sigma, N) for _ in range(N_QS // 2)]:
            for sign in (1.0, -1.0):  # antithetic pairs
                yield 1.0 / N_QS, H + np.diag(z @ (sign * d))
    elif name == "global J 10%":
        x, w = np.polynomial.hermite.hermgauss(N_GH_J)
        Hc = hamiltonian(theta, lambda p: len(p) == 2).toarray()
        for xk, wk in zip(x, w):
            yield wk / np.sqrt(np.pi), H + 0.1 * np.sqrt(2.0) * xk * Hc
    elif name == "X15 fields + qs detuning":
        gamma_c = theta[len(ansatz().h_terms)]
        sigma = np.sqrt((2.0 * gamma_c / 1.5) / 2.0)  # 2 sigma^2 = beta_s: the charge-1 decay of E14's f = 1 truth
        x, w = np.polynomial.hermite.hermgauss(N_GH_D)
        Hx = hamiltonian(theta_x15).toarray()
        for xk, wk in zip(x, w):
            yield wk / np.sqrt(np.pi), Hx + np.diag(np.sqrt(2.0) * sigma * xk * z.sum(axis=1))
    else:
        raise ValueError(name)


def truth_job(args):
    """Fine-window trajectory (oracle defect), coarse expectations and outcome distributions of one truth and state."""
    name, s = args
    path = PARTS / f"{name.replace(' ', '_').replace('%', 'pc').replace('+', 'and')}_{s}.npz"
    if path.exists():
        return path
    t0 = time.time()
    times, s1, s2, bases = load()
    obs = observables()
    strings = sorted(set(obs) | set(needed_strings(obs, ansatz())))
    singles = [((i, a),) for i in range(N) for a in "XYZ"]
    rho0 = product_state(series_from(per_setting_tables((s1, s2), bases, singles)), s)
    theta = truth_theta()
    gamma_c = theta[len(ansatz().h_terms)]
    theta_x15 = np.load(RESULTS / "e11_model_selection.npz")["theta_X15"]
    fine, allt = grids(times)
    idx, ph = expectation_tables(strings, N)
    m = spins.total_sz(N)
    q2 = (m[:, None] - m[None, :]) ** 2
    factor = name != "X15 fields + qs detuning"
    vals = {f: np.zeros((len(allt), len(strings))) for f in (1.0, 0.0)}
    rho_bar = np.zeros((len(times), 2**N, 2**N), complex)
    coarse = np.searchsorted(allt, times - 1e-12)
    rng = np.random.default_rng(1000 + TRUTHS.index(name))
    if name == "local white 5":
        model = OpenModel(H=hamiltonian(theta), Gamma=np.eye(N) * 5.0 / RAD_S, emission=np.zeros(N))
        rhos = evolve(model, rho0, allt, rtol=1e-9, atol=1e-11)
        runs = iter([(1.0, lambda k: rhos[k])])
    else:
        blocks = None if name == "X15 fields + qs detuning" else sectors()
        runs = ((w, Propagator(H, rho0, blocks)) for w, H in realisations(name, theta, theta_x15, rng))
        runs = ((w, lambda k, P=P: P(allt[k])) for w, P in runs)
    for w, rho_at in runs:
        for k, t in enumerate(allt):
            rho = rho_at(k)
            for f in ((1.0, 0.0) if name == "control" else (1.0,)):
                rf = rho * np.exp(-q2 * chi_exponent(f, gamma_c, t)) if factor else rho
                vals[f][k] += w * expectations(rf, idx, ph)
                if f == 1.0 and k in coarse:
                    rho_bar[list(coarse).index(k)] += w * rf
    prob = np.array([[setting_probabilities(E, b) for b in bases] for E in (pauli_tensor(r, N) for r in rho_bar)],
                    np.float32)
    fine_at = np.searchsorted(allt, fine - 1e-12)
    np.savez_compressed(path, fine=vals[1.0][fine_at], fine_ref=vals[0.0][fine_at], coarse=vals[1.0][coarse],
                        prob=prob, seconds=time.time() - t0)
    print(f"{path.stem}: {time.time() - t0:.0f} s", flush=True)
    return path


def route_check():
    """Block propagation against the Lindblad integrator for the bare Y15 Hamiltonian, 0.5 ms, state 1."""
    times, s1, s2, bases = load()
    singles = [((i, a),) for i in range(N) for a in "XYZ"]
    rho0 = product_state(series_from(per_setting_tables((s1, s2), bases, singles)), 0)
    H = hamiltonian(truth_theta())
    ref = evolve(OpenModel(H=H, Gamma=np.zeros((N, N)), emission=np.zeros(N)), rho0, [0.0, 0.5],
                 rtol=1e-10, atol=1e-12)[-1]
    return float(np.abs(Propagator(H, rho0, sectors())(0.5) - ref).max())


def load_truth(name):
    z = [np.load(PARTS / f"{name.replace(' ', '_').replace('%', 'pc').replace('+', 'and')}_{s}.npz") for s in (0, 1)]
    return {k: [zz[k] for zz in z] for k in ("fine", "fine_ref", "coarse", "prob")}


def _init(prob, defects, bases, strings, times):
    _W.update(prob=prob, defects=defects, bases=bases, strings=strings, times=times)


def realisation(args):
    name, seed = args
    series = synthetic_series(_W["prob"][name], _W["bases"], _W["strings"], np.random.default_rng(seed))
    P = extended_ansatz().n_params
    out = []
    for dname in ("oracle", "reference"):
        d, dt = _W["defects"][name][dname]
        th = fit_noise_block(series, _W["times"], d, dt)
        out.append([block_stats(th[:P]), block_stats(th[P:2 * P])])
    return np.array(out)  # (defect, memory, stat)


def defects(fine, traj, strings, times):
    return (trajectory_defect(fine, traj, strings, times, T_LEARN),
            trajectory_defect(fine, traj, strings, times, T_LEARN, weight=lambda s: s))


def run_sbc():
    times, s1, s2, bases = load()
    obs = observables()
    strings = sorted(set(obs) | set(needed_strings(obs, ansatz())))
    fine, _ = grids(times)
    tr = {n: load_truth(n) for n in TRUTHS}
    prob = {n: tr[n]["prob"] for n in TRUTHS}
    ref = defects(fine, tr["control"]["fine_ref"], strings, times)
    dfs = {n: {"oracle": defects(fine, tr[n]["fine"], strings, times), "reference": ref} for n in TRUTHS}
    P = extended_ansatz().n_params
    noise_free = {}
    for n in TRUTHS:
        series = {(s, p): tr[n]["coarse"][s][:, c] for s in (0, 1) for c, p in enumerate(strings)}
        th = fit_noise_block(series, times, *dfs[n]["oracle"])
        noise_free[n] = np.array(block_stats(th[:P]))
    jobs = [(n, 104729 * i + 1000 * k + 17) for k, n in enumerate(TRUTHS) for i in range(N_REAL)]
    with ProcessPoolExecutor(max_workers=8, initializer=_init, initargs=(prob, dfs, bases, strings, times)) as pool:
        stats = np.array(list(pool.map(realisation, jobs))).reshape(len(TRUTHS), N_REAL, 2, 2, -1)
    return stats, np.array([noise_free[n] for n in TRUTHS])


def conserved_from(coarse, strings):
    """Mean Z, mean ZZ over all pairs and <H_Y15> (rad/s) from (times, strings) values."""
    k = {p: m for m, p in enumerate(strings)}
    a = ansatz()
    th = truth_theta()
    mz = np.mean([coarse[:, k[((i, "Z"),)]] for i in range(N)], axis=0)
    mzz = np.mean([coarse[:, k[((i, "Z"), (j, "Z"))]] for i, j in PAIRS], axis=0)
    en = sum(th[m] * RAD_S * coarse[:, k[p]] for m, p in enumerate(a.h_terms) if th[m] != 0.0)
    return np.stack([mz, mzz, en], -1)


def analyse(stats, noise_free):
    times, s1, s2, bases = load()
    obs = observables()
    strings = sorted(set(obs) | set(needed_strings(obs, ansatz())))
    meas = measured()[0]
    iK, iS = 3 + list(COMBOS).index("SXX-SYY"), 3 + list(COMBOS).index("flip S=SXX+SYY")
    K = stats[:, :, :, 0, iK]                                   # (truth, realisation, defect)
    med = np.median(K, axis=1)
    lo, hi = np.percentile(K, [5, 95], axis=1)
    out = {"measured site-mean K, charged rms, flip S (rad/s)": [meas[iK], meas[0], meas[iS]]}
    c0 = TRUTHS.index("control")
    out["route check"] = {"block propagation vs Lindblad integrator, max |rho| difference at 0.5 ms": route_check(),
                          "control median K, oracle defect (rad/s)": float(med[c0, 0]),
                          "within 10 rad/s of E11e (-22.2)": bool(abs(med[c0, 0] - K_E11E_F1) <= 10.0)}
    rows, repro, half = {}, [], []
    for k, n in enumerate(TRUTHS):
        row = {}
        for j, dname in enumerate(("oracle", "reference")):
            s = stats[k, :, j, 0]
            row[f"{dname} defect"] = {
                "K median, 5th, 95th percentile": [float(med[k, j]), float(lo[k, j]), float(hi[k, j])],
                "K median minus control median": float(med[k, j] - med[c0, j]),
                "charged rms median [5, 95]": [float(np.median(s[:, 0])), *np.percentile(s[:, 0], [5, 95]).tolist()],
                "site means per combination, median": {c: float(np.median(s[:, 3 + m])) for m, c in enumerate(COMBOS)}}
        row["noise-free fit: K, charged rms, flip S"] = [float(noise_free[k, iK]), float(noise_free[k, 0]),
                                                        float(noise_free[k, iS])]
        if n != "control":
            r_ok = all(lo[k, j] <= meas[iK] <= hi[k, j] for j in (0, 1))
            h_ok = all(med[k, j] - med[c0, j] <= -37.0 for j in (0, 1))
            row["reproduces K"], row["sign and half the excess"] = bool(r_ok), bool(h_ok)
            if r_ok:
                repro.append(n)
            if h_ok:
                half.append(n)
        rows[n] = row
    out["truths"] = rows
    out["verdict"] = {"reproduce K": repro, "give the sign and half the excess": half}
    # diagnostics against E17: conserved quantities and all-equal shots
    e17 = np.load(RESULTS / "e17_neutral_diagnosis.npz")
    late = times >= 2.0 - 1e-9
    diag = {"data": {f"state {s + 1}": (e17["cq"][s, late].mean(0) - e17["cq"][s, 0]).tolist() for s in (0, 1)}}
    for n in TRUTHS:
        tr = load_truth(n)
        diag[n] = {}
        for s in (0, 1):
            c = conserved_from(tr["coarse"][s], strings)
            diag[n][f"state {s + 1}"] = (c[late].mean(0) - c[0]).tolist()
    out["conserved quantities: mean over t >= 2 ms minus t = 0 (mean Z, mean ZZ, energy in rad/s)"] = diag
    alleq = {}
    for s, sp in enumerate((s1, s2)):
        obs_eq = ((sp == 1).all(axis=3) | (sp == -1).all(axis=3)).mean(axis=(1, 2))
        row = {"data": [float(obs_eq[j]) for j in (0, 3, 10, 20)]}
        for n in ("control", "X15 fields + qs detuning"):
            p = load_truth(n)["prob"][s]
            row[n] = [float((p[j, :, 0] + p[j, :, -1]).mean()) for j in (0, 3, 10, 20)]
        alleq[f"state {s + 1}"] = row
    out["fraction of shots with all ten spins equal at 0 / 1.5 / 5 / 10 ms"] = alleq
    return out


def plot(stats, out):
    plt = style()
    meas = out["measured site-mean K, charged rms, flip S (rad/s)"][0]
    iK = 3 + list(COMBOS).index("SXX-SYY")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for k, n in enumerate(TRUTHS):
        for j, (mk, c) in enumerate((("o", CAT[0]), ("s", CAT[1]))):
            v = stats[k, :, j, 0, iK]
            ax.plot(v, np.full(v.size, k + 0.18 * (j - 0.5)) + np.linspace(-0.05, 0.05, v.size), mk, color=c, ms=3.2,
                    alpha=0.75, label=("oracle defect" if j == 0 else "reference defect") if k == 0 else None)
    ax.axvline(meas, color=INK, lw=1.6, label="BEQS data (E11c)")
    ax.axvline(0.0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(TRUTHS)), TRUTHS)
    ax.invert_yaxis()
    ax.set_xlabel("site-mean charge-2 imbalance $S_{XX} - S_{YY}$ of the fit (rad/s)")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("E11f: synthetic fits with one non-Markovian process per truth", fontsize=9)
    fig.suptitle("E11f: charge-2 imbalance under non-Markovian truths (data: Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e11f_nonmarkov_truths")


if __name__ == "__main__":
    if "--truths" in sys.argv:
        PARTS.mkdir(exist_ok=True)
        jobs = [(n, s) for n in TRUTHS for s in (0, 1)]
        with ProcessPoolExecutor(max_workers=2) as pool:
            list(pool.map(truth_job, jobs))
    elif "--sbc" in sys.argv:
        stats, noise_free = run_sbc()
        np.savez_compressed(RESULTS / "e11f_nonmarkov_truths.npz", stats=stats, noise_free=noise_free)
    if "--truths" not in sys.argv:
        z = np.load(RESULTS / "e11f_nonmarkov_truths.npz")
        out = analyse(z["stats"], z["noise_free"])
        save_json(RESULTS / "e11f_nonmarkov_truths.json", out)
        print(plot(z["stats"], out))
        for key, v in out.items():
            print(f"{key}: {v}")
