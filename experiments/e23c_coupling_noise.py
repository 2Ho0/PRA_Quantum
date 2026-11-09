"""E23c: does quasi-static noise of the overall coupling strength remove the neutral-class error? (preregistered)

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0), analysed
here from the public raw shots. E23 located the neutral error of Xb15uf mostly outside the conserved directions: the
pair pattern of Z_iZ_j in state 2 (dominated by the edge ions) and the flip-flops of state 1, with late-time phase
errors of the coherent oscillations. Models learned on longer windows predict the neutral classes worse, which points
to a non-Markovian process the time-local ansatz cannot hold. A screening (E23: responses of the E11f truths added to
Xb15uf, amplitude fitted on one initial state, factors on the other) singled out a quasi-static common scale of the
couplings, i.e. a shot-to-shot fluctuation of the interaction strength.
Model Xb15ufJ: Xb15uf (E20, learned on t <= 1.5 ms) with every two-body coupling multiplied by a scale lambda that is
constant within a shot and Gaussian across shots, lambda ~ N(1, sigma^2). The learning is unchanged: for sigma = 0.12
the induced dephasing changes the expectation values on t <= 1.5 ms by at most 0.02 (rms 0.004, below the shot noise
of about 0.01 per observable), so the window fit is kept as an approximation. Prediction: the Lindblad evolution on a
lambda grid 0.64-1.36 (step 0.06), averaged with Gaussian weights after cubic interpolation in lambda. The sigma of
each state is the value that minimises the neutral-class residual of the OTHER state on 1.5 < t <= 10 ms (grid
0-0.12), and each state is predicted with it.
Fixed before the run (the rule of E20): the coupling noise explains the neutral error if all three neutral classes reach
a calibration factor <= 1.2 without raising the factor of all 435 observables above 1.15 (factors as in E20, with the
Xb15 bootstrap spread). Reported, not judged: sigma of both states, the factors as a function of sigma, and the
combination with the collective flips of E22b by superposition (Xb15uc - Xb15uf added to the prediction).
Status: the mechanism was chosen after a screening on the same data (E23). This run checks the screening with the full
dynamics of the model's own generator; it is a confirmation, not a blind test.
Post hoc check (in analyse): the temporal variance of the neutral observables on t >= 4 ms (data corrected for the shot
noise) against both models. It tests whether the data's late oscillations are damped as the coupling noise predicts.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import gc  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402

from common import CAT, INK, MUTED, ORDINAL, RESULTS, save_figure, save_json, style  # noqa: E402
from e6_ion_learning import N, observables, per_setting_tables, product_state, series_from  # noqa: E402
from e6c_validation import charge_classes, combine, combine_sem, load  # noqa: E402
from e9_symmetry_data import extended_ansatz  # noqa: E402
from e11_model_selection import EMISSION_D52  # noqa: E402
from e20_flip_models import S_51, factors  # noqa: E402
from qerr.learning import gamma_matrix  # noqa: E402
from qerr.lindblad import OpenModel, lindblad_rhs  # noqa: E402
from qerr.pauli import expectation_tables, expectations, psparse  # noqa: E402

LAMBDAS = np.round(np.arange(0.64, 1.36 + 1e-9, 0.06), 2)
SIGMAS = np.round(np.arange(0.0, 0.12 + 1e-9, 0.0025), 4)
NEUTRAL = ("q=0  Z_i", "q=0  Z_iZ_j", "q=0  flip-flop")
T_LEARN = 1.5
PARTS = RESULTS / "e23c_parts"


def hamiltonians(theta):
    """(one-body part, two-body part) of the learned Hamiltonian as sparse matrices."""
    ext = extended_ansatz()
    one = sum(theta[m] * psparse(P, N) for m, P in enumerate(ext.h_terms) if len(P) == 1 and theta[m] != 0.0)
    two = sum(theta[m] * psparse(P, N) for m, P in enumerate(ext.h_terms) if len(P) == 2 and theta[m] != 0.0)
    return one.tocsr(), two.tocsr()


def predict_job(args):
    lam, s = args
    PARTS.mkdir(exist_ok=True)
    path = PARTS / f"lam{lam:.2f}_s{s}.npy"
    if path.exists():
        return np.load(path)
    t0 = time.time()
    e20 = np.load(RESULTS / "e20_flip_models.npz")
    theta, beta = e20["theta_Xb15uf"], float(e20["beta_Xb15uf"])
    times, s1, s2, bases = load()
    obs = observables()
    rho = product_state(series_from(per_setting_tables((s1, s2), bases, [((i, a),) for i in range(N) for a in "XYZ"])), s)
    one, two = hamiltonians(theta)
    ext = extended_ansatz()
    model = OpenModel(H=(one + lam * two).tocsr(), Gamma=gamma_matrix(theta, ext),
                      emission=np.full(N, EMISSION_D52 + S_51), absorption=np.full(N, S_51))
    D = 2**N
    gamma_c = float(theta[len(ext.h_terms)])
    Wc = OpenModel(H=sp.csr_matrix((D, D)), Gamma=np.ones((N, N)), emission=np.zeros(N))._elementwise
    idx, ph = expectation_tables(obs, N)
    out = [expectations(rho, idx, ph)]
    for a, b in zip(times[:-1], times[1:]):
        def rhs(u, y, a=a):
            X = y.reshape(D, D)
            return (lindblad_rhs(model, X) + max(beta * (a + u), -gamma_c) * Wc * X).ravel()
        sol = solve_ivp(rhs, (0.0, b - a), rho.ravel(), t_eval=[b - a], method="DOP853", rtol=1e-7, atol=1e-9)
        rho = sol.y[:, -1].reshape(D, D)
        del sol
        gc.collect()  # the solver's stage arrays sit in reference cycles (see qerr.lindblad.evolve)
        out.append(expectations(rho, idx, ph))
    out = np.array(out)
    np.save(path, out)
    print(f"lambda {lam:.2f} state {s + 1}: {time.time() - t0:.0f} s", flush=True)
    return out


def run():
    jobs = [(lam, s) for lam in LAMBDAS for s in (0, 1)]
    with ProcessPoolExecutor(max_workers=10) as pool:
        res = list(pool.map(predict_job, jobs))
    return np.array(res).reshape(len(LAMBDAS), 2, *res[0].shape)  # (lambda, state, time, obs)


def averaged(grid_pred, sigma):
    """Gaussian average over lambda of the per-lambda predictions (lambda, time, obs) for one state."""
    if sigma == 0.0:
        return grid_pred[list(LAMBDAS).index(1.0)]
    fine = np.linspace(LAMBDAS[0], LAMBDAS[-1], 1441)
    w = np.exp(-0.5 * ((fine - 1.0) / sigma) ** 2)
    w /= w.sum()
    return np.tensordot(w, CubicSpline(LAMBDAS, grid_pred, axis=0)(fine), axes=1)


def neutral_objective(pred_s, s, z, bands, oos):
    """Sum of the squared neutral calibration factors of one state (the calibration target)."""
    cls = charge_classes(observables())
    tot = 0.0
    for n in NEUTRAL:
        co = cls[n]
        d, p, se = combine(z["data"], co)[s], combine(pred_s[None], co)[0], combine_sem(z["sem"], co)[s]
        sd = combine(bands, co).std(axis=0, ddof=1)[s]
        tot += np.mean((d - p)[oos] ** 2) / np.mean((sd**2 + se**2)[oos])
    return tot


def analyse(grid):
    z = np.load(RESULTS / "e11_model_selection.npz")
    e20 = np.load(RESULTS / "e20_flip_models.npz")
    e22b = np.load(RESULTS / "e22b_collective_flips.npz")
    bands = np.load(RESULTS / "e15_coverage.npz")["band_Xb15"]
    times = z["times"]
    oos = times > T_LEARN + 1e-9
    out = {"validation: max |lambda = 1 run - E20 Xb15uf prediction|": float(
        np.abs(grid[list(LAMBDAS).index(1.0)] - e20["pred_Xb15uf"]).max())}
    curves = {s: np.array([neutral_objective(averaged(grid[:, s], sg), s, z, bands, oos) for sg in SIGMAS]) for s in (0, 1)}
    sig = {s: float(SIGMAS[int(np.argmin(curves[s]))]) for s in (0, 1)}
    out["sigma minimising the neutral residual of state 1, state 2"] = [sig[0], sig[1]]
    pred = np.stack([averaged(grid[:, s], sig[1 - s]) for s in (0, 1)])  # each state with the other state's sigma
    own = np.stack([averaged(grid[:, s], sig[s]) for s in (0, 1)])
    combo = pred + (e22b["pred"] - e20["pred_Xb15uf"])
    fac = {"Xb15uf": factors(e20["pred_Xb15uf"], z, bands, oos), "Xb15uc (E22b)": factors(e22b["pred"], z, bands, oos),
           "Xb15ufJ (other state's sigma)": factors(pred, z, bands, oos),
           "Xb15ufJ (own sigma, in-sample)": factors(own, z, bands, oos),
           "Xb15ufJ + collective flips (superposition)": factors(combo, z, bands, oos)}
    out["calibration factors, t > 1.5 ms (Xb15 bootstrap spread)"] = fac
    out["factors vs sigma (both states, same sigma)"] = {
        f"{sg:.3f}": factors(np.stack([averaged(grid[:, s], sg) for s in (0, 1)]), z, bands, oos)
        for sg in SIGMAS[::8]}
    f = fac["Xb15ufJ (other state's sigma)"]
    out["verdict"] = {"all neutral classes <= 1.2": bool(all(f[n] <= 1.2 for n in NEUTRAL)),
                      "all 435 observables <= 1.15": bool(f["all 435 observables"] <= 1.15)}
    out["verdict"]["coupling noise explains the neutral error"] = bool(all(out["verdict"].values()))
    cls = charge_classes(observables())
    late = times >= 4.0 - 1e-9
    amp = {}
    for n in NEUTRAL:
        co = cls[n]
        for s in (0, 1):
            d, se = combine(z["data"], co)[s][late], combine_sem(z["sem"], co)[s][late]
            var = lambda p: float(np.mean(combine(p, co)[s][late].var(axis=0, ddof=1)))
            amp[f"{n}, state {s + 1}"] = {"data (shot noise removed)": float(np.mean(d.var(axis=0, ddof=1)
                                                                                      - np.mean(se**2, axis=0))),
                                          "Xb15uf": var(e20["pred_Xb15uf"]), "Xb15ufJ": var(pred)}
    out["post hoc: temporal variance on t >= 4 ms (late oscillation amplitude squared)"] = amp
    return out, curves, pred


def plot(out, curves, pred):
    plt = style()
    z = np.load(RESULTS / "e11_model_selection.npz")
    e20 = np.load(RESULTS / "e20_flip_models.npz")
    obs = observables()
    k = {O: m for m, O in enumerate(obs)}
    times = z["times"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    ax = axes[0]
    for s, c in ((0, CAT[0]), (1, CAT[1])):
        ax.plot(SIGMAS, np.sqrt(curves[s] / 3), "-", color=c, lw=1.4, label=f"state {s + 1}")
        ax.axvline(out["sigma minimising the neutral residual of state 1, state 2"][s], color=c, lw=0.8, ls=":")
    ax.set_xlabel("rms of the coupling scale $\\sigma$")
    ax.set_ylabel("rms neutral calibration factor")
    ax.legend(fontsize=7)
    ax.set_title("(a) calibration of $\\sigma$ on each state", fontsize=9)
    ax = axes[1]
    fac = out["calibration factors, t > 1.5 ms (Xb15 bootstrap spread)"]
    names = ["Xb15uf", "Xb15uc (E22b)", "Xb15ufJ (other state's sigma)", "Xb15ufJ + collective flips (superposition)"]
    classes = list(fac["Xb15uf"])
    for j, (v, c) in enumerate(zip(names, [MUTED, ORDINAL[1], CAT[1], CAT[2]])):
        ax.plot([fac[v][n] for n in classes], np.arange(len(classes)) + 0.13 * (j - 1.5), "D", color=c, ms=5, label=v)
    ax.axvline(1.0, color=MUTED, ls=":", lw=1)
    ax.axvline(1.2, color=MUTED, ls="--", lw=0.8)
    ax.set_yticks(range(len(classes)), classes)
    ax.invert_yaxis()
    ax.set_xlabel("calibration factor, t > 1.5 ms")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("(b) calibration by charge class", fontsize=9)
    ax = axes[2]
    m = k[((0, "Z"), (9, "Z"))]
    ax.plot(times, z["data"][1][:, m], "o", color=INK, ms=3.5, label="data")
    ax.plot(times, e20["pred_Xb15uf"][1][:, m], "-", color=MUTED, lw=1.2, label="Xb15uf")
    ax.plot(times, pred[1][:, m], "-", color=CAT[1], lw=1.4, label="Xb15ufJ")
    ax.axvspan(0.0, T_LEARN, color=MUTED, alpha=0.08, lw=0)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("$\\langle Z_0 Z_9 \\rangle$, state 2")
    ax.legend(fontsize=7)
    ax.set_title("(c) an edge correlation", fontsize=9)
    fig.suptitle("E23c: quasi-static noise of the coupling strength (data: Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e23c_coupling_noise")


if __name__ == "__main__":
    path = RESULTS / "e23c_coupling_noise.npz"
    if "--replot" not in sys.argv:
        np.savez_compressed(path, grid=run(), lambdas=LAMBDAS)
    grid = np.load(path)["grid"]
    out, curves, pred = analyse(grid)
    save_json(RESULTS / "e23c_coupling_noise.json", out)
    print(plot(out, curves, pred))
    for key, v in out.items():
        print(f"{key}: {v}")
