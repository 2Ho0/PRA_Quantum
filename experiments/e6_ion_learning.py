"""E6a: re-learning the 10-ion generator of Kraft et al., PRX 16, 031037 (2026) from their public raw shots.

Data: Zenodo 10.5281/zenodo.17750419 (CC BY 4.0), fetched by data/fetch_beqs.py.
Pipeline: decode the shots; infer the random Pauli bases from the t = 0 statistics of the two known product
states; estimate every Pauli expectation value of weight <= 3; fit with the integral method on t <= 1.5 ms (the
BEQS learning window) plus three rounds of quadrature defect correction; bootstrap over the 200 settings;
validate on 1.5 < t <= 10 ms.
Ansatz: XX, YY, XY, YX on every pair; X, Y, Z on every site; a full symmetric dephasing matrix (BEQS Eq. 19);
sigma^- emission on every site. Internally ms and rad/ms; reported in rad/s. BEQS convention
H_XY = (1/2) sum J_ij (XX + YY), hence J_ij = theta_XX + theta_YY.
"""

import itertools
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import CAT, MUTED, ORDINAL, RESULTS, ROOT, save_figure, save_json, style
from qerr.charge import protective_collective_rate
from qerr.iondata import infer_bases, load_10ion, pauli_means, per_setting_means
from qerr.learning import (Ansatz, fit, gamma_matrix, integral_system, learn, needed_strings, quadrature_defect,
                           to_openmodel)
from qerr.lindblad import evolve
from qerr.pauli import expectation_tables, expectations

RAW = ROOT / "data" / "raw" / "state1_2"
N, T_LEARN, N_BOOT = 10, 1.5, 100
PAIRS = list(itertools.combinations(range(N), 2))
RAD_S = 1000.0  # rad/ms -> rad/s
DEFECT = {"n_sub": 4, "rtol": 1e-7, "atol": 1e-9}  # model signals are slow (J_nn ~ 0.6 rad/ms): dt_fine = 0.125 ms


def ansatz():
    h = [((i, a), (j, b)) for i, j in PAIRS for a, b in ("XX", "YY", "XY", "YX")]
    h += [((i, a),) for i in range(N) for a in "XYZ"]
    return Ansatz(n=N, h_terms=h, deph_pairs=[(i, i) for i in range(N)] + PAIRS, emission_sites=list(range(N)))


def observables():
    return [((i, a),) for i in range(N) for a in "XYZ"] + [((i, a), (j, b)) for i, j in PAIRS for a in "XYZ"
                                                         for b in "XYZ"]


def per_setting_tables(spins, bases, strings):
    tables = {}
    for p in strings:
        idx, m1 = per_setting_means(spins[0], bases, p)
        tables[p] = (idx, (m1, per_setting_means(spins[1], bases, p)[1]))
    return tables


def series_from(tables, weights=None):
    series = {}
    for p, (idx, per) in tables.items():
        w = np.ones(len(idx)) if weights is None else weights[idx]
        if len(idx) == 0 or w.sum() == 0:
            continue
        for s in (0, 1):
            series[(s, p)] = (w[:, None] * per[s]).sum(axis=0) / w.sum()
    return series


def product_state(series, s):
    rho = np.array([[1.0 + 0j]])
    for i in range(N):
        r = np.array([series[(s, ((i, a),))][0] for a in "XYZ"])
        r = r / max(1.0, np.linalg.norm(r))
        rho = np.kron(rho, 0.5 * np.array([[1 + r[2], r[0] - 1j * r[1]], [r[0] + 1j * r[1], 1 - r[2]]]))
    return rho


def physical(theta, ans):
    """Central model made CPTP: dephasing matrix projected to PSD, emission rates clipped at zero."""
    w, V = np.linalg.eigh(gamma_matrix(theta, ans))
    th = theta.copy()
    th[-N:] = np.clip(th[-N:], 0.0, None)
    return to_openmodel(th, ans, gamma=(V * np.clip(w, 0.0, None)) @ V.T, sparse=True)


def symmetrized(theta, ans):
    """Drop every U(1)-breaking Hamiltonian component: (XX-YY)/2, (XY+YX)/2, X and Y fields."""
    th = theta.copy()
    k = {p: m for m, p in enumerate(ans.h_terms)}
    for i, j in PAIRS:
        xx, yy, xy, yx = (k[((i, a), (j, b))] for a, b in ("XX", "YY", "XY", "YX"))
        th[xx] = th[yy] = 0.5 * (theta[xx] + theta[yy])
        th[xy], th[yx] = 0.5 * (theta[xy] - theta[yx]), -0.5 * (theta[xy] - theta[yx])
    for i in range(N):
        th[k[((i, "X"),)]] = th[k[((i, "Y"),)]] = 0.0
    return th


def predict(model, rho0, times, strings):
    idx, ph = expectation_tables(strings, N)
    out, rho = [expectations(rho0, idx, ph)], rho0
    for a, b in zip(times[:-1], times[1:]):
        rho = evolve(model, rho, [0.0, b - a], rtol=1e-7, atol=1e-9)[-1]
        out.append(expectations(rho, idx, ph))
    return np.array(out)


def validation_run(args):
    theta, rho0, times = args
    return predict(physical(theta, ansatz()), rho0, times, observables())


def derived(theta, ans):
    k = {p: m for m, p in enumerate(ans.h_terms)}
    get = lambda i, j, ab: theta[k[((i, ab[0]), (j, ab[1]))]]
    J, K, L, Dm = (np.zeros((N, N)) for _ in range(4))
    for i, j in PAIRS:
        J[i, j] = J[j, i] = get(i, j, "XX") + get(i, j, "YY")
        K[i, j] = K[j, i] = 0.5 * (get(i, j, "XX") - get(i, j, "YY"))
        L[i, j] = L[j, i] = 0.5 * (get(i, j, "XY") + get(i, j, "YX"))
        Dm[i, j] = 0.5 * (get(i, j, "XY") - get(i, j, "YX"))
        Dm[j, i] = -Dm[i, j]
    B = {a: np.array([theta[k[((i, a),)]] for i in range(N)]) for a in "XYZ"}
    G = gamma_matrix(theta, ans)
    w, V = np.linalg.eigh(G)
    Gp = (V * np.clip(w, 0.0, None)) @ V.T
    off = G[np.triu_indices(N, 1)]
    gc_struct = float(off.mean())  # Gamma ~ gamma_c 11^T + diag(g): least-squares collective rate
    d = np.arange(1, N)
    Jd = np.array([np.mean([J[i, i + dd] for i in range(N - dd)]) for dd in d])
    ok = Jd > 0
    slope, icept = np.polyfit(np.log(d[ok]), np.log(Jd[ok]), 1)
    return {"J": J, "K": K, "L": L, "D": Dm, "Bx": B["X"], "By": B["Y"], "Bz": B["Z"], "Gamma": G, "Gamma_psd": Gp,
            "psd_weight": float(np.linalg.norm(G - Gp) / np.linalg.norm(G)), "gamma_c_star": protective_collective_rate(Gp),
            "theta0": float(w[-1]), "v0_overlap": float(abs(V[:, -1].sum()) / np.sqrt(N)), "J_d": Jd,
            "J0_fit": float(np.exp(icept)), "alpha_fit": float(-slope), "emission": theta[-N:],
            "gamma_c_struct": gc_struct, "g_local": np.diag(G) - gc_struct, "offdiag": off}


def run():
    t0 = time.time()
    times, s1 = load_10ion(RAW, "state1")
    _, s2 = load_10ion(RAW, "state2")
    stats = lambda s: (s[0].mean(axis=1).astype(float), s[0].std(axis=1, ddof=1) / np.sqrt(s.shape[2]))
    (m1, e1), (m2, e2) = stats(s1), stats(s2)
    bases, margin = infer_bases(m1, m2, e1, e2)
    ans, obs = ansatz(), observables()
    strings = sorted(set(obs) | set(needed_strings(obs, ans)))
    tables = per_setting_tables((s1, s2), bases, strings)
    series = series_from(tables)
    rho0s = [product_state(series, s) for s in (0, 1)]
    thetas = learn(series, ans, obs, times, T_LEARN, rho0s, n_iter=3, **DEFECT)
    theta = thetas[-1]
    print(f"learned in {time.time() - t0:.0f} s; rounds max|dtheta| (rad/s):",
          [round(float(np.abs(a - b).max() * RAD_S), 2) for a, b in zip(thetas[1:], thetas[:-1])])
    defect = quadrature_defect(to_openmodel(theta, ans, sparse=True), rho0s, needed_strings(obs, ans), times, T_LEARN,
                               **DEFECT)
    rng = np.random.default_rng(2026)
    boot = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, bases.shape[0], bases.shape[0]), minlength=bases.shape[0]).astype(float)
        A, y, _ = integral_system(obs, ans, series_from(tables, w), times, T_LEARN, defect=defect)
        boot.append(fit(A, y))
    boot = np.array(boot)
    print(f"bootstrap done at {time.time() - t0:.0f} s")
    mean_obs, sem_obs, _ = pauli_means(s1, bases, obs)
    mean2, sem2, _ = pauli_means(s2, bases, obs)
    jobs = [(th, rho0s[s], times) for th in (theta, symmetrized(theta, ans)) for s in (0, 1)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        runs = list(pool.map(validation_run, jobs))
    pred = {"full": runs[:2], "symmetric": runs[2:]}
    print(f"validation simulated at {time.time() - t0:.0f} s")
    return dict(times=times, bases=bases, margin=margin, thetas=np.array(thetas), theta=theta, boot=boot,
                data=np.stack([mean_obs, mean2]), sem=np.stack([sem_obs, sem2]),
                pred_full=np.stack(pred["full"]), pred_sym=np.stack(pred["symmetric"])), ans, obs


def analyse(r, ans, obs):
    d = derived(r["theta"], ans)
    db = [derived(t, ans) for t in r["boot"]]
    sd = lambda key: np.std([x[key] for x in db], axis=0, ddof=1)
    acc = {"bases X/Y/Z": np.bincount(r["bases"].ravel(), minlength=3).tolist(),
           "basis margin min (SEM)": float(r["margin"].min()),
           "settings-qubits with margin < 5": int((r["margin"] < 5).sum()),
           "defect rounds max|dtheta| (rad/s)": [float(np.abs(a - b).max() * RAD_S)
                                                 for a, b in zip(r["thetas"][1:], r["thetas"][:-1])]}
    acc["J0 fit (rad/s) [BEQS App. B estimate 576]"] = d["J0_fit"] * RAD_S
    acc["J0 fit bootstrap sd (rad/s)"] = float(sd("J0_fit") * RAD_S)
    acc["alpha fit [BEQS App. B 1.19; BEQS learned power-law fits 0.64(7), 0.7(1)]"] = d["alpha_fit"]
    acc["alpha fit bootstrap sd"] = float(sd("alpha_fit"))
    acc["nearest-neighbour J mean (rad/s)"] = float(d["J_d"][0] * RAD_S)
    acc["Gamma dominant eigenvalue (rad/s) [BEQS 460]"] = d["theta0"] * RAD_S
    acc["Gamma dominant eigenvalue sd (rad/s)"] = float(sd("theta0") * RAD_S)
    acc["dominant eigenvector overlap with uniform"] = d["v0_overlap"]
    acc["Gamma PSD-projected weight (rel. Frobenius)"] = d["psd_weight"]
    acc["gamma_c* (rad/s), strict Lemma B.1 on PSD projection"] = float(d["gamma_c_star"] * RAD_S)
    acc["gamma_c* bootstrap sd (rad/s)"] = float(sd("gamma_c_star") * RAD_S)
    acc["gamma_c structured (mean off-diagonal Gamma, rad/s)"] = d["gamma_c_struct"] * RAD_S
    acc["gamma_c structured bootstrap sd (rad/s)"] = float(sd("gamma_c_struct") * RAD_S)
    acc["gamma_c structured / J_nn"] = float(d["gamma_c_struct"] / d["J_d"][0])
    acc["local rates g_i = Gamma_ii - gamma_c (rad/s): min, mean, max"] = [
        float(x * RAD_S) for x in (d["g_local"].min(), d["g_local"].mean(), d["g_local"].max())]
    acc["local rates g_i: min z-score (g_i / sd)"] = float((d["g_local"] / sd("g_local")).min())
    zoff = (d["offdiag"] - d["gamma_c_struct"]) / sd("offdiag")
    acc["off-diagonal Gamma minus gamma_c: rms z-score (1 = consistent with collective+local)"] = float(
        np.sqrt(np.mean(zoff**2)))
    acc["emission rates mean (rad/s) [BEQS expects ~2]"] = float(d["emission"].mean() * RAD_S)
    acc["emission rates bootstrap sd per site, mean (rad/s)"] = float(sd("emission").mean() * RAD_S)
    up = np.triu_indices(N, 1)
    for key in ("K", "L", "D"):
        z = np.abs(d[key][up]) / sd(key)[up]
        acc[f"{key}: rms (rad/s), max |z|, #|z|>3 of 45"] = [float(np.sqrt(np.mean(d[key][up] ** 2)) * RAD_S),
                                                             float(z.max()), int((z > 3).sum())]
    for key in ("Bx", "By"):
        z = np.abs(d[key]) / sd(key)
        acc[f"{key}: rms (rad/s), max |z|, #|z|>3 of 10"] = [float(np.sqrt(np.mean(d[key] ** 2)) * RAD_S),
                                                             float(z.max()), int((z > 3).sum())]
    acc["Bz: mean (rad/s), spread (rad/s)"] = [float(d["Bz"].mean() * RAD_S), float(np.ptp(d["Bz"]) * RAD_S)]
    acc["J_sym rms (rad/s)"] = float(np.sqrt(np.mean(d["J"][up] ** 2)) * RAD_S)
    t = r["times"]
    for win, mask in (("learning t<=1.5", (t > 0) & (t <= T_LEARN)), ("validation t>1.5", t > T_LEARN)):
        for name in ("full", "sym"):
            resid = (r[f"pred_{name}"] - r["data"])[:, mask]
            acc[f"rms(model - data), {win}, {name} model"] = float(np.sqrt(np.nanmean(resid**2)))
        acc[f"rms SEM, {win}"] = float(np.sqrt(np.nanmean(r["sem"][:, mask] ** 2)))
    return d, db, acc


def plot(r, d, db, obs):
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), gridspec_kw={"width_ratios": [1.15, 1.0, 1.25]})
    ax = axes[0]
    dist = np.array([j - i for i, j in PAIRS])
    up = np.triu_indices(N, 1)
    sdJ = np.std([x["J"][up] for x in db], axis=0, ddof=1)
    ax.errorbar(dist, d["J"][up] * RAD_S, yerr=sdJ * RAD_S, fmt="o", ms=3, color=CAT[0], lw=0.8,
                label="$J_{ij}$ (U(1)-symmetric)")
    for key, col, lab in (("K", CAT[1], "$|(J^{xx}-J^{yy})/2|$ (charge ±2)"), ("L", CAT[2], "$|(J^{xy}+J^{yx})/2|$ (charge ±2)")):
        sd = np.std([x[key][up] for x in db], axis=0, ddof=1)
        ax.errorbar(dist + (0.12 if key == "K" else 0.24), np.abs(d[key][up]) * RAD_S, yerr=sd * RAD_S, fmt="s",
                    ms=2.5, color=col, lw=0.8, label=lab)
    dd = np.linspace(1, N - 1, 50)
    ax.plot(dd, d["J0_fit"] * RAD_S / dd ** d["alpha_fit"], color=CAT[0], lw=1.0,
            label=f"fit {d['J0_fit'] * RAD_S:.0f}/d$^{{{d['alpha_fit']:.2f}}}$ rad/s")
    ax.set_xlabel("ion distance  d = |i − j|")
    ax.set_ylabel("coupling  [rad/s]")
    ax.set_title("learned couplings (bootstrap 1σ)", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper right")
    ax = axes[1]
    vmax = np.abs(d["Gamma"]).max() * RAD_S
    im = ax.imshow(d["Gamma"] * RAD_S, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(range(1, N + 1), fontsize=6.5)
    ax.set_yticklabels(range(1, N + 1), fontsize=6.5)
    ax.grid(False)
    ax.set_title(f"learned dephasing matrix Γ [rad/s];  collective γ_c = {d['gamma_c_struct'] * RAD_S:.0f} rad/s",
                 fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax = axes[2]
    t = r["times"]
    picks = [(1, ((4, "Y"),), "state 2: ⟨Y₅⟩"), (0, ((4, "Z"),), "state 1: ⟨Z₅⟩"), (1, ((4, "X"), (5, "X")), "state 2: ⟨X₅X₆⟩")]
    for (s, p, lab), col in zip(picks, CAT):
        k = obs.index(p)
        ax.errorbar(t, r["data"][s, :, k], yerr=r["sem"][s, :, k], fmt="o", ms=3, color=col, lw=0.8, label=lab)
        ax.plot(t, r["pred_full"][s, :, k], color=col, lw=1.2)
    ax.axvspan(0, T_LEARN, color="#e4e3df", alpha=0.6, lw=0)
    ax.text(0.1, 0.04, "learning\nwindow", transform=ax.get_xaxis_transform(), fontsize=7, color=MUTED)
    ax.set_xlabel("time  [ms]")
    ax.set_title("data (dots) vs learned model (lines)", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper right", ncol=1)
    fig.suptitle("E6a  re-learning the BEQS 10-ion generator from the public raw shots", fontsize=10, y=1.03)
    fig.tight_layout()
    return save_figure(fig, "e6_ion_learning")


if __name__ == "__main__":
    r, ans, obs = run()
    d, db, acc = analyse(r, ans, obs)
    np.savez_compressed(RESULTS / "e6_ion_learning.npz", theta=r["theta"], boot=r["boot"], thetas=r["thetas"],
                        h_terms=np.array([repr(p) for p in ans.h_terms]))
    save_json(RESULTS / "e6_ion_learning.json", {"acceptance": acc})
    print(plot(r, d, db, obs))
    for k, v in acc.items():
        print(f"{k}: {v}")
