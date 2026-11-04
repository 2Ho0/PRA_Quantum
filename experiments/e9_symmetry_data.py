"""E9: weak versus strong U(1) symmetry in the BEQS data, and a model-conditional certified collective rate.

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0).
(a) Extended ansatz: the E6a Hamiltonian and sigma^z-dephasing matrix plus the full single-site Kossakowski block
    on every ion (8 parameters each, replacing sigma^- emission). Its superoperator charges separate covariant noise
    (Q = 0: emission and absorption, which break only the strong symmetry) from non-covariant noise (Q = +-1, +-2,
    which break the weak symmetry).
(b) Structured dephasing Gamma = gamma_c 11^T + diag(g), g >= 0, on the E6a ansatz and on the extended one: point
    estimate, bootstrap, and the 95% one-sided lower bound of gamma_c. Under that model gamma_c* >= gamma_c
    (Lemma B.1), so the lower bound certifies the Zeno factors conditionally on the model.
(c) PSD-constrained full Gamma and the strict gamma_c* of Lemma B.1 on every bootstrap replicate.
(d) Shadow feasibility: within-setting randomized-measurement purity of the first k qubits at t = 0.
(e) 51-ion strong-symmetry rates from results/e6_symmetry.json.
The quadrature defect is computed once from the E6a model (results/e6_ion_learning.npz) and reused for every fit.
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import CAT, INK, MUTED, RESULTS, save_figure, save_json, style
from e6_ion_learning import (DEFECT, N, RAD_S, RAW, T_LEARN, ansatz, derived, observables, per_setting_tables,
                             product_state, series_from)
from qerr.bounds import _phi
from qerr.charge import protective_collective_rate
from qerr.iondata import infer_bases, load_10ion
from qerr.learning import (NOISE_KEYS, Ansatz, fit, fit_bounded, fit_psd, flip_rates, gamma_matrix, integral_system,
                           needed_strings, noise_charges, quadrature_defect, site_kossakowski, structured_map,
                           to_openmodel)

N_BOOT, WORKERS, T_ZENO = 100, 12, 10.0
COMBOS = {"flip S=SXX+SYY": (("SXX", 1), ("SYY", 1)), "imbalance TXY": (("TXY", 1),),
          "SXX-SYY": (("SXX", 1), ("SYY", -1)), "SXY": (("SXY", 1),),
          "SXZ": (("SXZ", 1),), "TXZ": (("TXZ", 1),), "SYZ": (("SYZ", 1),), "TYZ": (("TYZ", 1),)}
_W = {}


def extended_ansatz():
    a = ansatz()
    return Ansatz(n=N, h_terms=a.h_terms, deph_pairs=a.deph_pairs, emission_sites=[], noise_sites=list(range(N)))


def e6a_system(A_ext, ext):
    """Columns of the E6a ansatz from the extended system: an emission site at rate g is SXX = SYY = TXY = g/4."""
    k = len(ext.h_terms) + len(ext.deph_pairs)
    em = [(A_ext[:, k + 8 * m] + A_ext[:, k + 8 * m + 1] + A_ext[:, k + 8 * m + 3]) / 4 for m in range(N)]
    return np.column_stack([A_ext[:, :k]] + em)


def structured_lower(ans):
    P = structured_map(ans)
    lower = np.full(P.shape[1], -np.inf)
    k0 = len(ans.h_terms)
    lower[k0 + 1:k0 + 1 + ans.n] = 0.0  # local rates g_i >= 0; gamma_c and the rest free
    return lower


def _init(tables, times, defect):
    _W.update(tables=tables, times=times, defect=defect)


def replicate(w):
    """Every fit for one bootstrap weight vector (None = the data themselves)."""
    ext, base = extended_ansatz(), ansatz()
    A, y, _ = integral_system(observables(), ext, series_from(_W["tables"], w), _W["times"], T_LEARN,
                              defect=_W["defect"])
    A0 = e6a_system(A, ext)
    return {"ext": fit(A, y), "base": fit(A0, y),
            "struct": fit_bounded(A0 @ structured_map(base), y, structured_lower(base)),
            "struct_ext": fit_bounded(A @ structured_map(ext), y, structured_lower(ext)),
            "psd": fit_psd(A0, y, base)}


def combo_values(theta_ext, ext):
    """Per-site values of the charge-definite noise combinations (rad/ms)."""
    k = len(ext.h_terms) + len(ext.deph_pairs)
    block = theta_ext[k:].reshape(N, len(NOISE_KEYS))
    col = {key: block[:, u] for u, key in enumerate(NOISE_KEYS)}
    return {name: sum(sign * col[key] for key, sign in combo) for name, combo in COMBOS.items()}


def zeno(gamma_c, q, t=T_ZENO):
    return float(_phi(2 * gamma_c * q**2, t) / (t**2 / 2))


def setting_purities(shots, k):
    """Unbiased within-setting estimator of Tr rho_A^2, A = first k qubits: pair weight prod_j (2 if equal else -1)."""
    S, M, _ = shots.shape
    out = np.empty(S)
    for s in range(S):
        x = shots[s, :, :k]
        W = np.where(x[:, None, :] == x[None, :, :], 2.0, -1.0).prod(axis=2)
        out[s] = (W.sum() - np.trace(W)) / (M * (M - 1))
    return out


def purity_floor(bloch, S):
    """sd over S random-Pauli settings of the infinite-shot estimator for a product state with Bloch vectors r_j.

    Per site x = 1/2 + 3/2 (r.a)^2 for the measured axis a: E x = (1 + |r|^2)/2, E x^2 = 1/4 + |r|^2/2 + 3/4 sum r_a^4.
    """
    r2 = (bloch**2).sum(axis=1)
    m1 = (1 + r2) / 2
    m2 = 0.25 + 0.5 * r2 + 0.75 * (bloch**4).sum(axis=1)
    return np.sqrt(np.maximum(np.cumprod(m2) - np.cumprod(m1**2), 0.0) / S), np.cumprod(m1)


def contrast_corrected(series):
    """Every Pauli mean divided by prod_j c_{j,P_j}, c from t = 0 under the nominal initial states (a SPAM check)."""
    th = 0.33 * np.pi
    c = {}
    for j in range(N):
        c[(j, "X")] = series[(0, ((j, "X"),))][0] / np.sin(th)
        c[(j, "Z")] = series[(0, ((j, "Z"),))][0] / -np.cos(th)
        c[(j, "Y")] = series[(1, ((j, "Y"),))][0]
    corr = {key: v / np.prod([c[p] for p in key[1]]) for key, v in series.items() if key[1] != ()}
    return corr, np.array([[c[(j, a)] for a in "XYZ"] for j in range(N)])


def charge_resolved_scale(k, q=1):
    """Pi_q = ||P_q rho||_2^2 of a pure product state on the equator: C(2k, k+q) / 4^k."""
    from math import comb
    return comb(2 * k, k + q) / 4**k


def run():
    t0 = time.time()
    times, s1 = load_10ion(RAW, "state1")
    _, s2 = load_10ion(RAW, "state2")
    stats = lambda s: (s[0].mean(axis=1).astype(float), s[0].std(axis=1, ddof=1) / np.sqrt(s.shape[2]))
    (m1, e1), (m2, e2) = stats(s1), stats(s2)
    bases, _ = infer_bases(m1, m2, e1, e2)
    ext, base, obs = extended_ansatz(), ansatz(), observables()
    strings = sorted(set(obs) | set(needed_strings(obs, ext)))
    tables = per_setting_tables((s1, s2), bases, strings)
    series = series_from(tables)
    rho0s = [product_state(series, s) for s in (0, 1)]
    theta_e6a = np.load(RESULTS / "e6_ion_learning.npz")["theta"]
    defect = quadrature_defect(to_openmodel(theta_e6a, base, sparse=True), rho0s, needed_strings(obs, ext), times,
                               T_LEARN, **DEFECT)
    print(f"defect from the E6a model at {time.time() - t0:.0f} s", flush=True)
    rng = np.random.default_rng(2027)
    n_set = bases.shape[0]
    weights = [None] + [np.bincount(rng.integers(0, n_set, n_set), minlength=n_set).astype(float)
                        for _ in range(N_BOOT)]
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init, initargs=(tables, times, defect)) as pool:
        fits = list(pool.map(replicate, weights))
    print(f"{len(fits)} fits at {time.time() - t0:.0f} s", flush=True)
    bloch = [np.array([[series[(s, ((i, a),))][0] for a in "XYZ"] for i in range(N)]) for s in (0, 1)]
    purity = {}
    for s, shots in ((0, s1[0]), (1, s2[0])):
        for k in range(1, N + 1):
            est = setting_purities(shots, k)
            purity[(s, k)] = (float(est.mean()), float(est.std(ddof=1) / np.sqrt(len(est))))
    print(f"purities at {time.time() - t0:.0f} s", flush=True)
    corr, contrast = contrast_corrected(series)
    A, y, _ = integral_system(observables(), ext, corr, times, T_LEARN, defect=defect)
    return fits, bloch, purity, n_set, {"theta": fit(A, y), "contrast": contrast}


def analyse(fits, bloch, purity, n_set, spam):
    ext, base = extended_ansatz(), ansatz()
    k0 = len(base.h_terms)
    central, boot = fits[0], fits[1:]
    stack = lambda key: np.array([f[key] for f in boot])
    out = {}
    # (a) dissipator charges
    cv = combo_values(central["ext"], ext)
    bv = [combo_values(th, ext) for th in stack("ext")]
    charges = noise_charges()
    zs = {}
    for name, combo in COMBOS.items():
        sd = np.std([b[name] for b in bv], axis=0, ddof=1)
        z = cv[name] / sd
        zs[name] = z
        Q = sorted(charges[combo])
        out[f"(a) {name} (Q={Q}): rms (rad/s), max |z|, #|z|>3 of 10"] = [
            float(np.sqrt(np.mean(cv[name] ** 2)) * RAD_S), float(np.abs(z).max()), int((np.abs(z) > 3).sum())]
    noncov = np.concatenate([zs[n] for n in ("SXX-SYY", "SXY", "SXZ", "TXZ", "SYZ", "TYZ")])
    out["(a) non-covariant noise: #|z|>3 of 60, rms z"] = [int((np.abs(noncov) > 3).sum()),
                                                           float(np.sqrt(np.mean(noncov**2)))]
    em, ab = flip_rates(site_kossakowski(central["ext"], ext))
    em_b, ab_b = zip(*[flip_rates(site_kossakowski(th, ext)) for th in stack("ext")])
    tot = (em + ab).mean()
    tot_b = np.array([(a + b).mean() for a, b in zip(em_b, ab_b)])
    out["(a) emission rate, site mean (rad/s) +- sd"] = [float(em.mean() * RAD_S),
                                                       float(np.std([e.mean() for e in em_b], ddof=1) * RAD_S)]
    out["(a) absorption rate, site mean (rad/s) +- sd"] = [float(ab.mean() * RAD_S),
                                                         float(np.std([a.mean() for a in ab_b], ddof=1) * RAD_S)]
    out["(a) total flip rate per ion (rad/s): estimate, bootstrap 95th percentile"] = [
        float(tot * RAD_S), float(np.percentile(tot_b, 95) * RAD_S)]
    s_mean = np.array([0.5 * (a + b).mean() for a, b in zip(em_b, ab_b)])
    out["(a) site-mean flip S = SXX+SYY (rad/s): estimate, bootstrap mean, bias-corrected, sd (negative = unphysical)"] = [
        float(0.5 * tot * RAD_S), float(s_mean.mean() * RAD_S), float((tot - s_mean.mean()) * RAD_S),
        float(np.std(s_mean, ddof=1) * RAD_S)]
    cs = combo_values(spam["theta"], ext)
    em_s, ab_s = flip_rates(site_kossakowski(spam["theta"], ext))
    out["(a') SPAM check, contrast c_{j,a} from t = 0 (nominal states): min, max over ions for X, Y, Z"] = [
        [float(spam["contrast"][:, u].min()), float(spam["contrast"][:, u].max())] for u in range(3)]
    out["(a') contrast-corrected refit: combination, site mean (rad/s), rms z (raw bootstrap sd), #|z|>3"] = {
        name: [float(cs[name].mean() * RAD_S), float(np.sqrt(np.mean((cs[name] / np.std(
            [b[name] for b in bv], axis=0, ddof=1)) ** 2))), int((np.abs(cs[name] / np.std(
                [b[name] for b in bv], axis=0, ddof=1)) > 3).sum())] for name in COMBOS}
    out["(a') contrast-corrected refit: emission, absorption site means (rad/s)"] = [
        float(em_s.mean() * RAD_S), float(ab_s.mean() * RAD_S)]
    to_base = lambda th: np.concatenate([th[:k0 + len(base.deph_pairs)], np.zeros(N)])
    d_ext = derived(to_base(central["ext"]), base)
    db_ext = [derived(to_base(th), base) for th in stack("ext")]
    up = np.triu_indices(N, 1)
    for key in ("K", "L"):
        z = np.abs(d_ext[key][up]) / np.std([x[key][up] for x in db_ext], axis=0, ddof=1)
        out[f"(a) Hamiltonian {key} in the extended fit: max |z|, #|z|>3 of 45"] = [float(z.max()), int((z > 3).sum())]
    for key in ("Bx", "By"):
        z = np.abs(d_ext[key]) / np.std([x[key] for x in db_ext], axis=0, ddof=1)
        out[f"(a) Hamiltonian {key} in the extended fit: max |z|, #|z|>3 of 10"] = [float(z.max()), int((z > 3).sum())]
    # (b) structured collective rate
    gam = {}
    for key, label in (("struct", "E6a ansatz"), ("struct_ext", "extended ansatz")):
        g_c, g_b = central[key][k0], stack(key)[:, k0]
        lo_pct = float(np.percentile(g_b, 5))
        lo_basic = float(2 * g_c - np.percentile(g_b, 95))  # reverse percentile: corrects the bootstrap bias
        lo = min(lo_pct, lo_basic)
        gam[key] = (g_c, g_b, lo)
        out[f"(b) structured gamma_c, {label} (rad/s): estimate, bootstrap mean, sd, 95% lower bound "
            f"(percentile, basic), certified = min"] = [
            float(g_c * RAD_S), float(g_b.mean() * RAD_S), float(np.std(g_b, ddof=1) * RAD_S), lo_pct * RAD_S,
            lo_basic * RAD_S, lo * RAD_S]
        out[f"(b) Zeno factor at {T_ZENO:g} ms, {label}: q=1, q=2 at estimate; q=1, q=2 at certified bound"] = [
            zeno(g_c, 1), zeno(g_c, 2), zeno(lo, 1), zeno(lo, 2)]
        g_loc = central[key][k0 + 1:k0 + 1 + N]
        out[f"(b) local rates g_i, {label} (rad/s): min, mean, max, #zero"] = [
            float(g_loc.min() * RAD_S), float(g_loc.mean() * RAD_S), float(g_loc.max() * RAD_S),
            int((g_loc < 1e-9).sum())]
    G_un = gamma_matrix(central["base"], base)
    off = G_un[up]
    out["(b) mean off-diagonal Gamma of the unconstrained fit (rad/s) [E6a: 30.5]"] = float(off.mean() * RAD_S)
    # (c) PSD-constrained full Gamma
    gstar = np.array([protective_collective_rate(gamma_matrix(th, base)) for th in stack("psd")])
    G_psd = gamma_matrix(central["psd"], base)
    w = np.linalg.eigvalsh(G_psd)
    out["(c) PSD fit: strict gamma_c* (rad/s), eigenvalues of Gamma (rad/s)"] = [
        float(protective_collective_rate(G_psd) * RAD_S), [float(x * RAD_S) for x in w]]
    out["(c) strict gamma_c* over bootstrap: fraction zero, median, 95th percentile (rad/s)"] = [
        float(np.mean(gstar < 1e-9)), float(np.median(gstar) * RAD_S), float(np.percentile(gstar, 95) * RAD_S)]
    # (d) shadow feasibility
    for s, name in ((0, "state 1"), (1, "state 2")):
        floor, prod_purity = purity_floor(bloch[s], n_set)
        rows = []
        for k in range(1, N + 1):
            mean, sd = purity[(s, k)]
            rows.append([k, mean, sd, float(floor[k - 1]), float(prod_purity[k - 1])])
        out[f"(d) {name}: k, purity estimate, sd, setting floor sd, product-state purity"] = rows
    out["(d) charge-resolved purity Pi_1 of an equatorial product state at k = 10"] = charge_resolved_scale(N)
    # (e) 51 ions
    e6b = json.loads((RESULTS / "e6_symmetry.json").read_text())["acceptance"]
    m, sm = e6b["51 ions: slope of M per ms (+- 1 se)"]
    v, sv = e6b["51 ions: slope of varM per ms (+- 1 se)"]
    out["(e) 51 ions: <M> drift vs D5/2 decay (per ms)"] = [m, sm, e6b[
        "expected |d<M>/dt| from D5/2 decay alone, 25.5 / tau (per ms)"]]
    out["(e) 51 ions: balanced sector-changing rate per ion (1/s) = (varM slope - |<M> slope|)/51"] = [
        float((v - abs(m)) / 51 * 1e3), float(np.hypot(sv, sm) / 51 * 1e3)]
    return out, {"cv": cv, "bv": bv}, gam, gstar


def plot(combos, gam, gstar, out):
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), gridspec_kw={"width_ratios": [1.25, 1.0, 1.05]})
    ax = axes[0]
    cols = [("flip S=SXX+SYY", "$S_{xx}{+}S_{yy}$", CAT[0]), ("imbalance TXY", "$T_{xy}$", CAT[0]),
            ("SXZ", "$S_{xz}$", CAT[1]), ("TXZ", "$T_{xz}$", CAT[1]), ("SYZ", "$S_{yz}$", CAT[1]),
            ("TYZ", "$T_{yz}$", CAT[1]), ("SXX-SYY", "$S_{xx}{-}S_{yy}$", CAT[2]), ("SXY", "$S_{xy}$", CAT[2])]
    rng = np.random.default_rng(1)
    for x, (key, _, col) in enumerate(cols):
        vals = combos["cv"][key] * RAD_S
        sd_mean = np.std([b[key].mean() for b in combos["bv"]], ddof=1) * RAD_S
        ax.plot(x + 0.13 * rng.uniform(-1, 1, N), vals, "o", ms=2.8, color=col, alpha=0.6, lw=0)
        ax.errorbar(x, vals.mean(), yerr=2 * sd_mean, fmt="_", ms=14, mew=1.8, color=INK, elinewidth=1.2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[1] for c in cols], fontsize=7.5)
    for x0, x1, lab, col in ((-0.4, 1.4, "Q = 0", CAT[0]), (1.6, 5.4, "Q = ±1", CAT[1]), (5.6, 7.4, "Q = ±2", CAT[2])):
        ax.plot([x0, x1], [1.03, 1.03], color=col, lw=1.6, transform=ax.get_xaxis_transform(), clip_on=False)
        ax.text((x0 + x1) / 2, 1.06, lab, transform=ax.get_xaxis_transform(), ha="center", fontsize=7.5, color=col)
    ax.text(0.02, 0.04, r"a valid generator has $S_{xx}{+}S_{yy}\geq 0$", transform=ax.transAxes, fontsize=7,
            color=MUTED)
    ax.set_ylabel("estimate per ion  [rad/s]")
    ax.set_title("noise beyond σᶻ dephasing: ions (dots), mean ± 2 s.e. (bars)", fontsize=9, pad=20)
    ax = axes[1]
    rng = np.random.default_rng(0)
    entries = [("structured γ_c\n(E6a ansatz)", gam["struct"][1], gam["struct"][2], CAT[0]),
               ("structured γ_c\n(extended)", gam["struct_ext"][1], gam["struct_ext"][2], CAT[1]),
               ("strict γ_c*\n(PSD full Γ)", gstar, None, CAT[2])]
    for x, (label, vals, lo, col) in enumerate(entries):
        ax.plot(x + 0.18 * rng.uniform(-1, 1, len(vals)), vals * RAD_S, "o", ms=2.4, color=col, alpha=0.6, lw=0)
        if lo is not None:
            ax.plot([x - 0.3, x + 0.3], [lo * RAD_S] * 2, color=col, lw=1.8)
    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels([e[0] for e in entries], fontsize=7.5)
    ax.set_ylabel("collective rate  [rad/s]")
    ax.set_title("bootstrap replicates; bar = certified 95% lower bound", fontsize=9)
    ax = axes[2]
    ks = np.arange(1, N + 1)
    for s, (name, col) in enumerate((("state 1", CAT[0]), ("state 2", CAT[1]))):
        rows = np.array(out[f"(d) {name}: k, purity estimate, sd, setting floor sd, product-state purity"])
        ax.semilogy(ks, rows[:, 2], "o", ms=4, color=col, label=f"{name}: sd of estimate")
        ax.semilogy(ks, rows[:, 3], color=col, lw=1.0, label=f"{name}: setting floor")
    ax.semilogy(ks, [charge_resolved_scale(k) for k in ks], color=MUTED, ls="--", lw=1.0,
                label="signal: Π₁ of an equatorial product state")
    ax.set_xticks(ks)
    ax.set_xlabel("subsystem size k (qubits 1..k)")
    ax.set_title("randomized-measurement purity at t = 0 (200 settings)", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left")
    fig.suptitle("E9  weak vs strong U(1) symmetry in the BEQS 10-ion data; certified collective rate", fontsize=10,
                 y=1.03)
    fig.tight_layout()
    return save_figure(fig, "e9_symmetry_data")


FIT_KEYS = ("ext", "base", "struct", "struct_ext", "psd")

if __name__ == "__main__":
    if "--replot" not in sys.argv:  # --replot: analyse and plot the cached fits only
        fits, bloch, purity, n_set, spam = run()
        np.savez_compressed(RESULTS / "e9_symmetry_data.npz", bloch=np.array(bloch), n_set=n_set,
                            spam_theta=spam["theta"], spam_contrast=spam["contrast"],
                            purity=np.array([[purity[(s, k)] for k in range(1, N + 1)] for s in (0, 1)]),
                            **{key: np.array([f[key] for f in fits]) for key in FIT_KEYS})
    z = np.load(RESULTS / "e9_symmetry_data.npz")
    fits = [{key: z[key][i] for key in FIT_KEYS} for i in range(z["ext"].shape[0])]
    purity = {(s, k): tuple(z["purity"][s, k - 1]) for s in (0, 1) for k in range(1, N + 1)}
    spam = {"theta": z["spam_theta"], "contrast": z["spam_contrast"]}
    out, combos, gam, gstar = analyse(fits, list(z["bloch"]), purity, int(z["n_set"]), spam)
    save_json(RESULTS / "e9_symmetry_data.json", {"results": out})
    print(plot(combos, gam, gstar, out))
    for k, v in out.items():
        print(f"{k}: {v}")
