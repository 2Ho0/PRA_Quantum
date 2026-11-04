"""E3: BEQS-like learned-uncertainty ensemble (Corollary B1', Theorems A2/B3 in practice).

Gaussian uncertainty sigma per coefficient of a 2-local charge-definite basis (range <= 2;
M = 99 terms, of which M_sym = 33 are U(1)-symmetric). Three ensembles: all terms random,
symmetric terms only, charged terms only. Antithetic pairs make the ensemble-mean shift a
low-variance estimate of the second-order bias.

What is tested (and where it can hold):
  * B1': charged parameters enter only at second order, so in the perturbative window
    (sigma sqrt(M) t << 1) the charged band is O(sigma^2 t^2) while the symmetric band is
    O(sigma t); their ratio scales like sigma. Outside that window (BEQS-like sigma, late t)
    the separation is lost, which is reported, not hidden.
  * Symmetric imperfections never leave the charge-0 sector, so their ensemble is exactly
    independent of the collective-dephasing rate gamma_c.
  * Charged imperfections are Zeno-suppressed by gamma_c at late times.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import CAT, INK, MUTED, RESULTS, save_figure, base_model, neel, ops, save_json, style
from qerr.bounds import beqs_concentration_bound
from qerr.lindblad import OpenModel, evolve, expect
from qerr.models import charge_basis_terms, delta_h, sample_coeffs

N, N_PAIRS = 6, 50
SIGMAS = [0.03, 0.01]              # BEQS-like (sigma ~ 13 rad/s at J_nn ~ 400 rad/s) and well calibrated
GAMMA_CS = [0.0, 0.1, 1.0]          # 0.1 ~ BEQS 10-ion collective rate in units of J_nn
GAMMA_LOC, GAMMA_EM = 0.03, 0.005
T = np.concatenate([[0.0], np.geomspace(0.05, 20.0, 60)])
T_PERT = 1.0                        # early time used for the sigma-scaling test
ENSEMBLES = {"all terms": (0, 1, 2), "symmetric only (|q|=0)": (0,), "charged only (|q|>0)": (1, 2)}


def observables():
    o2, o3 = ops(N, 2), ops(N, 3)
    stag = sum((-1) ** i * ops(N, i)["z"] for i in range(N)) / N
    return {"Z2 Z3": o2["z"] @ o3["z"], "staggered magnetization": stag}


def simulate(args):
    H, Gamma, emission = args
    rhos = evolve(OpenModel(H=H, Gamma=Gamma, emission=emission), neel(N), T, rtol=1e-9, atol=1e-11)
    return np.array([expect(rhos, O).real for O in observables().values()])


def run():
    terms = charge_basis_terms(N, max_range=2)
    charges = np.array([t.abs_charge for t in terms])
    jobs, keys = [], []
    for sigma in SIGMAS:
        g = sample_coeffs(len(terms), sigma, N_PAIRS, np.random.default_rng(2026))
        for gc in GAMMA_CS:
            model = base_model(N, gc, GAMMA_LOC, GAMMA_EM)
            jobs.append((model.H, model.Gamma, model.emission))
            keys.append((sigma, gc, "base", -1))
            for ens, qs in ENSEMBLES.items():
                mask = np.isin(charges, qs)
                for s in range(len(g)):
                    jobs.append((model.H + delta_h(terms, g[s] * mask), model.Gamma, model.emission))
                    keys.append((sigma, gc, ens, s))
    with ProcessPoolExecutor(max_workers=min(20, os.cpu_count() or 1)) as pool:
        raw = dict(zip(keys, pool.map(simulate, jobs, chunksize=8)))
    out = {"M": len(terms), "M_sym": int((charges == 0).sum())}
    for sigma in SIGMAS:
        for gc in GAMMA_CS:
            base = raw[(sigma, gc, "base", -1)]
            for ens in ENSEMBLES:
                dev = np.array([raw[(sigma, gc, ens, s)] for s in range(2 * N_PAIRS)]) - base[None]
                out[(sigma, gc, ens)] = {
                    "mean_shift": np.abs(dev.mean(axis=0)),
                    "half_width95": 0.5 * (np.percentile(dev, 97.5, axis=0) - np.percentile(dev, 2.5, axis=0)),
                    "dev": dev}
    return out


def acceptance(out):
    acc = {}
    it = np.searchsorted(T, T_PERT)
    for i, name in enumerate(observables()):
        hw = lambda s, e: out[(s, 0.1, e)]["half_width95"][i]
        ratio = {s: hw(s, "charged only (|q|>0)")[it] / hw(s, "symmetric only (|q|=0)")[it] for s in SIGMAS}
        bound_sym = beqs_concentration_bound(T, SIGMAS[0], out["M_sym"])
        informative = (bound_sym < 1.0) & (T > 0)
        covered = bool(np.all(hw(SIGMAS[0], "all terms")[informative] <= bound_sym[informative]))
        m = [out[(SIGMAS[0], gc, "charged only (|q|>0)")]["mean_shift"][i][-1] for gc in GAMMA_CS]
        sym = [out[(s, gc, "symmetric only (|q|=0)")]["dev"][:, i] for s in SIGMAS[:1] for gc in GAMMA_CS]
        acc[name] = {
            f"ch/sym width ratio at t={T[it]:.2f}, sigma=0.03": float(ratio[0.03]),
            f"ch/sym width ratio at t={T[it]:.2f}, sigma=0.01": float(ratio[0.01]),
            "B1': ratio(0.01)/ratio(0.03) in [0.2, 0.5] (expect 1/3)": bool(0.2 <= ratio[0.01] / ratio[0.03] <= 0.5),
            "BEQS Eq.(11) with M_sym covers observed all-terms band wherever bound < 1": covered,
            "late-time (t=20) ch/sym width ratio, sigma=0.03": float(hw(0.03, "charged only (|q|>0)")[-1]
                                                                     / hw(0.03, "symmetric only (|q|=0)")[-1]),
            "Zeno: charged mean shift at t=20 decreases gc 0 -> 0.1 -> 1": bool(m[0] > m[1] > m[2]),
            "charged mean shift at t=20 for gc = 0, 0.1, 1": [float(x) for x in m],
            "symmetric ensemble independent of gamma_c (max diff)": float(max(np.abs(d - sym[0]).max() for d in sym)),
        }
    return acc


def plot(out):
    plt = style()
    obs = list(observables())
    fig, axes = plt.subplots(len(obs), 3, figsize=(12, 6.6), sharex=True)
    M, Ms, s0 = out["M"], out["M_sym"], SIGMAS[0]
    pos = lambda y: np.where(y > 0, y, np.nan)
    for i, name in enumerate(obs):
        ax = axes[i, 0]
        for ens, col in zip(ENSEMBLES, CAT):
            ax.loglog(T, pos(out[(s0, 0.1, ens)]["half_width95"][i]), color=col, label=ens)
        ax.loglog(T, pos(beqs_concentration_bound(T, s0, M)), color=INK, ls=":", lw=1.2, label=f"BEQS Eq.(11), M={M}")
        ax.loglog(T, pos(beqs_concentration_bound(T, s0, Ms)), color=MUTED, ls="-.", lw=1.2,
                  label=f"same with M_sym={Ms} (B1′)")
        ax.axhline(1.0, color=MUTED, lw=0.8)
        ax.set_ylim(1e-6, 30)
        ax.set_ylabel(f"⟨{name}⟩", fontsize=8.5)
        ax = axes[i, 1]
        for sigma, ls in zip(SIGMAS, ["-", "--"]):
            with np.errstate(invalid="ignore"):  # 0/0 at t = 0
                r = (out[(sigma, 0.1, "charged only (|q|>0)")]["half_width95"][i]
                     / out[(sigma, 0.1, "symmetric only (|q|=0)")]["half_width95"][i])
            ax.loglog(T, pos(r), color=CAT[2], ls=ls, label=f"σ = {sigma}")
        ax.axhline(1.0, color=MUTED, lw=0.8)
        ax.set_ylim(1e-4, 10)
        ax = axes[i, 2]
        for gc, ls in zip(GAMMA_CS, ["-", "--", ":"]):
            ax.loglog(T, pos(out[(s0, gc, "charged only (|q|>0)")]["mean_shift"][i]), color=CAT[2], ls=ls,
                      label=f"charged only, γ_c = {gc}")
        ax.loglog(T, pos(out[(s0, 0.1, "symmetric only (|q|=0)")]["mean_shift"][i]), color=CAT[1],
                  label="symmetric only (identical for all γ_c)")
        ax.set_ylim(1e-8, 1)
    axes[0, 0].set_title(f"95% half-width, σ={s0}, γ_c=0.1", fontsize=9)
    axes[0, 1].set_title("charged / symmetric half-width (B1′: ∝ σ early)", fontsize=9)
    axes[0, 2].set_title(f"ensemble-mean shift (bias), σ={s0}", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("time  t  [1/J0]")
    axes[0, 0].legend(fontsize=6.8, loc="lower right")
    axes[0, 1].legend(fontsize=6.8, loc="upper left")
    axes[0, 2].legend(fontsize=6.8, loc="lower right")
    fig.suptitle(f"E3  learned-uncertainty ensemble, N={N}, M={M} 2-local terms, {2 * N_PAIRS} antithetic samples",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    return save_figure(fig, "e3_ensemble")


if __name__ == "__main__":
    out = run()
    acc = acceptance(out)
    save_json(RESULTS / "e3_ensemble.json", {"M": out["M"], "M_sym": out["M_sym"], "acceptance": acc})
    np.savez(RESULTS / "e3_mean_shift.npz", T=T, **{f"s{s}_g{gc}_{ens.split()[0]}": out[(s, gc, ens)]["mean_shift"]
                                                    for s in SIGMAS for gc in GAMMA_CS for ens in ENSEMBLES})
    print(plot(out))
    for name, a in acc.items():
        print(name)
        for k, v in a.items():
            print(f"   {k}: {v}")
