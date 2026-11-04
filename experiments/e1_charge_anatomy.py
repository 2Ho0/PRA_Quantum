"""E1: charge-resolved error anatomy (Theorems A, B1, B2, B3).

N = 6 long-range XY chain, Neel initial state (charge 0), collective dephasing sweep.
Imperfections of fixed operator norm in three charge classes (|q| = 0, 1, 2) are
probed with observables of charge 0, |q| = 1 and |q| = 2.
"""

import numpy as np

from common import (CAT, MUTED, ORDINAL, RESULTS, save_figure, base_model, charge_norms, neel, ops,
                    random_perturbation, save_json, style)
from qerr.bounds import charged_observable_bound, forgetting_bound, opnorm, osc, zeno_bound
from qerr.charge import protective_collective_rate
from qerr.lindblad import OpenModel, evolve, expect

N = 6
GAMMA_CS = [0.0, 0.1, 0.5, 2.0]
EPS = [0.02, 0.04]
GAMMA_LOC, GAMMA_EM = 0.02, 0.005
T = np.linspace(0.0, 20.0, 81)

o2, o3 = ops(N, 2), ops(N, 3)
OBSERVABLES = {
    "O0 = Z2 Z3 (q=0)": (o2["z"] @ o3["z"], 0),
    "O1 = X2 (|q|=1)": (o2["x"], 1),
    "O2 = (X2X3-Y2Y3)/2 (|q|=2)": (0.5 * (o2["x"] @ o3["x"] - o2["y"] @ o3["y"]), 2),
}
PERTURBATIONS = {"P0 (|q|=0)": (0,), "P1 (|q|=1)": (1,), "P2 (|q|=2)": (2,)}


def reachable(q_obs, p):
    """From a charge-0 state, imperfections of charge +-qP only populate charges in qP * Z."""
    qP = PERTURBATIONS[p][0]
    return q_obs == 0 or (qP != 0 and q_obs % qP == 0)


def run():
    rho0 = neel(N)
    unit = {p: random_perturbation(N, qs, 1.0, seed=10 + k) for k, (p, qs) in enumerate(PERTURBATIONS.items())}
    data, flags = {}, {"bounds_hold": True, "exact_symmetry": True}
    for gc in GAMMA_CS:
        model = base_model(N, gc, GAMMA_LOC, GAMMA_EM)
        g_star = protective_collective_rate(model.Gamma)
        base = evolve(model, rho0, T)
        for p, dH1 in unit.items():
            for eps in EPS:
                dH = eps * dH1
                pert = evolve(OpenModel(H=model.H + dH, Gamma=model.Gamma, emission=model.emission), rho0, T)
                for o, (O, qO) in OBSERVABLES.items():
                    delta = np.abs(expect(pert, O) - expect(base, O))
                    rec = {"t": T, "delta": delta}
                    if eps == EPS[-1]:
                        bounds = {"Thm A": forgetting_bound(model, O, dH, T)}
                        if qO != 0:
                            On = charge_norms(N, O)
                            bounds["Thm B2"] = sum(charged_observable_bound(T, opnorm(dH), g_star, q, nq)
                                                   for q, nq in On.items())
                        elif p != "P0 (|q|=0)":
                            bounds["Thm B3"] = zeno_bound(T, opnorm(dH), charge_norms(N, dH), g_star, osc(O))
                        rec["bounds"] = bounds
                        tightest = np.min(np.vstack(list(bounds.values())), axis=0)
                        if np.any(delta > tightest + 1e-9):
                            flags["bounds_hold"] = False
                        if not reachable(qO, p) and delta.max() > 1e-9:
                            flags["exact_symmetry"] = False
                    data[(gc, p, eps, o)] = rec
    return data, flags


def metrics(data, flags):
    out = {"ratio_2eps_over_eps": {}, "max_delta_eps0.04": {}, "plateau_O1P1_times_gamma": {}}
    first_order_ok, sym_ok = True, True
    for (gc, p, eps, o), rec in data.items():
        if eps != EPS[0]:
            continue
        big = data[(gc, p, EPS[1], o)]["delta"].max()
        small = rec["delta"].max()
        if small < 1e-10:
            continue
        r = big / small
        key = f"gc={gc} | {p} | {o}"
        out["ratio_2eps_over_eps"][key] = r
        out["max_delta_eps0.04"][key] = big
        if o.startswith("O0") and p != "P0 (|q|=0)" and not 3.6 <= r <= 4.4:
            first_order_ok = False
        if o.startswith("O0") and p == "P0 (|q|=0)" and not 1.8 <= r <= 2.2:
            sym_ok = False
    for gc in GAMMA_CS:
        d = data[(gc, "P1 (|q|=1)", EPS[1], "O1 = X2 (|q|=1)")]["delta"]
        out["plateau_O1P1_times_gamma"][str(gc)] = float(d[T >= 15].mean() * max(gc, 1e-12))
    zeno = [data[(gc, "P2 (|q|=2)", EPS[1], "O0 = Z2 Z3 (q=0)")]["delta"].max() for gc in GAMMA_CS[1:]]
    flags.update({"first_order_vanishing (ratio 3.6-4.4)": first_order_ok,
                  "symmetric_first_order (ratio 1.8-2.2)": sym_ok,
                  "zeno_monotone_O0P2 (gc 0.1->0.5->2)": bool(np.all(np.diff(zeno) < 0))})
    out["acceptance"] = flags
    return out


def plot(data):
    plt = style()
    fig, axes = plt.subplots(3, 3, figsize=(9.5, 8.2), sharex=True)
    for r, o in enumerate(OBSERVABLES):
        for c, p in enumerate(PERTURBATIONS):
            ax = axes[r, c]
            if not reachable(OBSERVABLES[o][1], p):
                ax.text(0.5, 0.5, "exactly 0\n(charge selection rule:\nthis charge is unreachable)",
                        ha="center", va="center", transform=ax.transAxes, color=MUTED, fontsize=8)
                ax.set_yticks([])
            else:
                for gc, col in zip(GAMMA_CS, ORDINAL):
                    d = data[(gc, p, EPS[1], o)]["delta"].copy()
                    d[d <= 0] = np.nan
                    ax.semilogy(T, d, color=col, label=f"$\\gamma_c$ = {gc}")
                bounds = data[(GAMMA_CS[-1], p, EPS[1], o)]["bounds"]
                name = "Thm B3" if "Thm B3" in bounds else "Thm B2" if "Thm B2" in bounds else "Thm A"
                b = bounds[name].copy()
                b[b <= 0] = np.nan
                ax.semilogy(T, b, color=ORDINAL[-1], ls="--", lw=1.2,
                            label=f"proven bound ($\\gamma_c$ = {GAMMA_CS[-1]})")
                ax.text(0.98, 0.04, f"bound: {name}", ha="right", va="bottom", transform=ax.transAxes,
                        color=MUTED, fontsize=7.5)
                if OBSERVABLES[o][1] == 0 and p.startswith("P0"):
                    ax.text(0.30, 0.72, "all $\\gamma_c$ coincide: collective noise\nacts trivially on charge-0 sector",
                            ha="left", va="center", transform=ax.transAxes, color=MUTED, fontsize=7.5)
            if r == 0:
                ax.set_title(f"imperfection {p}")
            if c == 0:
                ax.set_ylabel(f"|Δ⟨O⟩|   {o}", fontsize=8)
            if r == 2:
                ax.set_xlabel("time  t  [1/J0]")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"E1  error anatomy, N={N}, ‖δH‖ = {EPS[1]}, Néel state", y=1.04, fontsize=10)
    fig.tight_layout()
    return save_figure(fig, "e1_charge_anatomy")


if __name__ == "__main__":
    data, flags = run()
    m = metrics(data, flags)
    save_json(RESULTS / "e1_charge_anatomy.json", m)
    print(plot(data))
    for k, v in m["acceptance"].items():
        print(f"{k}: {v}")
    for k in sorted(m["ratio_2eps_over_eps"]):
        print(f"ratio {k}: {m['ratio_2eps_over_eps'][k]:.3f}   max|Δ|(0.04) = {m['max_delta_eps0.04'][k]:.2e}")
    print("plateau(O1,P1)*gamma_c:", m["plateau_O1P1_times_gamma"])
