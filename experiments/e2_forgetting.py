"""E2: Heisenberg forgetting bound (Theorem A, Corollaries A.1-A.2, Remark A.3).

Two dissipative settings for the N = 6 XY chain with a generic (all-charge) imperfection:
  (a) U(1)-conserving: local dephasing only. Observables keep a conserved part,
      so the bound grows linearly with slope ||[dH, O_inf]|| (Remark A.3).
  (b) primitive: local dephasing + emission. The observable forgets and the bound saturates.
Both are compared with the BEQS Duhamel bound 2 t ||dH|| ||O||, which grows linearly forever.
"""

import numpy as np

from common import CAT, INK, MUTED, RESULTS, save_figure, base_model, neel, ops, random_perturbation, save_json, style
from qerr.bounds import beqs_bias_bound, forgetting_bound, opnorm, osc
from qerr.lindblad import OpenModel, evolve, expect

N, EPS = 6, 0.05
T = np.linspace(0.0, 40.0, 161)
SETTINGS = {"(a) U(1)-conserving: local dephasing 0.3": dict(gamma_loc=0.3, gamma_em=0.0),
            "(b) primitive: local dephasing 0.3 + emission 0.3": dict(gamma_loc=0.3, gamma_em=0.3)}


def run():
    o2, o3 = ops(N, 2), ops(N, 3)
    O = o2["z"] @ o3["z"]
    dH = random_perturbation(N, (0, 1, 2), EPS, seed=7)
    rho0 = neel(N)
    out = {}
    for name, kw in SETTINGS.items():
        model = base_model(N, 0.0, **kw)
        pert = OpenModel(H=model.H + dH, Gamma=model.Gamma, emission=model.emission)
        delta = np.abs(expect(evolve(pert, rho0, T), O) - expect(evolve(model, rho0, T), O))
        fb = forgetting_bound(model, O, dH, T, n_fine=801)
        Os = evolve(model, O, T, heisenberg=True)
        oscs = np.array([osc(Ou) for Ou in Os])
        late_slope = (fb[-1] - fb[np.searchsorted(T, 30.0)]) / (T[-1] - 30.0)
        out[name] = dict(delta=delta, fb=fb, beqs=beqs_bias_bound(T, opnorm(dH), opnorm(O)), osc=oscs,
                         late_slope=late_slope, comm_inf=opnorm(dH @ Os[-1] - Os[-1] @ dH))
    return out


def acceptance(out):
    a, b = out.values()
    ok_order = all(np.all(r["delta"] <= r["fb"] + 1e-9) and np.all(r["fb"] <= r["beqs"] + 1e-9) for r in out.values())
    sat = (b["fb"][-1] - b["fb"][np.searchsorted(T, 30.0)]) / b["fb"][-1]
    slope_match = abs(a["late_slope"] - a["comm_inf"]) / a["comm_inf"]
    return {"observed <= ThmA <= BEQS everywhere": bool(ok_order),
            "(b) saturation: FB(40)-FB(30) < 5% of FB(40)": bool(sat < 0.05), "(b) relative late growth": float(sat),
            "(a) late slope matches ||[dH,O_inf]|| within 10%": bool(slope_match < 0.10),
            "(a) late slope": float(a["late_slope"]), "(a) ||[dH, O_inf]||": float(a["comm_inf"]),
            "BEQS/ThmA at t=40, (a)": float(a["beqs"][-1] / a["fb"][-1]),
            "BEQS/ThmA at t=40, (b)": float(b["beqs"][-1] / b["fb"][-1])}


def plot(out):
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for ax, (name, r) in zip(axes[:2], out.items()):
        positive = lambda y: np.where(y > 0, y, np.nan)
        ax.semilogy(T, positive(r["beqs"]), color=CAT[1], ls=":", lw=1.8, label="BEQS bound  2t‖δH‖‖O‖")
        ax.semilogy(T, positive(r["fb"]), color=CAT[0], ls="--", lw=1.8, label="Thm A  ∫‖[δH, O_u]‖du")
        ax.semilogy(T, positive(r["delta"]), color=INK, lw=1.4, label="observed |Δ⟨O⟩|")
        ax.set_ylim(1e-5, 10)
        ax.set_title(name, fontsize=8.5)
        ax.set_xlabel("time  t  [1/J0]")
    axes[0].set_ylabel("error in ⟨Z2 Z3⟩")
    axes[0].legend(fontsize=7.5, loc="lower right")
    for (name, r), ls in zip(out.items(), ["-", "--"]):
        axes[2].semilogy(T, np.maximum(r["osc"], 1e-12), color=INK, ls=ls, lw=1.4, label=name.split(":")[0])
    axes[2].set_title("forgetting: osc(O_u), half the spectral spread of O_u", fontsize=8.5)
    axes[2].set_xlabel("u  [1/J0]")
    axes[2].legend(fontsize=7.5, loc="lower left")
    axes[2].text(0.97, 0.80, "(a) keeps a conserved part:\nosc → const > 0", ha="right", va="top",
                 transform=axes[2].transAxes, color=MUTED, fontsize=7.5)
    fig.suptitle(f"E2  Heisenberg forgetting bound, N={N}, ‖δH‖={EPS} (all charges), γ_c = 0", fontsize=10, y=1.03)
    fig.tight_layout()
    return save_figure(fig, "e2_forgetting")


if __name__ == "__main__":
    out = run()
    acc = acceptance(out)
    save_json(RESULTS / "e2_forgetting.json", {"acceptance": acc})
    print(plot(out))
    for k, v in acc.items():
        print(f"{k}: {v}")
