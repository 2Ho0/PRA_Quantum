"""E8: three classes of imperfection relative to a weakly U(1)-symmetric reference with collective dephasing.

Reference (N = 6): the E1 model (long-range XY, weak random z fields, local sigma^z dephasing 0.05) plus collective
dephasing gamma_c, started in the Neel state (charge 0). Imperfections of strength kappa on every site:
  I    charged Hamiltonian error  dH = kappa sum_j X_j               superoperator charge Q = +-1
  II   covariant flips            jumps sqrt(kappa) sigma+_j, sigma-_j   Q = 0 (breaks only the strong symmetry)
  III  X dephasing                jumps sqrt(kappa) X_j  = II + V_{+2} + V_{-2},  V_{+-2}(rho) = kappa sum sigma^+-_j rho sigma^+-_j
Observables (charge 0): the leakage P(M != 0) out of the initial magnetization sector and the staggered
magnetization m_s = sum_j (-1)^j Z_j / N. Predictions: I is second order and Zeno-suppressed; II is first order and
exactly gamma_c-independent (all dynamics stays in B_0, where collective dephasing vanishes); III is first order
through its covariant part, and its remainder III - II is second order and Zeno-suppressed.
For uniform flip rates the magnetization distribution obeys a closed birth-death equation (V_{+-2}^dag annihilates
every diagonal operator, and the flip part of dP(M)/dt depends on the configuration only through M), so the leakage
of III and II coincide exactly; the charged remainder is therefore tested on m_s. Its order is tested at a fixed
covariant flip rate (class IIIh: flips kappa/2 plus X dephasing kappa/2, i.e. flips kappa and charged strength kappa/2):
varying kappa in III changes the flip rate too, and the resulting kappa^3 cross term competes with the Zeno-suppressed
kappa^2 term at large gamma_c.
Bound for III (Theorem B6, global form):
  |Delta(t)| <= int_0^t ||V_0^dag(O_u)|| du + 4 kappa^2 N^2 osc(O) phi(8 gamma_c, t),
with O_u the reference Heisenberg evolution, ||V_ch||_{1->1} <= 2 kappa N and nu_{+-2} <= kappa N.
"""

import sys

import numpy as np
from scipy.integrate import cumulative_trapezoid

from common import CAT, MUTED, ORDINAL, RESULTS, base_model, neel, save_figure, save_json, style
from qerr import spins
from qerr.bounds import _phi, osc
from qerr.charge import superop_charge_part, superop_charges
from qerr.lindblad import OpenModel, evolve, expect, superop

N = 6
GAMMA_LOC = 0.05
GAMMA_CS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
KAPPAS = [1e-3, 2e-3]
T = np.linspace(0.0, 20.0, 81)
T_CHECK = 10.0
CLASSES = ["I", "II", "III"]
TOL = {"rtol": 1e-12, "atol": 1e-14}


def observables():
    z = spins.zvals(N)
    ms = np.diag((z * (-1.0) ** np.arange(N)).sum(axis=1) / N).astype(complex)
    leak = np.diag((np.abs(spins.total_sz(N)) > 0.25).astype(float)).astype(complex)
    return {"leakage": leak, "m_s": ms}


def perturbed(ref: OpenModel, cls, kappa):
    op = lambda j, k: spins.pauli_op(N, j, k)
    if cls == "I":
        return OpenModel(H=ref.H + kappa * sum(op(j, "x") for j in range(N)), Gamma=ref.Gamma, emission=ref.emission)
    if cls == "II":
        jumps = [np.sqrt(kappa) * op(j, k) for j in range(N) for k in "+-"]
    elif cls == "IIIh":
        jumps = [np.sqrt(kappa / 2) * op(j, k) for j in range(N) for k in "+-"]
        jumps += [np.sqrt(kappa / 2) * op(j, "x") for j in range(N)]
    else:
        jumps = [np.sqrt(kappa) * op(j, "x") for j in range(N)]
    return OpenModel(H=ref.H, Gamma=ref.Gamma, emission=ref.emission, jumps=tuple(jumps))


def noise_superop(kappa):
    D = 2**N
    jumps = tuple(np.sqrt(kappa) * spins.pauli_op(N, j, "x") for j in range(N))
    return superop(OpenModel(H=np.zeros((D, D), complex), Gamma=np.zeros((N, N)), emission=np.zeros(N), jumps=jumps))


def b6_bound(ref, O, S0_dag, kappa, gamma_c, fine):
    """Theorem B6 (global form) for X dephasing: covariant part by Theorem A, charged part by the Zeno bound.

    S0_dag is the (column-stacking) matrix of V_0^dag, the Heisenberg dual of the covariant part of the noise.
    """
    D = 2**N
    Ou = evolve(ref, O, fine, heisenberg=True, **TOL)
    integrand = [np.linalg.norm((S0_dag @ X.reshape(-1, order="F")).reshape(D, D, order="F"), 2) for X in Ou]
    sym = cumulative_trapezoid(integrand, fine, initial=0.0)
    charged = 4 * kappa**2 * N**2 * osc(O) * np.array([_phi(8 * gamma_c, t) for t in fine])
    return sym, charged


def run():
    rho0, obs = neel(N), observables()
    s0_dag = {kappa: superop_charge_part(N, noise_superop(kappa), 0).conj().T.copy() for kappa in KAPPAS}
    out = {}
    for gc in GAMMA_CS:
        ref = base_model(N, gc, GAMMA_LOC, 0.0)
        r0 = evolve(ref, rho0, T, **TOL)
        base = {k: expect(r0, O).real for k, O in obs.items()}
        for cls in CLASSES + ["IIIh"]:
            for kappa in KAPPAS:
                rs = evolve(perturbed(ref, cls, kappa), rho0, T, **TOL)
                out[(gc, cls, kappa)] = {k: expect(rs, O).real - base[k] for k, O in obs.items()}
        fine = np.linspace(0.0, T[-1], 321)
        out[(gc, "bound")] = {}
        for k, O in obs.items():
            for kappa in KAPPAS:
                sym, ch = b6_bound(ref, O, s0_dag[kappa], kappa, gc, fine)
                out[(gc, "bound")][(k, kappa)] = (np.interp(T, fine, sym), np.interp(T, fine, ch))
    return out


def analyse(out):
    i10 = int(np.argmin(np.abs(T - T_CHECK)))
    k1, k2 = KAPPAS
    acc = {}
    S = noise_superop(1.0)
    acc["superoperator charges of D[X] (Frobenius norms)"] = {str(k): v for k, v in superop_charges(N, S).items()}
    acc["D[X]_0 equals D[sigma+] + D[sigma-] (max abs diff)"] = float(np.abs(
        superop_charge_part(N, S, 0) - superop(OpenModel(
            H=np.zeros((2**N, 2**N), complex), Gamma=np.zeros((N, N)), emission=np.zeros(N),
            jumps=tuple(spins.pauli_op(N, j, k) for j in range(N) for k in "+-")))).max())
    ratio = lambda gc, obs, cls: float(out[(gc, cls, k2)][obs][i10] / out[(gc, cls, k1)][obs][i10])
    rem = lambda gc, kappa: out[(gc, "III", kappa)]["m_s"] - out[(gc, "II", kappa)]["m_s"]
    lk = "leakage"
    acc[f"leakage: doubling ratio at t={T_CHECK:g} per gamma_c {GAMMA_CS}"] = {
        cls: [round(ratio(gc, lk, cls), 3) for gc in GAMMA_CS] for cls in CLASSES}
    acc[f"leakage: |Delta| at t={T_CHECK:g}, kappa={k1:g}, per gamma_c"] = {
        cls: [float(abs(out[(gc, cls, k1)][lk][i10])) for gc in GAMMA_CS] for cls in CLASSES}
    two = [gc for gc in GAMMA_CS if gc >= 0.5]
    acc["leakage: class I second order (ratio in [3.5, 4.5], gamma_c >= 0.5)"] = bool(all(
        3.5 <= ratio(gc, lk, "I") <= 4.5 for gc in two))
    acc["leakage: class I decreases with gamma_c (gamma_c >= 0.5)"] = bool(np.all(np.diff(
        [abs(out[(gc, "I", k1)][lk][i10]) for gc in two]) < 0))
    acc["leakage: classes II and III first order (ratio in [1.8, 2.2])"] = bool(all(
        1.8 <= ratio(gc, lk, c) <= 2.2 for gc in GAMMA_CS for c in ("II", "III")))
    d2 = np.array([out[(gc, "II", k1)][lk] for gc in GAMMA_CS])
    spread = float(np.abs(d2 - d2[0]).max() / np.abs(d2).max())
    acc["leakage: class II gamma_c-independent (max rel. spread over gamma_c < 1e-9)"] = [spread, bool(spread < 1e-9)]
    same = max(float(np.abs(out[(gc, "III", kappa)][lk] - out[(gc, "II", kappa)][lk]).max())
               for gc in GAMMA_CS for kappa in KAPPAS)
    acc["leakage: III and II identical (closed birth-death equation for P(M)), max |diff| < 1e-12"] = [
        same, bool(same < 1e-12)]
    acc[f"m_s: doubling ratio at t={T_CHECK:g} per gamma_c {GAMMA_CS}"] = {
        cls: [round(ratio(gc, "m_s", cls), 3) for gc in GAMMA_CS] for cls in ("II", "III")}
    r_rem = [float(rem(gc, k2)[i10] / rem(gc, k1)[i10]) for gc in GAMMA_CS]
    acc[f"m_s: remainder III-II ratio when flips and charged part are doubled together (informational)"] = [
        round(r, 3) for r in r_rem]
    fixed = {(gc, kappa): float(rem(gc, kappa)[i10] / (out[(gc, "IIIh", kappa)]["m_s"][i10]
                                                       - out[(gc, "II", kappa)]["m_s"][i10]))
             for gc in GAMMA_CS for kappa in KAPPAS}
    acc[f"m_s: charged remainder ratio R(k)/R(k/2) at fixed flip rate, per gamma_c, kappa={k2:g}"] = [
        round(fixed[(gc, k2)], 4) for gc in GAMMA_CS]
    acc["m_s: charged remainder second order in its own strength (ratio in [3.9, 4.1], all gamma_c, kappa)"] = bool(
        all(3.9 <= r <= 4.1 for r in fixed.values()))
    mag = [float(abs(rem(gc, k1)[i10])) for gc in GAMMA_CS]
    acc[f"m_s: |remainder III-II| at t={T_CHECK:g}, kappa={k1:g}, per gamma_c"] = mag
    acc["m_s: remainder decreases with gamma_c (gamma_c >= 0.25)"] = bool(np.all(np.diff(mag[1:]) < 0))
    d2 = np.array([out[(gc, "II", k1)]["m_s"] for gc in GAMMA_CS])
    spread = float(np.abs(d2 - d2[0]).max() / np.abs(d2).max())
    acc["m_s: class II gamma_c-independent (max rel. spread over gamma_c < 1e-9)"] = [spread, bool(spread < 1e-9)]
    for obs in ("leakage", "m_s"):
        ok = True
        for gc in GAMMA_CS:
            for kappa in KAPPAS:
                sym, ch = out[(gc, "bound")][(obs, kappa)]
                ok &= bool(np.all(sym + ch >= np.abs(out[(gc, "III", kappa)][obs]) - 1e-12))
        acc[f"{obs}: Theorem B6 bound >= |Delta_III| at all times"] = ok
    return acc


def plot(out):
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5))
    k1, k2 = KAPPAS
    gc = 1.0
    i10 = int(np.argmin(np.abs(T - T_CHECK)))
    names = {"I": "I  charged Hamiltonian (κΣXⱼ)", "II": "II  covariant flips (σ±)", "III": "III  X dephasing"}
    ax = axes[0]
    for cls, col, ls in zip(CLASSES, CAT, ("-", "-", (0, (4, 3)))):
        ax.semilogy(T[1:], np.abs(out[(gc, cls, k2)]["leakage"][1:]), color=col, ls=ls, label=names[cls])
    sym, ch = out[(gc, "bound")][("leakage", k2)]
    ax.semilogy(T[1:], (sym + ch)[1:], color=MUTED, ls="--", lw=1.0, label="Theorem B6 bound for III")
    ax.text(0.97, 0.62, "II and III coincide exactly", transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)
    ax.set_xlabel("time  t  [1/J₀]")
    ax.set_title(f"leakage P(M ≠ 0), κ = {k2:g}, γ_c = {gc:g}", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower right")
    ax = axes[1]
    pos = [g for g in GAMMA_CS if g > 0]
    for cls, col, mk in zip(CLASSES, CAT, "os^"):
        ax.loglog(pos, [abs(out[(g, cls, k2)]["leakage"][i10]) for g in pos], marker=mk, ms=4, color=col,
                  label=f"{names[cls].split('  ')[0]}: leakage")
    ax.loglog(pos, [abs(out[(g, "III", k2)]["m_s"][i10] - out[(g, "II", k2)]["m_s"][i10]) for g in pos],
              marker="^", ms=4, color=CAT[2], ls=":", mfc="white", label="III − II: staggered magnetization")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{g:g}" for g in pos])
    ax.minorticks_off()
    ax.set_xlabel("collective dephasing  γ_c")
    ax.set_title(f"t = {T_CHECK:g}: only the charged parts feel γ_c", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 0.86), ncol=2)
    ax = axes[2]
    series = [("I (leakage)", [out[(g, "I", k2)]["leakage"][i10] / out[(g, "I", k1)]["leakage"][i10] for g in GAMMA_CS],
               CAT[0], "o", None),
              ("II (leakage)", [out[(g, "II", k2)]["leakage"][i10] / out[(g, "II", k1)]["leakage"][i10]
                                for g in GAMMA_CS], CAT[1], "s", None),
              ("III (leakage)", [out[(g, "III", k2)]["leakage"][i10] / out[(g, "III", k1)]["leakage"][i10]
                                 for g in GAMMA_CS], CAT[2], "^", None),
              ("charged part of III, fixed flips (m_s)", [(out[(g, "III", k2)]["m_s"][i10] - out[(g, "II", k2)]["m_s"][i10])
                                             / (out[(g, "IIIh", k2)]["m_s"][i10] - out[(g, "II", k2)]["m_s"][i10])
                                             for g in GAMMA_CS], CAT[2], "^", "white")]
    for u, (label, r, col, mk, mfc) in enumerate(series):
        ax.plot(np.arange(len(r)) + (u - 1.5) * 0.12, r, mk, ms=4.5, color=col, mfc=mfc or col, lw=0, label=label)
    for level in (2, 4):
        ax.axhline(level, color=MUTED, lw=0.7, ls="--")
    ax.set_xticks(range(len(GAMMA_CS)))
    ax.set_xticklabels([f"{g:g}" for g in GAMMA_CS])
    ax.set_ylim(1.5, 4.5)
    ax.set_xlabel("γ_c")
    ax.set_ylabel("Δ(2κ) / Δ(κ)")
    ax.set_title("order of the response: 2 = first, 4 = second", fontsize=9)
    ax.legend(fontsize=6.5, ncol=2, loc="center right")
    fig.suptitle("E8  weak vs strong U(1) symmetry: three classes of imperfection (N = 6)", fontsize=10, y=1.03)
    fig.tight_layout()
    return save_figure(fig, "e8_symmetry_classes")


def save_cache(out, path):
    flat = {}
    for key, val in out.items():
        if key[1] == "bound":
            for (obs, kappa), (sym, ch) in val.items():
                flat[f"{key[0]}|bound|{obs}|{kappa}|sym"], flat[f"{key[0]}|bound|{obs}|{kappa}|ch"] = sym, ch
        else:
            for obs, arr in val.items():
                flat[f"{key[0]}|{key[1]}|{key[2]}|{obs}"] = arr
    np.savez_compressed(path, **flat)


def load_cache(path):
    out = {}
    with np.load(path) as z:
        for name in z.files:
            parts = name.split("|")
            gc = float(parts[0])
            if parts[1] == "bound":
                obs, kappa, part = parts[2], float(parts[3]), parts[4]
                pair = out.setdefault((gc, "bound"), {}).setdefault((obs, kappa), [None, None])
                pair[0 if part == "sym" else 1] = z[name]
            else:
                out.setdefault((gc, parts[1], float(parts[2])), {})[parts[3]] = z[name]
    return out


if __name__ == "__main__":
    cache = RESULTS / "e8_symmetry_classes.npz"
    if "--replot" not in sys.argv:  # --replot: analyse and plot the cached trajectories only
        save_cache(run(), cache)
    out = load_cache(cache)
    acc = analyse(out)
    save_json(RESULTS / "e8_symmetry_classes.json", {"acceptance": acc})
    print(plot(out))
    for key, v in acc.items():
        print(f"{key}: {v}")
