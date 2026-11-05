"""E13b: a local form of Corollary F-qf (pointwise pairing) and how much it tightens E13.

Corollary F-qf bounds the integrand by global factors, a(u) max_b |grad n'_b(s)| + m(u) max_{x!=y} |C'_xy(s)|, and is
therefore blind to the distance between the observable and the region where the state has gradients (domain walls
were 14x loose). Hoelder pairing entry by entry keeps that information and is still rigorous:
    |<i[dH, O_u]>_{rho'_s}| <= sum_b |e_b| |w_{x+1,x}(u) - w_{x,x+1}(u)| |n'_x(s) - n'_{x+1}(s)|
                              + sum_{x != y} |M_xy(u)| |C'_xy(s)|,
    F_loc(t) = int_0^t P(u, t - u) du.
P is computed on the full (u, s) grid of E13 by two matrix products per state, and F_loc by interpolation on that grid.
Model and states as E13 (results/e13_two_sided_bound.npz for the true errors and the global bound).
The pointwise inequality is exact; its numerical evaluation is not. Bilinear interpolation on the first grid cell and
float32 storage (values below 1e-38 underflow) can put F_loc a hair below |error|; the diagnostic reports by how much.
"""

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from common import CAT, INK, MUTED, RESULTS, save_figure, save_json, style
from e13_two_sided_bound import C, EPS, G_DEPH, N, STATES, TOL, U, convolve, hopping, occupations
from qerr.hydro import commutator_coefficients, evolve_two_point


def observable_side():
    w0 = np.zeros((N, N), complex)
    w0[C, C] = 1.0
    W = evolve_two_point(-np.ones(N - 1), G_DEPH * np.ones(N), w0, U, **TOL)
    dh = hopping(EPS)
    A = np.array([EPS * np.abs(np.diag(w, -1) - np.diag(w, 1)) for w in W], np.float32)          # (T, N-1)
    off = ~np.eye(N, dtype=bool)
    Mabs = np.array([np.abs(commutator_coefficients(dh, w))[off] for w in W], np.float32)         # (T, N(N-1))
    return A, Mabs


def state_job(args):
    name, A, Mabs = args
    Cs = evolve_two_point((-1.0 + EPS) * np.ones(N - 1), G_DEPH * np.ones(N), np.diag(occupations(name)).astype(complex),
                          U, **TOL)
    off = ~np.eye(N, dtype=bool)
    grad = np.array([np.abs(np.diff(np.real(np.diag(c)))) for c in Cs], np.float32)              # (T, N-1)
    coh = np.array([np.abs(c)[off] for c in Cs], np.float32)                                      # (T, N(N-1))
    return (A @ grad.T + Mabs @ coh.T).astype(float)                                              # P[u, s]


def f_local(P):
    interp = RegularGridInterpolator((U, U), P, bounds_error=False, fill_value=None)
    out = [0.0]
    for t in U[1:]:
        th = np.linspace(0.0, np.pi, 600)
        u = 0.5 * t * (1.0 - np.cos(th))
        vals = interp(np.column_stack([u, t - u])) * 0.5 * t * np.sin(th)
        out.append(float(np.trapezoid(vals, th)))
    return np.array(out)


def run():
    A, Mabs = observable_side()
    with ProcessPoolExecutor(max_workers=3) as pool:
        P = dict(zip(STATES, pool.map(state_job, [(s, A, Mabs) for s in STATES])))
    return P


def analyse(P):
    z = np.load(RESULTS / "e13_two_sided_bound.npz")
    obs = z["obs"]
    out, curves = {}, {}
    a, m = obs[:, 0], obs[:, 1]
    for s in STATES:
        err = np.abs(z[f"state_{s}_1"][:, 0] - z[f"state_{s}_0"][:, 0])
        gamma, kappa = z[f"state_{s}_1"][:, 1], z[f"state_{s}_1"][:, 2]
        F_glob = np.array([0.0] + [convolve(a, gamma, t) + convolve(m, kappa, t) for t in U[1:]])
        F_loc = f_local(P[s])
        curves[s] = (err, F_glob, F_loc)
        out[s] = {"F_loc >= |error| at all t": bool(np.all(F_loc >= err - 1e-10)),
                  "max (|error| - F_loc) / peak error": float(np.max(err - F_loc) / err.max()),
                  "F_loc <= F_global at all t": bool(np.all(F_loc <= F_glob * (1 + 1e-6) + 1e-12)),
                  "sup F_loc / peak error": float(F_loc.max() / err.max()),
                  "sup F_global / peak error": float(F_glob.max() / err.max()),
                  "F_loc(T) / F_loc(T/10)": float(F_loc[-1] / F_loc[int(np.argmin(np.abs(U - 0.1 * U[-1])))])}
    out["acceptance"] = {
        "local bound holds for every state": all(out[s]["F_loc >= |error| at all t"] for s in STATES),
        "local bound never exceeds the global one": all(out[s]["F_loc <= F_global at all t"] for s in STATES),
        "domain walls within 10x of the peak error": all(out[s]["sup F_loc / peak error"] < 10 for s in ("wall 10", "wall 40")),
    }
    return out, curves


def plot(curves, out):
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    pos = U > 0
    ax = axes[0]
    for s, c in (("wall 10", CAT[0]), ("wall 40", CAT[1])):
        err, Fg, Fl = curves[s]
        ax.loglog(U[pos], Fg[pos], color=c, lw=1, ls=":", label=f"global F, {s}")
        ax.loglog(U[pos], Fl[pos], color=c, lw=1.8, label=f"local F, {s}")
        ax.loglog(U[pos], np.where(err[pos] > 0, err[pos], np.nan), color=c, lw=1, ls="--", label=f"true |error|, {s}")
    ax.set_ylim(1e-6, 0.1)
    ax.set_xlabel("t [1/J]")
    ax.set_ylabel("error of $n_C$")
    ax.set_title("(a) domain walls: the local pairing sees the distance")
    ax.legend(fontsize=6.5, loc="lower right")
    ax = axes[1]
    yy = np.arange(len(STATES))
    ax.plot([out[s]["sup F_global / peak error"] for s in STATES], yy, "D", mfc="white", mec=MUTED, label="global (E13)")
    ax.plot([out[s]["sup F_loc / peak error"] for s in STATES], yy, "D", color=CAT[0], label="local")
    ax.axvline(1.0, color=INK, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yticks(yy, STATES)
    ax.invert_yaxis()
    ax.set_xlabel("sup$_t$ bound / peak true error")
    ax.set_title("(b) tightness")
    ax.legend(fontsize=7)
    fig.suptitle("E13b: pointwise pairing in Corollary F-qf")
    fig.tight_layout()
    return save_figure(fig, "e13b_local_pairing")


if __name__ == "__main__":
    if "--replot" not in sys.argv:
        P = run()
        np.savez_compressed(RESULTS / "e13b_local_pairing.npz", **{f"P_{s}": v for s, v in P.items()})
    zz = np.load(RESULTS / "e13b_local_pairing.npz")
    P = {s: zz[f"P_{s}"] for s in STATES}
    out, curves = analyse(P)
    save_json(RESULTS / "e13b_local_pairing.json", out)
    print(plot(curves, out))
    for k, v in out.items():
        print(f"{k}: {v}")
