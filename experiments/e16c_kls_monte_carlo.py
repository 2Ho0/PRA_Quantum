"""E16c: the parabolic estimate of Proposition F.4 for the symmetric KLS process at N = 32-128 (Monte Carlo).

E16b computed P = sup_t t |dn_C/dt| exactly for N <= 18. Its size test (b) failed for the random profile, but that
profile and the observed site C = N/2 - 1 both changed with N, so the test mixed size with the local environment.
Here the random occupations are centred on C (the same values around C for every N), and the chains reach N = 128.
Process as E16b: open chain, hop across bond b at rate w_b = 2 G / (G^2 + E_b^2), G = 4 (J = 1),
E_b = Delta (Z_{b-1} - Z_{b+2}), a missing neighbour contributes 0; Delta = 0 is SSEP, Delta = 1, 2 symmetric KLS.
Estimator: exact continuous-time Monte Carlo (uniformisation at the maximal rate 2/G, one clock per run, R runs per
case). t dn_C/dt is estimated in two unbiased ways at every grid time and the one with the smaller standard error is
used: from the generator, t E[(L n_C)(eta_t)] with (L n_C)(eta) = w_{C-1}(eta_{C-1} - eta_C) + w_C(eta_{C+1} - eta_C)
(precise at small t, its error grows like t), and the log-time difference of n_C over the two neighbouring grid
points (error roughly constant in log t). P is the largest estimate on a log grid 0.02 <= t <= 1000; at N = 18 the
exact master equation gives the reference.
States: domain wall, Neel (as E16b) and random product occupations centred on C (rng 7, 256 values).
Fixed before the run (SE is the Monte Carlo standard error of t dn_C/dt at the maximising time):
  (a) validation: at N = 18, |P_MC - P_exact| <= 3 SE + 0.002 for every state and Delta;
  (b) no growth with the size: |P(128) - P(32)| <= 3 sqrt(SE_128^2 + SE_32^2) + 0.005 for every state and Delta;
  (c) the SSEP constant: P <= 0.264 osc(n_0) + 3 SE for every N, state and Delta (Proposition F.4, A/2 = 0.264);
  (d) no late regrowth: at N = 128, max over 100 <= t <= 1000 of t |dn_C/dt| <= P/2 + 3 SE.
The conjecture of Proposition F.4 survives this test if (b)-(d) hold for Delta = 1 and 2.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
from scipy.sparse.linalg import expm_multiply  # noqa: E402

from common import CAT, INK, MUTED, ORDINAL, RESULTS, save_figure, save_json, style  # noqa: E402
from e16b_classical_reduction import G, generator, product_measure  # noqa: E402

DELTAS = (0.0, 1.0, 2.0)
SIZES = (18, 32, 64, 128)
STATES = ("wall", "Neel", "random (centred)")
R_RUNS = 100000
GRID = np.geomspace(0.02, 1000.0, 57)
A_HALF = 0.264  # Proposition F.4: P <= (A/2) osc(n_0) for SSEP, A = 0.527
W_MAX = 2.0 / G
_RANDOM = np.random.default_rng(7).uniform(0.0, 1.0, 256)


def site(n):
    return n // 2 - 1


def occupations(name, n):
    x = np.arange(n)
    if name == "wall":
        return (x < n // 2).astype(float)
    if name == "Neel":
        return ((x + 1) % 2).astype(float)
    return _RANDOM[x - site(n) + 64]


def rates(z, b, delta):
    """Hop rate across bond (b, b+1) for the padded spin array z (real sites 1..n, padding 0)."""
    rows = np.arange(z.shape[0])
    e = delta * (z[rows, b - 1] - z[rows, b + 2])
    return 2.0 * G / (G**2 + e.astype(float) ** 2)


def l_nc(z, c, delta):
    """(L n_C)(eta) per run; z padded, the observed site sits at column c + 1."""
    n_ = (1 - z) // 2
    k = c + 1
    rows = np.arange(z.shape[0])
    left = rates(z, np.full(z.shape[0], k - 1), delta) * (n_[rows, k - 1] - n_[rows, k])
    right = rates(z, np.full(z.shape[0], k), delta) * (n_[rows, k + 1] - n_[rows, k])
    return left + right


def mc_job(args):
    """Monte Carlo n_C(t) on GRID and the estimate of t dn_C/dt with its standard error."""
    n, delta, state, seed = args
    t0 = time.time()
    rng = np.random.default_rng(seed)
    R, c = R_RUNS, site(n)
    occ = occupations(state, n)
    z = np.zeros((R, n + 2), np.int8)
    z[:, 1:n + 1] = np.where(rng.random((R, n)) < occ, -1, 1)  # Z = 1 - 2 n
    rows = np.arange(R)
    tau = np.zeros(R)
    nxt = np.zeros(R, np.int64)
    rec_n = np.zeros((len(GRID), R), np.int8)
    rec_g = np.zeros((len(GRID), R), np.float32)
    lam = (n - 1) * W_MAX
    done = np.zeros(R, bool)
    while not done.all():
        tau += rng.exponential(1.0 / lam, R)
        while True:
            m = (~done) & (tau > GRID[np.minimum(nxt, len(GRID) - 1)])
            if not m.any():
                break
            idx = np.nonzero(m)[0]
            rec_n[nxt[idx], idx] = (1 - z[idx, c + 1]) // 2
            rec_g[nxt[idx], idx] = l_nc(z[idx], c, delta)
            nxt[idx] += 1
            done |= nxt >= len(GRID)
        b = rng.integers(1, n, R)  # bond (b, b+1) in padded columns, b = 1..n-1
        move = z[rows, b] != z[rows, b + 1]
        acc = move & (rng.random(R) * W_MAX < rates(z, b, delta)) & ~done
        i = np.nonzero(acc)[0]
        zb = z[i, b[i]].copy()
        z[i, b[i]] = z[i, b[i] + 1]
        z[i, b[i] + 1] = zb
    inst = GRID[:, None] * rec_g.astype(float)
    est, se = inst.mean(axis=1), inst.std(axis=1, ddof=1) / np.sqrt(R)
    lg = np.log(GRID)
    for k in range(1, len(GRID)):  # log-time difference over the neighbouring grid points (backward at the end)
        hi = min(k + 1, len(GRID) - 1)
        diff = (rec_n[hi].astype(float) - rec_n[k - 1]) / (lg[hi] - lg[k - 1])
        s_d = diff.std(ddof=1) / np.sqrt(R)
        if s_d < se[k]:
            est[k], se[k] = diff.mean(), s_d
    print(f"MC N={n} Delta={delta:g} {state}: {time.time() - t0:.0f} s", flush=True)
    return rec_n.mean(axis=1), est / GRID, se / GRID  # dn_C/dt and its standard error


def exact_job(args):
    """Exact n_C(t) and dn_C/dt on GRID from the master equation (N = 18)."""
    n, delta, state = args
    t0 = time.time()
    W = generator(n, delta)
    p = product_measure(n, occupations(state, n))
    s = np.arange(2**n)
    nc = ((s >> site(n)) & 1).astype(float)
    f = W.T @ nc  # (L n_C) as a function on configurations
    out_n, out_g, t_prev = [], [], 0.0
    for t in GRID:
        p = expm_multiply(W * (t - t_prev), p)
        t_prev = t
        out_n.append(float(nc @ p))
        out_g.append(float(f @ p))
    print(f"exact N={n} Delta={delta:g} {state}: {time.time() - t0:.0f} s", flush=True)
    return np.array(out_n), np.array(out_g)


def summary(g, se=None):
    tg = GRID * np.abs(g)
    k = int(np.argmax(tg))
    late = GRID >= 100.0
    return {"P": float(tg[k]), "t at max": float(GRID[k]), "SE": float(GRID[k] * se[k]) if se is not None else 0.0,
            "late max (100 <= t <= 1000)": float(tg[late].max()),
            "late SE": float((GRID * se)[late][np.argmax(tg[late])]) if se is not None else 0.0}


def run():
    cases = [(n, d, s) for n in SIZES for d in DELTAS for s in STATES]
    jobs = [(n, d, s, 1000 * i + 11) for i, (n, d, s) in enumerate(cases)]
    with ProcessPoolExecutor(max_workers=6) as pool:
        mc = dict(zip(cases, pool.map(mc_job, jobs)))
        ex = dict(zip([(d, s) for d in DELTAS for s in STATES],
                      pool.map(exact_job, [(18, d, s) for d in DELTAS for s in STATES])))
    return mc, ex


def analyse(mc, ex):
    out = {"settings": {"G": G, "runs per case": R_RUNS, "grid": [float(GRID[0]), float(GRID[-1]), len(GRID)],
                        "observed site": "C = N/2 - 1", "SSEP constant A/2": A_HALF}}
    rows, checks = {}, {"(a) validation at N = 18": True, "(b) no growth from N = 32 to 128": True,
                        "(c) SSEP constant": True, "(d) no late regrowth at N = 128": True}
    kls_ok = True
    for d in DELTAS:
        for s in STATES:
            key = f"Delta = {d:g}, {s}"
            osc = float(np.ptp(occupations(s, 128)))
            per = {n: summary(mc[(n, d, s)][1], mc[(n, d, s)][2]) for n in SIZES}
            exact = summary(ex[(d, s)][1])
            a = abs(per[18]["P"] - exact["P"]) <= 3 * per[18]["SE"] + 0.002
            b = abs(per[128]["P"] - per[32]["P"]) <= 3 * np.hypot(per[128]["SE"], per[32]["SE"]) + 0.005
            c = all(per[n]["P"] <= A_HALF * osc + 3 * per[n]["SE"] for n in SIZES)
            dd = per[128]["late max (100 <= t <= 1000)"] <= per[128]["P"] / 2 + 3 * per[128]["late SE"]
            rows[key] = {"P exact, N = 18": exact["P"], **{f"P MC, N = {n}": [per[n]["P"], per[n]["SE"]] for n in SIZES},
                         "t at max, N = 128": per[128]["t at max"], "late max, N = 128": per[128]["late max (100 <= t <= 1000)"],
                         "bound 0.264 osc": A_HALF * osc, "(a)": bool(a), "(b)": bool(b), "(c)": bool(c), "(d)": bool(dd)}
            for name, ok in zip(checks, (a, b, c, dd)):
                checks[name] = bool(checks[name] and ok)
            if d > 0:
                kls_ok = kls_ok and b and c and dd
    out["cases"] = rows
    out["verdict"] = dict(checks, **{"conjecture survives (b)-(d) for Delta = 1, 2": bool(kls_ok)})
    ratios = {}
    for s in STATES:
        for n in SIZES:
            p0 = summary(mc[(n, 0.0, s)][1], mc[(n, 0.0, s)][2])["P"]
            ratios[f"{s}, N = {n}"] = [summary(mc[(n, d, s)][1], mc[(n, d, s)][2])["P"] / p0 for d in DELTAS[1:]]
    out["P(Delta) / P(0) for Delta = 1, 2"] = ratios
    return out


def plot(mc, ex):
    plt = style()
    fig, axes = plt.subplots(len(DELTAS), len(STATES), figsize=(12.0, 7.2), sharex=True, sharey=True)
    for i, d in enumerate(DELTAS):
        for k, s in enumerate(STATES):
            ax = axes[i, k]
            for n, c in zip(SIZES, ORDINAL):
                ax.plot(GRID, GRID * np.abs(mc[(n, d, s)][1]), "-", color=c, lw=1.1, label=f"N = {n} (MC)")
            ax.plot(GRID, GRID * np.abs(ex[(d, s)][1]), ":", color=INK, lw=1.2, label="exact, N = 18")
            ax.axhline(A_HALF * float(np.ptp(occupations(s, 128))), color=CAT[1], lw=0.9, ls="--",
                       label="SSEP bound 0.264 osc($n_0$)")
            ax.set_xscale("log")
            if i == 0:
                ax.set_title(s, fontsize=9)
            if i == len(DELTAS) - 1:
                ax.set_xlabel("t (units of 1/J)")
            if k == 0:
                ax.set_ylabel(f"$t\\,|dn_C/dt|$, $\\Delta$ = {d:g}")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("E16c: the parabolic estimate for SSEP ($\\Delta$ = 0) and symmetric KLS up to N = 128 (Monte Carlo)")
    fig.tight_layout()
    return save_figure(fig, "e16c_kls_monte_carlo")


if __name__ == "__main__":
    if "--replot" not in sys.argv:
        mc, ex = run()
        np.savez_compressed(RESULTS / "e16c_kls_monte_carlo.npz", grid=GRID,
                            **{f"mc_{n}_{d:g}_{s.split()[0]}": np.stack(v) for (n, d, s), v in mc.items()},
                            **{f"ex_{d:g}_{s.split()[0]}": np.stack(v) for (d, s), v in ex.items()})
    zz = np.load(RESULTS / "e16c_kls_monte_carlo.npz")
    mc = {(n, d, s): tuple(zz[f"mc_{n}_{d:g}_{s.split()[0]}"]) for n in SIZES for d in DELTAS for s in STATES}
    ex = {(d, s): tuple(zz[f"ex_{d:g}_{s.split()[0]}"]) for d in DELTAS for s in STATES}
    out = analyse(mc, ex)
    save_json(RESULTS / "e16c_kls_monte_carlo.json", out)
    print(plot(mc, ex))
    for key, v in out.items():
        print(f"{key}: {v}")
