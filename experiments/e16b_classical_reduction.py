"""E16b: Theorem F for interacting chains in the strong-dephasing limit (classical reduction, exact master equation).

At leading order in J/g the populations of the dephased XXZ chain of E16 (H = 1/2 sum (XX + YY + Delta ZZ), hopping
amplitude J = 1, number dephasing g) follow a classical symmetric exclusion process: a particle hops across bond b at
rate w_b = 2 J^2 g / (g^2 + E_b^2), with E_b = Delta (Z_{b-1} - Z_{b+2}) the ZZ energy cost of the hop (Z = 1 - 2n;
a missing neighbour at the open ends contributes 0). Delta = 0 is the symmetric simple exclusion process (SSEP),
Delta != 0 a symmetric Katz-Lebowitz-Spohn (KLS) process; both are reversible.
A uniform hopping miscalibration J -> (1 + eps) J multiplies every rate by lambda = (1 + eps)^2, i.e. it rescales time:
the error is exactly Delta_cl(t) = n_C(lambda t) - n_C(t), and |Delta_cl(t)| <= (lambda - 1) sup_{s in [t, lambda t]}
s |dn_C/ds|. So the error is time-uniform if and only if the parabolic estimate sup_t t |dn_C/dt| < infinity holds,
uniformly in the system size (docs/theory/phase4-theory.md, Proposition F.4).
Exact master equation on 2^N configurations (sparse generator, Krylov exponentials), g = 4, eps = 0.01, C = N/2 - 1,
initial product states as in E16 (domain wall, Neel, lowest open-chain density mode, random occupations).
Fixed before the run:
  (a) relevance of the reduction: at N = 10 the quantum error of E16 and Delta_cl agree within a factor 1.5 in
      sup_t |error| for every state at Delta = 0 and 1;
  (b) the parabolic estimate: sup_t t |dn_C/dt| grows by less than 20% from N = 10 to N = 18 for every state and
      Delta = 0, 1, 2 (no growth with the size, as the conjecture requires).
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.linalg import expm_multiply  # noqa: E402

from common import CAT, INK, MUTED, ORDINAL, RESULTS, save_figure, save_json, style  # noqa: E402

G, EPS = 4.0, 0.01
LAM = (1.0 + EPS) ** 2
SIZES, DELTAS = (10, 14, 18), (0.0, 1.0, 2.0)
STATES = ("wall", "Neel", "lowest mode", "random")
T_MAX = 2000.0


def occupations(name, n):
    x = np.arange(n)
    if name == "wall":
        return (x < n // 2).astype(float)
    if name == "Neel":
        return ((x + 1) % 2).astype(float)
    if name == "lowest mode":
        return 0.5 + 0.4 * np.cos(np.pi * (x + 0.5) / n)
    return np.random.default_rng(7).uniform(0.0, 1.0, n)  # E16: random occupation probabilities


def generator(n, delta):
    """Column generator W (dp/dt = W p) of the exclusion process; bit i of the configuration index is n_i."""
    dim = 2**n
    s = np.arange(dim)
    occ = lambda i: (s >> i) & 1 if 0 <= i < n else np.zeros(dim, dtype=np.int64)
    zval = lambda i: (1 - 2 * occ(i)) if 0 <= i < n else np.zeros(dim, dtype=np.int64)
    rows, cols, vals = [], [], []
    for b in range(n - 1):
        move = occ(b) != occ(b + 1)
        e = delta * (zval(b - 1) - zval(b + 2))
        w = 2.0 * G / (G**2 + e.astype(float) ** 2)
        src = s[move]
        rows.append(src ^ ((1 << b) | (1 << (b + 1))))
        cols.append(src)
        vals.append(w[move])
    rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    W = sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim))
    out = np.asarray(W.sum(axis=0)).ravel()
    return (W - sp.diags(out)).tocsr()


def product_measure(n, occ):
    s = np.arange(2**n)
    p = np.ones(2**n)
    for i in range(n):
        bit = (s >> i) & 1
        p *= np.where(bit == 1, occ[i], 1.0 - occ[i])
    return p


def job(args):
    n, delta, state, times = args
    W = generator(n, delta)
    c = n // 2 - 1
    nC = ((np.arange(2**n) >> c) & 1).astype(float)
    q = W.T @ nC  # d<n_C>/dt = q . p_t
    p = product_measure(n, occupations(state, n))
    vals, rates = [nC @ p], [q @ p]
    for a, b in zip(times[:-1], times[1:]):
        p = expm_multiply(W * (b - a), p)
        vals.append(nC @ p)
        rates.append(q @ p)
    return np.array(vals), np.array(rates)


def grid():
    base = np.concatenate([[0.0], np.geomspace(0.01, T_MAX, 160)])
    return np.unique(np.concatenate([base, LAM * base[1:]]))


def run():
    times = grid()
    jobs = [(n, d, st, times) for n in SIZES for d in DELTAS for st in STATES]
    e16 = np.arange(0.0, 40.0 + 1e-9, 0.125)
    jobs += [(10, d, st, e16) for d in (0.0, 1.0) for st in STATES]
    with ProcessPoolExecutor(max_workers=12) as pool:
        res = list(pool.map(job, jobs))
    return times, e16, jobs, res


def analyse(times, e16, jobs, res):
    k = len(SIZES) * len(DELTAS) * len(STATES)
    out = {"lambda": LAM}
    sup_rate, sup_err = {}, {}
    base = np.concatenate([[0.0], np.geomspace(0.01, T_MAX, 160)])
    for (n, d, st, _), (vals, rates) in zip(jobs[:k], res[:k]):
        sup_rate[(n, d, st)] = float(np.max(times * np.abs(rates)))
        nl = np.interp(LAM * base[1:], times, vals)
        sup_err[(n, d, st)] = float(np.max(np.abs(nl - np.interp(base[1:], times, vals))))
    out["sup_t t |dn_C/dt|"] = {f"Delta = {d:g}, {st}": [sup_rate[(n, d, st)] for n in SIZES] for d in DELTAS for st in STATES}
    out["sup_t |Delta_cl(t)| (classical error)"] = {
        f"Delta = {d:g}, {st}": [sup_err[(n, d, st)] for n in SIZES] for d in DELTAS for st in STATES}
    growth = {f"Delta = {d:g}, {st}": sup_rate[(SIZES[-1], d, st)] / sup_rate[(SIZES[0], d, st)]
              for d in DELTAS for st in STATES}
    out["(b) growth of sup t |dn_C/dt| from N = 10 to N = 18"] = growth
    out["(b) no growth (< 20%) for every state and Delta"] = bool(all(g < 1.2 for g in growth.values()))
    out["post hoc: P(Delta) / P(0) at the same N (the random profile near C changes with N)"] = {
        f"Delta = {d:g}, {st}": [sup_rate[(n, d, st)] / sup_rate[(n, 0.0, st)] for n in SIZES]
        for d in DELTAS[1:] for st in STATES}
    out["post hoc: largest P over all runs, against the SSEP bound A osc(n_0) / 2 = 0.264 osc(n_0)"] = [
        float(max(sup_rate.values())), 0.527 / 2]
    q = np.load(RESULTS / "e16_interacting_forgetting.npz")
    rows, ok = {}, True
    for (n, d, st, _), (vals, rates) in zip(jobs[k:], res[k:]):
        cl = np.interp(LAM * e16, e16, vals, right=np.nan)
        cl = np.where(np.isfinite(cl), cl - vals, np.nan)
        quantum = q[f"{int(d)}|{st}|error"]
        ratio = float(np.nanmax(np.abs(quantum)) / np.nanmax(np.abs(cl)))
        ok &= 1 / 1.5 <= ratio <= 1.5
        rows[f"Delta = {d:g}, {st}"] = {"sup |quantum error| (E16)": float(np.nanmax(np.abs(quantum))),
                                         "sup |Delta_cl|": float(np.nanmax(np.abs(cl))), "ratio": ratio}
    out["(a) N = 10: quantum error of E16 against the classical reduction (t <= 40 / lambda)"] = rows
    out["(a) agreement within a factor 1.5 for every state and Delta"] = bool(ok)
    return out, sup_rate


def plot(times, jobs, res, out):
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), sharey=True)
    k = len(SIZES) * len(DELTAS) * len(STATES)
    for ax, d in zip(axes, DELTAS):
        for (n, dd, st, _), (vals, rates) in zip(jobs[:k], res[:k]):
            if dd != d or st not in ("wall", "lowest mode"):
                continue
            c = ORDINAL[SIZES.index(n) + 1]
            ax.loglog(times[1:], times[1:] * np.abs(rates[1:]), color=c, ls="-" if st == "wall" else "--", lw=1.3,
                      label=f"N = {n}, {st}")
        ax.set_title(f"Delta = {d:g}" + (" (SSEP)" if d == 0 else " (symmetric KLS)"), fontsize=9)
        ax.set_xlabel("t (1/J)")
        ax.set_ylim(1e-4, 1)
    axes[0].set_ylabel("t |d<n_C>/dt|")
    axes[0].legend(fontsize=6.5)
    fig.suptitle("E16b: the parabolic estimate behind time-uniform errors, classical reduction of the dephased XXZ chain")
    fig.tight_layout()
    return save_figure(fig, "e16b_classical_reduction")


if __name__ == "__main__":
    if "--replot" not in sys.argv:
        times, e16, jobs, res = run()
        np.savez_compressed(RESULTS / "e16b_classical_reduction.npz", times=times, e16=e16,
                            vals=np.array([r[0] for r in res], dtype=object),
                            rates=np.array([r[1] for r in res], dtype=object),
                            jobs=np.array([(n, d, st) for n, d, st, _ in jobs], dtype=object))
    z = np.load(RESULTS / "e16b_classical_reduction.npz", allow_pickle=True)
    times, e16 = z["times"], z["e16"]
    jobs = [(int(n), float(d), str(st), None) for n, d, st in z["jobs"]]
    res = [(np.asarray(v, float), np.asarray(r, float)) for v, r in zip(z["vals"], z["rates"])]  # both grids hold 321 points
    out, _ = analyse(times, e16, jobs, res)
    save_json(RESULTS / "e16b_classical_reduction.json", out)
    print(plot(times, jobs, res, out))
    for key, v in out.items():
        print(f"{key}: {v}")
