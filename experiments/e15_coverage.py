"""E15: do the certified and the statistical bands cover the measured BEQS data out of sample?

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0).
Reference models from E11, learned on t <= 1.5 ms and predicted for 1.5 < t <= 10 ms: X15 (U(1)-symmetric couplings,
all single-site fields, collective dephasing, fixed D5/2 emission) and Xb15 (X15 with the collective rate
gamma_c + beta t, the memory model).
(a) Statistical band. N_REP bootstrap refits of E11 (the 200 settings resampled) per model are simulated exactly; the
    band is 1.96 sqrt(sd_boot^2 + SEM^2) about the point prediction, and the calibration factor of a class is
    rms(data - prediction) / rms(sqrt(sd_boot^2 + SEM^2)), which is 1 for a calibrated model.
(b) Certified band (X15 only; Theorem A needs a time-independent reference). For replicate k the perturbation is
    V_k = L_k - L_ref, and Theorem A in its general-V form bounds the change of every expectation value, for every
    initial state, by b_k(t) = int_0^t ||V_k^dag(O_u)|| du with O_u = e^{u L_ref^dag} O. The bootstrap spread of
    theta_k - theta_hat stands in for that of theta_hat - theta_true and b_k is even in the deviation, so the band is
    the 95% bootstrap quantile B(t). A point is covered if |data - prediction| <= B(t) + 1.96 SEM, and the band is
    vacuous once B(t) >= osc(O). Corollary A.1 (the BEQS form 2 t ||dH_k|| osc(O)) is given for orientation.
    Observables, one per charge class: mean Z, mean X, mean Y and, for the central pair (4, 5), the flip-flop
    (XX + YY)/2, ZZ and the charge-2 part (XX - YY)/2. Norms by Lanczos (ARPACK, tol 1e-9), integrals by the
    trapezoid rule on a 0.25 ms grid (the change from the 0.5 ms grid is reported).
Predictions fixed before the run: the certified band covers >= 90% of the out-of-sample points of every observable,
and the memory model has the smaller calibration factor in every charged class (collective noise acts only there).
Added after the run had started (not part of the predictions): the statistical band of the six observables of (b),
1.96 sd_boot from the X15 refits of (a), so that the widths of the certified and the probabilistic band can be compared
on the same observable.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker; the workers run in parallel

import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from common import CAT, INK, MUTED, RESULTS, save_figure, save_json, style
from e6_ion_learning import N, ansatz, observables, physical
from e6c_validation import charge_classes, combine, combine_sem
from e11_model_selection import validate_job
from qerr.learning import refine
from qerr.lindblad import evolve
from qerr.pauli import psparse

T_LEARN, T_LATE, Z95 = 1.5, 5.0, 1.96
N_BOOT, N_REP, WORKERS = 30, 12, 16
MODELS = ("X15", "Xb15")
PARTS = RESULTS / "e15_parts"  # one file per finished job, so an interrupted run resumes
C = N // 2 - 1  # the central pair (4, 5)
OBS = {
    "mean Z": [(((i, "Z"),), 1.0 / N) for i in range(N)],
    "mean X": [(((i, "X"),), 1.0 / N) for i in range(N)],
    "mean Y": [(((i, "Y"),), 1.0 / N) for i in range(N)],
    "flip-flop (4,5)": [(((C, "X"), (C + 1, "X")), 0.5), (((C, "Y"), (C + 1, "Y")), 0.5)],
    "ZZ (4,5)": [(((C, "Z"), (C + 1, "Z")), 1.0)],
    "charge-2 pair (4,5)": [(((C, "X"), (C + 1, "X")), 0.5), (((C, "Y"), (C + 1, "Y")), -0.5)],
}


def spectral_norm(M, v0=None):
    """||M|| of a Hermitian matrix (dense or sparse) and the top eigenvector, for warm starts."""
    if not (M.nnz if hasattr(M, "nnz") else np.any(M)):
        return 0.0, v0
    try:
        w, v = eigsh(M, k=1, which="LM", tol=1e-9, ncv=24, v0=v0, maxiter=300)
        return float(abs(w[0])), v[:, 0]
    except ArpackNoConvergence:
        return float(np.abs(np.linalg.eigvalsh(M.toarray() if hasattr(M, "toarray") else M)).max()), v0


def certified_job(name):
    ans = ansatz()
    z = np.load(RESULTS / "e11_model_selection.npz")
    ref = physical(z["theta_X15"], ans)
    O = sum(c * psparse(p, N) for p, c in OBS[name]).toarray().astype(complex)
    osc = float(np.ptp(np.linalg.eigvalsh(O)))
    U = refine(z["times"], 2)
    Ou = evolve(ref, O, U, heisenberg=True, rtol=1e-8, atol=1e-10)
    W0 = ref._elementwise
    integrand, hnorm = np.zeros((N_BOOT, len(U))), np.zeros(N_BOOT)
    for k, th in enumerate(z["boot_theta_X15"][:N_BOOT]):
        alt = physical(th, ans)
        dH, dW = (alt.H - ref.H).tocsr(), alt._elementwise - W0  # emission is fixed, so it cancels
        hnorm[k] = spectral_norm(dH)[0]
        v0 = None
        for m, A in enumerate(Ou):
            A = 0.5 * (A + A.conj().T)
            integrand[k, m], v0 = spectral_norm(1j * (dH @ A - A @ dH) + dW * A, v0)
    return osc, integrand, hnorm


def band_job(variant, k, s):
    z = np.load(RESULTS / "e11_model_selection.npz")
    return validate_job((variant, z[f"boot_theta_{variant}"][k], float(z[f"boot_beta_{variant}"][k]), s))


def job(spec):
    path = PARTS / (re.sub(r"[^A-Za-z0-9.-]+", "_", "_".join(map(str, spec))) + ".npz")
    if path.exists():
        z = np.load(path)
        return (float(z["osc"]), z["integrand"], z["hnorm"]) if spec[0] == "certified" else z["pred"]
    t0 = time.time()
    if spec[0] == "certified":
        res = certified_job(spec[1])
        np.savez_compressed(path, osc=res[0], integrand=res[1], hnorm=res[2])
    else:
        res = band_job(*spec[1:])
        np.savez_compressed(path, pred=res)
    print(f"{path.stem} done in {time.time() - t0:.0f} s", flush=True)
    return res


def run():
    specs = [("certified", n) for n in OBS] + [("band", v, k, s) for v in MODELS for k in range(N_REP) for s in (0, 1)]
    PARTS.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        res = list(pool.map(job, specs))
    cert = dict(zip(OBS, res[:len(OBS)]))
    bands = {v: np.array(res[len(OBS) + 2 * N_REP * i:len(OBS) + 2 * N_REP * (i + 1)]).reshape(N_REP, 2, *res[-1].shape)
             for i, v in enumerate(MODELS)}
    return cert, bands


def analyse_certified(cert, z, boot):
    times, data, sem, pred = z["times"], z["data"], z["sem"], z["pred_X15"]
    U = refine(times, 2)
    k = {p: m for m, p in enumerate(observables())}
    oos = times > T_LEARN + 1e-9
    out, curves = {}, {}
    for name, (osc, integ, hnorm) in cert.items():
        b = cumulative_trapezoid(integ, U, axis=1, initial=0.0)[:, ::2]  # U[2j] = times[j]
        b_coarse = cumulative_trapezoid(integ[:, ::2], times, axis=1, initial=0.0)
        B = np.percentile(b, 95, axis=0)
        beqs = np.percentile(2.0 * hnorm * osc, 95) * times
        co = [{k[p]: c for p, c in OBS[name]}]
        d, p, s = combine(data, co)[..., 0], combine(pred, co)[..., 0], combine_sem(sem, co)[..., 0]
        dev = np.abs(d - p)
        covered = dev <= B + Z95 * s
        vac = times[B >= osc]
        sd = combine(boot, co)[..., 0].std(axis=0, ddof=1)  # (initial state, time)
        at = [np.argmin(np.abs(times - t)) for t in (1.5, 5.0, 10.0)]
        curves[name] = (dev, s, B, beqs, osc, sd)
        out[name] = {
            "out-of-sample coverage, Theorem A band + 1.96 SEM": float(covered[:, oos].mean()),
            "out-of-sample coverage, shot noise only": float((dev <= Z95 * s)[:, oos].mean()),
            "out-of-sample coverage while the band is non-vacuous": (
                float(covered[:, oos & (B < osc)].mean()) if np.any(oos & (B < osc)) else None),
            "band vacuous from (ms)": float(vac[0]) if vac.size else None,
            "Theorem A band at 1.5 / 5 / 10 ms": [float(B[np.argmin(np.abs(times - t))]) for t in (1.5, 5.0, 10.0)],
            "Corollary A.1 (BEQS form) at 1.5 ms": float(beqs[np.argmin(np.abs(times - 1.5))]),
            "Theorem A integrand at u = 0, bootstrap median (1/ms)": float(np.median(integ[:, 0])),
            "max |data - prediction| out of sample": float(dev[:, oos].max()),
            "quadrature: max relative change from the 0.5 ms grid": float(
                np.max(np.abs(b_coarse[:, 1:] - b[:, 1:]) / np.maximum(b[:, 1:], 1e-300))),
            "osc(O)": osc,
            "statistical band of the same observable (added after the run)": {
                "1.96 sd_boot at 1.5 / 5 / 10 ms, larger of the two initial states": [
                    float(Z95 * sd[:, j].max()) for j in at],
                "Theorem A band / 1.96 sd_boot at 5 / 10 ms": [float(B[j] / (Z95 * sd[:, j].max())) for j in at[1:]],
                "out-of-sample coverage, 1.96 sqrt(sd_boot^2 + SEM^2)": float(
                    (dev <= Z95 * np.sqrt(sd**2 + s**2))[:, oos].mean())},
        }
    return out, curves


def analyse_bands(bands, z):
    times, data, sem = z["times"], z["data"], z["sem"]
    cls = dict(charge_classes(observables()), **{"all 435 observables": [{m: 1.0} for m in range(data.shape[-1])]})
    oos, late = times > T_LEARN + 1e-9, times > T_LATE + 1e-9
    out = {}
    for v in MODELS:
        pred, row = z[f"pred_{v}"], {}
        for name, coeffs in cls.items():
            d, p, s = combine(data, coeffs), combine(pred, coeffs), combine_sem(sem, coeffs)
            sd = combine(bands[v], coeffs).std(axis=0, ddof=1)
            tot, dev = np.sqrt(sd**2 + s**2), d - p
            hit = np.abs(dev) <= Z95 * tot
            row[name] = {"coverage t > 1.5 ms": float(hit[:, oos].mean()), "coverage t > 5 ms": float(hit[:, late].mean()),
                         "calibration factor t > 1.5 ms": float(np.sqrt(np.mean(dev[:, oos] ** 2) / np.mean(tot[:, oos] ** 2))),
                         "median bootstrap sd / SEM, t > 1.5 ms": float(np.median(sd[:, oos] / s[:, oos]))}
        out[v] = row
    charged = [n for n in cls if n.startswith("q=±")]
    return out, charged


def plot_certified(curves, times, out):
    plt = style()
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), sharex=True)
    for ax, name in zip(axes.ravel(), OBS):
        dev, s, B, beqs, osc, sd = curves[name]
        ax.axvspan(0.0, T_LEARN, color=MUTED, alpha=0.08, lw=0)
        for st, (c, mk) in enumerate(((CAT[0], "o"), (CAT[1], "s"))):
            ax.errorbar(times + 0.06 * (st - 0.5), dev[st], yerr=Z95 * s[st], fmt=mk, ms=3.5, color=c, lw=0.8,
                        capsize=0, label=f"initial state {st + 1}")
            ax.plot(times, Z95 * sd[st], color=c, lw=1.0, ls="-.",
                    label="statistical band 1.96 sd_boot" if st == 0 else None)
        ax.plot(times, B, color=INK, lw=1.6, label="Theorem A, 95% band")
        ax.plot(times, beqs, color=INK, lw=1.0, ls="--", label="Corollary A.1 (BEQS form)")
        ax.axhline(osc, color=MUTED, lw=0.8, ls=":", label="osc(O): vacuous above")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 3 * osc)
        r = out[name]
        vac = r["band vacuous from (ms)"]
        stat = r["statistical band of the same observable (added after the run)"]
        ax.set_title(f"{name}\nTheorem A covers {r['out-of-sample coverage, Theorem A band + 1.96 SEM']:.0%}, "
                     + (f"vacuous from {vac:g} ms" if vac is not None else "never vacuous")
                     + f"\nstatistical band covers {stat['out-of-sample coverage, 1.96 sqrt(sd_boot^2 + SEM^2)']:.0%}",
                     fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("t (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("|data - X15 prediction|")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle("E15a: certified bands against held-out data (data: Kraft et al.; grey: learning window)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return save_figure(fig, "e15_certified")


def plot_bands(out):
    plt = style()
    names = list(out["X15"])
    yy = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True)
    for ax, key, ref in ((axes[0], "calibration factor t > 1.5 ms", 1.0), (axes[1], "coverage t > 1.5 ms", 0.95)):
        for v, c, off in (("X15", MUTED, -0.12), ("Xb15", CAT[0], 0.12)):
            ax.plot([out[v][n][key] for n in names], yy + off, "D", color=c, ms=5,
                    label={"X15": "X15 (Markov)", "Xb15": "Xb15 (memory)"}[v])
        ax.axvline(ref, color=INK, lw=0.8, ls=":")
        ax.set_xlabel(key)
    axes[0].set_yticks(yy, names)
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=7)
    axes[0].set_title("(a) rms deviation / predicted spread")
    axes[1].set_title("(b) coverage of the 95% band")
    fig.suptitle("E15b: statistical bands (bootstrap + shot noise) against held-out data (data: Kraft et al.)")
    fig.tight_layout()
    return save_figure(fig, "e15_bands")


if __name__ == "__main__":
    if "--replot" not in sys.argv:
        cert, bands = run()
        np.savez_compressed(RESULTS / "e15_coverage.npz", **{f"band_{v}": b for v, b in bands.items()},
                            **{f"{n}|osc": c[0] for n, c in cert.items()}, **{f"{n}|integrand": c[1] for n, c in cert.items()},
                            **{f"{n}|hnorm": c[2] for n, c in cert.items()})
    zz = np.load(RESULTS / "e15_coverage.npz")
    cert = {n: (float(zz[f"{n}|osc"]), zz[f"{n}|integrand"], zz[f"{n}|hnorm"]) for n in OBS}
    bands = {v: zz[f"band_{v}"] for v in MODELS}
    z = np.load(RESULTS / "e11_model_selection.npz")
    out_c, curves = analyse_certified(cert, z, bands["X15"])
    out_b, charged = analyse_bands(bands, z)
    out = {"certified (Theorem A, X15)": out_c, "statistical bands": out_b, "predictions": {
        "certified band covers >= 90% out of sample for every observable":
            all(r["out-of-sample coverage, Theorem A band + 1.96 SEM"] >= 0.9 for r in out_c.values()),
        "memory model has the smaller calibration factor in every charged class":
            all(out_b["Xb15"][n]["calibration factor t > 1.5 ms"] < out_b["X15"][n]["calibration factor t > 1.5 ms"]
                for n in charged)}}
    save_json(RESULTS / "e15_coverage.json", out)
    print(plot_certified(curves, z["times"], out_c))
    print(plot_bands(out_b))
    for key, v in out.items():
        print(f"{key}: {v}")
