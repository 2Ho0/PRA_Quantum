"""E29b: where the data's window misfit of estimator T comes from (post hoc diagnostic, not preregistered)

Data: as E6a (measured by Kraft et al., PRX 16, 031037 (2026); Zenodo 10.5281/zenodo.17750419, CC BY 4.0), analysed
here from the public raw shots.
Question (after E29): T passes rule S3 in the synthetic drift worlds (pooled factors 1.20, 1.19, 1.21) but fails rule
B'' on the data (1.73, 1.45, 1.60; all three classes rejected), with chi^2 / nu = 2.42 in the window and couplings
12.7% above least squares. Two questions, answered without changing the verdicts of E29:
  (1) Inflation. Does the rejection survive when the data band is widened by the in-model pooled factor of each class
      (the band was about 20% too narrow in step B)? The null is rebuilt with the widened band.
  (2) Source of the misfit. T refitted on the data (same estimator, offset, prior scale and band as E29 step A) with
      (a) the 0.5 ms point left out (state 2's Z_iZ_j spike there, E23, E23k), (b) state 1 only, (c) state 2 only.
      Reported per variant: chi^2 / nu, lambda_T = sum J_T J_LS / sum J_LS^2, J_nn, and the late factors of the
      fitted states with their own band (p-values from the same null construction as rule B'').
Run: `python e29b_window_diagnostic.py` (three refits in parallel, about 10 minutes).
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from common import RESULTS, save_json  # noqa: E402
from e24_eiv_correction import N_WIN, NEUTRAL  # noqa: E402
from e29_neutral_trajectory import (CN, K_BAND, N_NULL_B, NN, SEEDS, SLICE, Neutral, band_draws, current_prior,  # noqa: E402
                                    fit_t, load)

VARIANTS = {"without t = 0.5 ms": ((2, 3), (0, 1)), "state 1 only": ((1, 2, 3), (0,)),
            "state 2 only": ((1, 2, 3), (1,))}


def factors_and_p(central, draws, data, sem, states, rng, scale=None):
    """Factors and one-sided null p-values per neutral class over the given states (scale widens the band sd)."""
    out = {}
    for cls in NEUTRAL:
        sl = SLICE[cls]
        k = 1.0 if scale is None else scale[cls]
        dr = central[None] + k * (draws - draws.mean(axis=0)[None])
        var = dr[:, states][..., sl].var(axis=0, ddof=1)
        s = sem[states][..., sl]
        tot = var + s**2
        f = float(np.sqrt(np.mean((data[states][..., sl] - central[states][..., sl]) ** 2) / np.mean(tot)))
        dev_b = dr[:, states][..., sl] - dr[:, states][..., sl].mean(axis=0)
        null = np.empty(N_NULL_B)
        for i in range(N_NULL_B):
            d = np.tensordot(rng.normal(size=K_BAND) / np.sqrt(K_BAND - 1), dev_b, axes=1) + rng.normal(size=s.shape) * s
            null[i] = np.sqrt(np.mean(d**2) / np.mean(tot))
        out[cls] = {"factor": f, "p": float(np.mean(null >= f)), "null 95%": float(np.quantile(null, 0.95))}
    return out


def run_variant(name):
    ti, st = VARIANTS[name]
    a = np.load(RESULTS / "e29_stepA.npz")
    times = load()[0]
    model = Neutral(a["bloch"])
    tw, tl = times[list(ti)], times[N_WIN:]
    y = (a["y"] - a["delta"][:, :N_WIN])[list(st)][:, list(ti)]
    sem = a["sem"][list(st)][:, list(ti)]
    x_hat, Sig, tau, info = fit_t(lambda x, tg: model(x, tg)[list(st)], y, sem, a["x_ls"], current_prior()[1], tw)
    rng = np.random.default_rng(SEEDS["band"] + 2900 + len(name))
    central, draws = band_draws(x_hat, Sig, a["bloch"], a["bsem"], tl, rng, offset=a["delta"][:, N_WIN:])
    dh = rng.normal(size=(K_BAND, 2)) @ np.linalg.cholesky(np.cov(a["hxy"].T)).T
    draws = draws + dh[:, 0, None, None, None] * a["Rx"][None, :, N_WIN:] + dh[:, 1, None, None, None] * a["Ry"][None, :, N_WIN:]
    return name, x_hat, info, central, draws


def main():
    z = np.load(RESULTS / "e11_model_selection.npz")
    a = np.load(RESULTS / "e29_stepA.npz")
    data = (z["data"] @ CN)[:, N_WIN:]
    sem = np.sqrt((z["sem"] ** 2) @ CN**2)[:, N_WIN:]
    e29 = json.loads((RESULTS / "e29_neutral_trajectory.json").read_text(encoding="utf-8"))
    scale = {c: e29["step B, arm redrawn (confirmatory)"]["per neutral class"][c]["pooled factor"] for c in NEUTRAL}
    rng = np.random.default_rng(29290)
    out = {"(1) E29 step A band widened by the in-model pooled factor": {
        "widening per class": scale,
        "both states": factors_and_p(a["central"], a["draws"], data, sem, [0, 1], rng, scale),
        "per state, E29 band as preregistered": {f"state {s + 1}": factors_and_p(a["central"], a["draws"], data, sem, [s], rng)
                                                 for s in (0, 1)}}}
    JL = a["x_ls"][:45]
    with ProcessPoolExecutor(max_workers=3) as pool:
        res = list(pool.map(run_variant, VARIANTS))
    out["(2) refits of T on the data"] = {}
    for name, x_hat, info, central, draws in res:
        st = list(VARIANTS[name][1])
        JT = x_hat[:45]
        out["(2) refits of T on the data"][name] = {
            "chi^2 / nu": info["chi2"] / info["nu"], "iterations": info["iterations"],
            "lambda_T = sum J_T J_LS / sum J_LS^2": float(JT @ JL / (JL @ JL)),
            "J_nn mean (rad/s)": float(JT[NN].mean() * 1e3),
            "late factors of the fitted states, own band": factors_and_p(central, draws, data, sem, st, rng)}
    out["E29 step A for reference: chi^2 / nu, lambda_T, J_nn (rad/s)"] = [
        float(a["chi2"] / a["nu"]), float(a["x_hat"][:45] @ JL / (JL @ JL)), float(a["x_hat"][:45][NN].mean() * 1e3)]
    return out


if __name__ == "__main__":
    out = main()
    save_json(RESULTS / "e29b_window_diagnostic.json", out)
    print(json.dumps(out, indent=1))
