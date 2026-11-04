"""E4: Gaussian uncertainty on a dissipative rate breaks long-time predictions (Proposition C).

One two-level system with emission rate gamma: p(t) = p0 exp(-gamma t). With gamma ~ N(mu, s^2),
E[p(t)] = exp(-mu t + s^2 t^2 / 2), which exceeds 1 (unphysical) for t > 2 mu / s^2. A log-normal
rate with the same mean and variance keeps every sample physical. Values mimic the BEQS 10-ion
emission estimate (mu ~ 2 rad/s, error bar of the same size), with time in seconds.
"""

import numpy as np
from scipy import integrate, stats

from common import CAT, INK, MUTED, RESULTS, save_json, style

MU, S = 2.0, 2.0
T = np.linspace(0.0, 2.0, 201)


def gaussian_mean(t):
    return np.exp(-MU * t + S**2 * t**2 / 2)


def lognormal_mean(t):
    var_ln = np.log(1 + S**2 / MU**2)
    dist = stats.lognorm(s=np.sqrt(var_ln), scale=np.exp(np.log(MU) - var_ln / 2))
    return np.array([integrate.quad(lambda g: np.exp(-g * ti) * dist.pdf(g), 0, np.inf)[0] for ti in T])


def run():
    rng = np.random.default_rng(1)
    gam = rng.normal(MU, S, size=400_000)
    t_mc = T[T <= 1.2]
    mc = np.array([np.exp(-gam * t).mean() for t in t_mc])
    ln = lognormal_mean(T)
    t_star = 2 * MU / S**2
    acc = {"t* = 2 mu / s^2": t_star,
           "P(gamma < 0) (non-CP samples)": float(stats.norm.cdf(-MU / S)),
           "Gaussian mean exceeds 1 for t > t*": bool(np.all(gaussian_mean(T[T > t_star * 1.01]) > 1)),
           "log-normal mean <= 1 everywhere": bool(np.all(ln <= 1 + 1e-12)),
           "MC matches analytic Gaussian mean for t <= 1.2 (max rel err)": float(np.max(np.abs(mc / gaussian_mean(t_mc) - 1)))}
    return dict(mc=(t_mc, mc), ln=ln, acc=acc)


def plot(res):
    plt = style()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.semilogy(T, gaussian_mean(T), color=CAT[1], label="Gaussian rate: exp(−μt + s²t²/2)")
    t_mc, mc = res["mc"]
    ax.semilogy(t_mc[::10], mc[::10], "o", color=CAT[1], ms=4, mfc="white", label="Gaussian rate, Monte Carlo")
    ax.semilogy(T, res["ln"], color=CAT[0], ls="--", label="log-normal rate (same mean, sd)")
    ax.axhline(1.0, color=INK, lw=0.9)
    ax.axvline(res["acc"]["t* = 2 mu / s^2"], color=MUTED, lw=0.8, ls=":")
    ax.text(res["acc"]["t* = 2 mu / s^2"] * 1.03, 30, "t* = 2μ/s²", color=MUTED, fontsize=8)
    ax.text(1.98, 1.25, "physical maximum p(0) = 1", color=INK, fontsize=7.5, ha="right")
    ax.set_xlabel("time  t  [s]   (μ = 2 rad/s, s = 2 rad/s)")
    ax.set_ylabel("ensemble mean of excited population")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("E4  Gaussian rate uncertainty is unphysical at long times", fontsize=9.5)
    fig.tight_layout()
    path = RESULTS / "e4_cp_pathology.png"
    fig.savefig(path, bbox_inches="tight")
    return path


if __name__ == "__main__":
    res = run()
    save_json(RESULTS / "e4_cp_pathology.json", {"acceptance": res["acc"]})
    print(plot(res))
    for k, v in res["acc"].items():
        print(f"{k}: {v}")
