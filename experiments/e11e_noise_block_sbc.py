"""E11e: is the charged-sector mismatch of E11c an artefact of the fit? Simulation-based calibration of the noise block.

The truths of E14 (Y15 couplings and z fields, D5/2 emission, white/quasi-static collective mixtures) contain no
single-ion noise at all. Synthetic data are sampled exactly like the BEQS data (measured by Kraft et al., PRX 16,
031037 (2026); the same 200 settings, 200 shots, both initial states; qerr.shots) for f = 0, 1/2, 1, and fitted with
the E11c ansatz: U(1)-symmetric couplings, all single-site fields, collective Gamma and the full single-ion
Kossakowski block on t <= 1.5 ms, without and with the collective slope beta. Two defects: the truth's own trajectory
(oracle) and the white truth's trajectory, which mimics the reference-model defect used for the data.
Fixed before the run, for the fit without memory:
  "artefact" if for some truth and defect the median charged rms reaches half the measured value and the median
  number of non-PSD XY blocks is at least 5 of 10;
  "in the data" if the measured charged rms exceeds the 95th percentile for every truth and defect;
  "inconclusive" otherwise.
After the first run (verdict: in the data), a second run with the same seeds also stores the site mean of every
combination; the statistics of the rule are reproduced exactly, and the comparison per combination is post hoc.
"""

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import INK, MUTED, ORDINAL, RESULTS, save_figure, save_json, style
from e6_ion_learning import RAD_S, ansatz, observables
from e6c_validation import load
from e9_symmetry_data import COMBOS, extended_ansatz
from e11c_noise_ansatz import fit_noise_block, summary
from e14_shot_calibration import defects_for, synthetic_series, truth_job, truth_theta
from qerr.learning import needed_strings

TRUTHS, N_REAL = (0.0, 0.5, 1.0), 20
CHARGED = [n for n in COMBOS if not n.startswith(("flip", "imbalance"))]
_W = {}


def block_stats(theta):
    """(rms of the charged site means in rad/s, ions with a non-PSD XY block, site-mean total flip rate, then the site
    mean of every combination in COMBOS)."""
    cv, em, ab, _ = summary(theta, extended_ansatz())
    rms = float(np.sqrt(np.mean([cv[n].mean() ** 2 for n in CHARGED])))
    return (rms, int(np.sum(np.abs(cv["SXX-SYY"]) > cv["flip S=SXX+SYY"])), float((em + ab).mean()),
            *[float(cv[n].mean()) for n in COMBOS])


def _init(prob, defects, bases, strings, times):
    _W.update(prob=prob, defects=defects, bases=bases, strings=strings, times=times)


def realisation(args):
    f, seed = args
    series = synthetic_series(_W["prob"][f], _W["bases"], _W["strings"], np.random.default_rng(seed))
    P = extended_ansatz().n_params
    out = []
    for g in (f, 0.0):  # oracle defect, white-truth defect
        d = _W["defects"][g]
        th = fit_noise_block(series, _W["times"], d[((0.0, 1.5), False)], d[((0.0, 1.5), True)])
        out.append([block_stats(th[:P]), block_stats(th[P:2 * P])])
    return np.array(out)  # (defect, memory, stat)


def run():
    times, s1, s2, bases = load()
    obs = observables()
    strings = sorted(set(obs) | set(needed_strings(obs, ansatz())))
    with ProcessPoolExecutor(max_workers=2) as pool:
        per_state = list(pool.map(truth_job, (0, 1)))
    prob = {f: [per_state[s][f][1] for s in (0, 1)] for f in TRUTHS}
    defects = {f: defects_for([per_state[s][f][0] for s in (0, 1)], strings, times) for f in TRUTHS}
    del per_state
    jobs = [(f, 7919 * i + int(1000 * f)) for f in TRUTHS for i in range(N_REAL)]
    with ProcessPoolExecutor(max_workers=8, initializer=_init, initargs=(prob, defects, bases, strings, times)) as pool:
        return np.array(list(pool.map(realisation, jobs))).reshape(len(TRUTHS), N_REAL, 2, 2, -1)


def measured():
    z = np.load(RESULTS / "e11c_noise_ansatz.npz")
    P = extended_ansatz().n_params
    return [block_stats(z["central"][:P]), block_stats(z["central"][P:2 * P])]


def truth_flip_rate():
    """Site-mean total flip rate of the truths: the D5/2 emission alone (rad/s)."""
    a = ansatz()
    k = len(a.h_terms) + len(a.deph_pairs)
    return float(truth_theta()[k:k + len(a.emission_sites)].mean() * RAD_S)


def analyse(stats):
    meas = measured()
    out = {"measured (E11c), without / with memory: charged rms, non-PSD ions, flip rate": [m[:3] for m in meas]}
    rule = {"artefact": False, "in the data": True}
    for i, f in enumerate(TRUTHS):
        for j, dname in enumerate(("oracle defect", "white-truth defect")):
            row = {}
            for m, mname in enumerate(("without memory", "with memory")):
                s = stats[i, :, j, m]
                row[mname] = {"charged rms median, 95th percentile (rad/s)": [float(np.median(s[:, 0])),
                                                                             float(np.percentile(s[:, 0], 95))],
                              "non-PSD ions median, min, max": [float(np.median(s[:, 1])), int(s[:, 1].min()),
                                                                int(s[:, 1].max())],
                              "flip rate median, 5th, 95th percentile (rad/s)": [
                                  float(np.median(s[:, 2])), *np.percentile(s[:, 2], [5, 95]).tolist()]}
            out[f"truth f={f:g}, {dname}"] = row
            s = stats[i, :, j, 0]
            rule["artefact"] |= bool(np.median(s[:, 0]) >= 0.5 * meas[0][0] and np.median(s[:, 1]) >= 5)
            rule["in the data"] &= bool(meas[0][0] > np.percentile(s[:, 0], 95))
    out["verdict"] = ("artefact" if rule["artefact"] else "in the data" if rule["in the data"] else "inconclusive")
    s = stats[:, :, :, 0]
    p5, p95 = np.percentile(s[..., 2], [5, 95], axis=1)
    out["post hoc, not part of the rule"] = {
        "truth total flip rate, D5/2 emission (rad/s)": truth_flip_rate(),
        "measured charged rms / synthetic median: min, max over truths and defects": [
            float(x) for x in meas[0][0] / np.percentile(np.median(s[..., 0], axis=1), [0, 100])[::-1]],
        "measured charged rms / synthetic 95th percentile: min, max": [
            float(x) for x in meas[0][0] / np.percentile(np.percentile(s[..., 0], 95, axis=1), [0, 100])[::-1]],
        "measured flip rate inside the synthetic 5-95% range, per truth (oracle, white-truth defect)": {
            f"f={f:g}": [bool(p5[i, j] <= meas[0][2] <= p95[i, j]) for j in range(2)] for i, f in enumerate(TRUTHS)}}
    if s.shape[-1] > 3:  # site means per combination (stored since the second run)
        med = np.median(s[..., 3:], axis=1)  # (truth, defect, combination)
        i1 = TRUTHS.index(1.0)
        out["post hoc, not part of the rule"]["site means per combination, without memory (rad/s)"] = {
            n: {"measured": meas[0][3 + c],
                "synthetic median per truth, oracle defect": [float(med[i, 0, c]) for i in range(len(TRUTHS))],
                "synthetic 5th, 95th percentile at f = 1, oracle defect": np.percentile(
                    s[i1, :, 0, 3 + c], [5, 95]).tolist(),
                "measured minus synthetic median: min, max over truths and defects": [
                    float((meas[0][3 + c] - med[..., c]).min()), float((meas[0][3 + c] - med[..., c]).max())]}
            for c, n in enumerate(COMBOS)}
        out["post hoc, not part of the rule"][
            "site-mean S_XX after subtracting the synthetic median (oracle defect): value (rad/s), z with the data "
            "bootstrap"] = {f"f={f:g}": psd_check(s[i, :, 0, 3:], meas[0][3:]) for i, f in enumerate(TRUTHS)}
    return out


def psd_check(synthetic, data):
    """S_XX = (S + K)/2 of the site-mean XY block, corrected by the synthetic median; a GKSL block needs S_XX >= 0."""
    iS, iK = list(COMBOS).index("flip S=SXX+SYY"), list(COMBOS).index("SXX-SYY")
    P = extended_ansatz().n_params
    boot = [summary(th, extended_ansatz())[0] for th in np.load(RESULTS / "e11c_noise_ansatz.npz")["boot"][:, :P]]
    sd_data = np.std([(b["flip S=SXX+SYY"] + b["SXX-SYY"]).mean() / 2 for b in boot], ddof=1)
    xx = (synthetic[:, iS] + synthetic[:, iK]) / 2
    value = (data[iS] + data[iK]) / 2 - np.median(xx)
    return [float(value), float(value / np.hypot(sd_data, xx.std(ddof=1) / np.sqrt(len(xx))))]


def plot(stats, out):
    from matplotlib.ticker import MaxNLocator
    plt = style()
    meas = out["measured (E11c), without / with memory: charged rms, non-PSD ions, flip rate"]
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 3.6))
    labels = ("rms of the charged site means (rad/s)", "ions with a non-PSD XY block", "site-mean total flip rate (rad/s)")
    for k, (ax, lab) in enumerate(zip(axes, labels)):
        for i, f in enumerate(TRUTHS):
            for j, mk in enumerate(("o", "s")):
                v = stats[i, :, j, 0, k]
                ax.plot(v, np.full(v.size, i + 0.15 * (j - 0.5)) + np.linspace(-0.04, 0.04, v.size), mk,
                        color=ORDINAL[i + 1], ms=3.5, alpha=0.8,
                        label=("oracle defect" if j == 0 else "white-truth defect") if i == 1 and k == 0 else None)
        ax.axvline(meas[0][k], color=INK, lw=1.4, label="BEQS data (E11c)")
        ax.set_yticks(range(len(TRUTHS)), [f"truth f = {f:g}" for f in TRUTHS])
        ax.set_xlabel(lab)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[2].axvline(out["post hoc, not part of the rule"]["truth total flip rate, D5/2 emission (rad/s)"], color=MUTED,
                    ls="--", lw=1.2)
    axes[0].legend(fontsize=7)
    axes[0].set_title("(a) charged entries without single-ion noise in the truth", fontsize=9)
    axes[1].set_title("(b) non-PSD blocks", fontsize=9)
    axes[2].set_title("(c) total flip rate, post hoc (dashed: the truth, D5/2 emission)", fontsize=9)
    fig.suptitle(f"E11e: calibration of the single-ion noise block (verdict: {out['verdict']})")
    fig.tight_layout()
    return save_figure(fig, "e11e_noise_block_sbc")


def plot_combinations(stats, out):
    """Post hoc: the site mean of every combination, synthetic (oracle defect, no memory) against the data."""
    plt = style()
    meas = measured()[0][3:]
    names = list(COMBOS)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, f in enumerate(TRUTHS):
        v = stats[i, :, 0, 0, 3:]
        for c in range(len(names)):
            ax.plot(v[:, c], np.full(N_REAL, c + 0.22 * (i - 1)), "o", color=ORDINAL[i + 1], ms=3, alpha=0.7,
                    label=f"synthetic, truth f = {f:g}" if c == 0 else None)
    ax.plot(meas, range(len(names)), "D", color=INK, ms=6, label="BEQS data (E11c)")
    ax.axvline(0.0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel("site mean over the 10 ions (rad/s)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("E11e, post hoc: noise-block combinations without single-ion noise in the truth", fontsize=9)
    fig.tight_layout()
    return save_figure(fig, "e11e_combinations")


if __name__ == "__main__":
    if "--replot" not in sys.argv:
        np.savez_compressed(RESULTS / "e11e_noise_block_sbc.npz", stats=run())
    stats = np.load(RESULTS / "e11e_noise_block_sbc.npz")["stats"]
    out = analyse(stats)
    save_json(RESULTS / "e11e_noise_block_sbc.json", out)
    print(plot(stats, out))
    if stats.shape[-1] > 3:
        print(plot_combinations(stats, out))
    for key, v in out.items():
        print(f"{key}: {v}")
