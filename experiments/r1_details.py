"""R1 details (post hoc): how far inside the tolerance the reproduced results lie.

R1 (experiments/reproduce_all.py, rules committed in fdcc319) reproduced every committed result from the raw data
without caches and compared the 59 result JSON files with the reference commit. Its verdict is in
results/r1_reproduction.json: 55 files identical, 4 agree, none differ, no verdict changed. This script was written
after that verdict and does not change it. It describes the margin:

1. JSON values (the leaves the R1 rule compares, with the same exclusions): the difference of each value divided by
   the difference that the "agree" rule allows, d / (1e-9 + 1e-3 max(|a|, |b|)). Numbers inside strings are compared
   one by one, as in the rule, and a string leaf takes the largest ratio of its numbers.
2. npz arrays (not covered by the rule; the figures are drawn from them): the same ratio for every element.

A ratio of 0 means bit-identical; the rule accepts a value at a ratio of at most 1.

Run:  python experiments/r1_details.py --clone <R1 clone>   (computes results/r1_details.json and draws the figure)
      python experiments/r1_details.py --replot             (redraws the figure from results/r1_details.json)
"""

import io
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import CAT, INK, MUTED, RESULTS, ROOT, save_figure, save_json, style
from reproduce_all import ATOL_AG, EXCLUDE, NUM_RE, REFERENCE, RTOL_AG, flatten

# log10 of the ratio. Smaller nonzero ratios go into the first bin: they reach 1e-295 for values that are zero to
# working precision (e.g. 1e-300 against 0), so the axis would otherwise be useless.
EDGES = np.arange(-18.0, 0.5 + 1e-9, 0.5)


def number_ratio(a, b):
    """d / (allowed d) for two numbers (real or complex); 0 if bit-identical."""
    if a == b:
        return 0.0
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return 0.0
    d, m = abs(a - b), max(abs(a), abs(b))
    r = d / (ATOL_AG + RTOL_AG * m)
    return math.inf if math.isnan(r) else r


def string_ratio(a, b):
    """Largest ratio of the numbers inside two strings; inf if the text around the numbers changed."""
    if NUM_RE.sub("#", a) != NUM_RE.sub("#", b):
        return math.inf
    return max([number_ratio(float(u), float(v)) for u, v in zip(NUM_RE.findall(a), NUM_RE.findall(b))] or [0.0])


def leaf_ratio(a, b):
    def is_num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)
    if is_num(a) and is_num(b):
        return number_ratio(float(a), float(b))
    if isinstance(a, str) and isinstance(b, str):
        return string_ratio(a, b)
    return 0.0 if type(a) is type(b) and a == b else math.inf


def array_ratios(a, b):
    """Ratios of all elements of two loaded npz entries (numeric, string or object arrays)."""
    if isinstance(a, np.ndarray) and a.dtype == object:
        a, b = a.tolist(), (b.tolist() if isinstance(b, np.ndarray) else b)
    if isinstance(a, (list, tuple, dict)):
        if type(a) is not type(b) or len(a) != len(b) or (isinstance(a, dict) and set(a) != set(b)):
            return np.array([np.inf])
        pairs = [(a[k], b[k]) for k in a] if isinstance(a, dict) else list(zip(a, b))
        parts = [array_ratios(u, v) for u, v in pairs]
        return np.concatenate(parts) if parts else np.zeros(0)
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return np.array([np.inf])
    if a.dtype.kind in "US" and b.dtype.kind in "US":
        return np.array([string_ratio(u, v) for u, v in zip(a.ravel().tolist(), b.ravel().tolist())])
    if a.dtype.kind not in "biufc" or b.dtype.kind not in "biufc":
        return np.array([0.0 if np.array_equal(a, b) else np.inf])
    kind = complex if "c" in a.dtype.kind + b.dtype.kind else float
    x, y = np.atleast_1d(a.astype(kind)).ravel(), np.atleast_1d(b.astype(kind)).ravel()
    with np.errstate(invalid="ignore"):
        r = np.abs(x - y) / (ATOL_AG + RTOL_AG * np.maximum(np.abs(x), np.abs(y)))
    r[(x == y) | (np.isnan(x) & np.isnan(y))] = 0.0
    r[np.isnan(r)] = np.inf
    return r


def summary(ratios, where):
    ratios = np.asarray(ratios, float)
    positive = ratios[(ratios > 0) & np.isfinite(ratios)]
    counts, _ = np.histogram(np.clip(np.log10(positive), EDGES[0], EDGES[-1] - 1e-12), EDGES)
    k = int(np.argmax(ratios)) if ratios.size else 0
    return {"values": int(ratios.size), "bit identical": int(np.sum(ratios == 0)),
            "outside the tolerance (ratio > 1)": int(np.sum(ratios > 1)),
            "largest ratio": float(ratios[k]) if ratios.size else 0.0, "largest at": where[k] if ratios.size else None,
            "smallest nonzero ratio": float(positive.min()) if positive.size else None,
            "histogram": {"log10 ratio edges": EDGES.tolist(), "counts": counts.tolist()}}


def git_show(path):
    return subprocess.run(["git", "-C", str(ROOT), "show", f"{REFERENCE}:{path}"], capture_output=True,
                          check=True).stdout


def compute(clone):
    clone = Path(clone)
    names = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "--name-only", REFERENCE, "results/"],
                           capture_output=True, text=True, check=True).stdout.split()
    jr, jw = [], []
    for n in sorted(x for x in names if x.endswith(".json")):
        ref = dict(flatten(json.loads(git_show(n).decode("utf-8"))))
        new = dict(flatten(json.loads((clone / n).read_text(encoding="utf-8"))))
        for p, a in ref.items():
            key = " / ".join(map(str, p))
            if any(e in key for e in EXCLUDE):
                continue
            jr.append(leaf_ratio(a, new[p]) if p in new else math.inf)
            jw.append(f"{Path(n).name}: {key}")
    nr, nw, arrays = [], [], 0
    npz = sorted(x for x in names if x.endswith(".npz"))
    for n in npz:
        # both files are our own outputs (git history and the R1 clone), so object arrays may be unpickled
        ref = np.load(io.BytesIO(git_show(n)), allow_pickle=True)
        new = np.load(clone / n, allow_pickle=True)
        for k in ref.files:
            r = array_ratios(ref[k], new[k]) if k in new.files else np.array([np.inf])
            nr.append(r)
            nw.extend([f"{Path(n).name}: {k}"] * r.size)
            arrays += 1
    setup = json.loads((clone / "repro_setup.json").read_text(encoding="utf-8"))
    out = {"note": "post hoc description of R1 (does not change its verdict); ratio = difference / allowed difference "
                   "of the 'agree' rule, d / (1e-9 + 1e-3 max(|a|,|b|)); 0 = bit-identical",
           "reference commit": REFERENCE, "R1 setup commit": setup["commit"],
           "json": {"files": sum(x.endswith(".json") for x in names), **summary(jr, jw)},
           "npz": {"files": len(npz), "arrays": arrays, **summary(np.concatenate(nr), nw)}}
    save_json(RESULTS / "r1_details.json", out)
    return out


def plot(d):
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(510 / 72, 2.5), sharex=True)
    titles = {"json": "(a) Result values (JSON, compared by the R1 rule)",
              "npz": "(b) Arrays behind the figures (npz, post hoc)"}
    for ax, key in zip(axes, ("json", "npz")):
        p = d[key]
        edges = np.array(p["histogram"]["log10 ratio edges"])
        counts = np.array(p["histogram"]["counts"], float)
        keep = counts > 0
        ax.bar(edges[:-1][keep], counts[keep], width=np.diff(edges)[keep], align="edge", color=CAT[0],
               edgecolor="white", linewidth=0.5)
        ax.set_yscale("log")
        top = 10.0 ** (math.ceil(np.log10(counts.max())) + 3)  # headroom for the note above the bars
        ax.set_ylim(0.5, top)
        ax.axvline(0.0, color=INK, ls="--", lw=0.8)
        ax.text(-0.25, top / 3, "allowed\ndifference", ha="right", va="top", color=INK, fontsize=8)
        ax.set_xlim(edges[0], 1.0)
        ticks = np.arange(edges[0], 0.5, 3.0)
        labels = [f"{t:.0f}".replace("-", "−") for t in ticks]
        labels[0] = "≤" + labels[0]  # the first bin also holds all smaller nonzero ratios
        ax.set_xticks(ticks, labels)
        ax.set_title(titles[key], loc="left")
        ax.set_xlabel(r"$\log_{10}$(difference / allowed difference)")
        ax.text(0.02, 0.97, f"bit-identical: {p['bit identical']:,} of {p['values']:,}\n"
                            f"largest: {p['largest ratio']:.1e} of the allowed\n"
                            f"beyond the allowed: {p['outside the tolerance (ratio > 1)']}",
                transform=ax.transAxes, ha="left", va="top", color=MUTED, fontsize=8)
    axes[0].set_ylabel("values")
    fig.suptitle("R1: reproduced results against the committed ones")
    fig.tight_layout()
    return save_figure(fig, "r1_details")


if __name__ == "__main__":
    if "--replot" in sys.argv:
        data = json.loads((RESULTS / "r1_details.json").read_text(encoding="utf-8"))
    elif "--clone" in sys.argv:
        data = compute(sys.argv[sys.argv.index("--clone") + 1])
    else:
        sys.exit(__doc__)
    print(plot(data))
    for k in ("json", "npz"):
        print(k, {x: v for x, v in data[k].items() if x != "histogram"})
