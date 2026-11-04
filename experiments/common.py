"""Shared helpers for the experiment scripts: base model, operators, JSON output, plot style."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qerr.charge import charge_projector  # noqa: E402
from qerr.lindblad import OpenModel  # noqa: E402
from qerr.models import charge_basis_terms, delta_h, xy_couplings, xy_hamiltonian  # noqa: E402
from qerr.spins import basis_state, pauli_op  # noqa: E402

RESULTS = ROOT / "results"

# Validated with the dataviz skill's validator (light surface):
# ordinal blue ramp for ordered sweeps; categorical slots 1-3 for distinct entities.
ORDINAL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def base_model(n, gamma_c, gamma_loc, gamma_em, B_inhom=0.1, seed=0, alpha=1.0) -> OpenModel:
    """Long-range XY (J0 = 1) + weak random z fields, collective + local sigma^z dephasing, emission."""
    rng = np.random.default_rng(seed)
    H = xy_hamiltonian(n, xy_couplings(n, 1.0, alpha), B=B_inhom * rng.normal(size=n))
    return OpenModel(H=H, Gamma=gamma_c * np.ones((n, n)) + gamma_loc * np.eye(n),
                     emission=gamma_em * np.ones(n))


def neel(n):
    return basis_state(n, "01" * (n // 2) + "0" * (n % 2))


def random_perturbation(n, abs_charges, norm, seed, max_range=None):
    """Random combination of charge-basis terms with |q| in abs_charges, rescaled to operator norm `norm`."""
    rng = np.random.default_rng(seed)
    terms = [t for t in charge_basis_terms(n, max_range) if t.abs_charge in abs_charges]
    dH = delta_h(terms, rng.normal(size=len(terms)))
    return dH * (norm / np.linalg.norm(dH, 2))


def charge_norms(n, X):
    """{q: ||P_q X||} for the nonzero charged components."""
    out = {}
    for q in range(-n, n + 1):
        nq = np.linalg.norm(charge_projector(n, X, q), 2)
        if nq > 1e-12:
            out[q] = float(nq)
    return out


def ops(n, i):
    return {k: pauli_op(n, i, k) for k in "xyz"}


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=float))


def save_figure(fig, stem):
    """Write results/<stem>.png (with its suptitle) and results/<stem>.pdf (for the paper: caption replaces the
    suptitle, so it is hidden); return the PNG path."""
    png = RESULTS / f"{stem}.png"
    fig.savefig(png, bbox_inches="tight")
    title = fig._suptitle
    if title is not None:
        title.set_visible(False)
    fig.savefig(RESULTS / f"{stem}.pdf", bbox_inches="tight")
    if title is not None:
        title.set_visible(True)
    return png


def triangle_integral(f, dt):
    """int_{a+b<=t_K} f(a, b) da db for every K, f sampled at (i dt, j dt), i + j <= K_max.

    Exact for the piecewise-linear interpolant: weight 1 inside, 1/2 on edges, 1/6 at the corners.
    """
    K = f.shape[0] - 1
    out = np.zeros(K + 1)
    for Kp in range(1, K + 1):
        k, j = np.meshgrid(np.arange(Kp + 1), np.arange(Kp + 1), indexing="ij")
        w = np.where(k + j <= Kp, 1.0, 0.0) * np.where((k == 0) | (j == 0) | (k + j == Kp), 0.5, 1.0)
        w[0, 0], w[Kp, 0], w[0, Kp] = 1 / 6, 1 / 6, 1 / 6
        out[Kp] = np.sum(np.where(w > 0, w * np.nan_to_num(f[: Kp + 1, : Kp + 1]), 0.0)) * dt**2
    return out


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 1.6, "legend.frameon": False,
        "text.color": INK, "figure.facecolor": "white", "axes.facecolor": "white",
    })
    return plt
