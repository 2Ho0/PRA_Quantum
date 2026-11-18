"""E31: structure of the public data of the Google analog XY simulator (descriptive; supports research note Q54)

Data: measured by Google Quantum AI (Andersen et al., Nature 638, 79 (2025)), Zenodo 10.5281/zenodo.14060446 (CC BY 4.0);
files fetched with data/fetch_andersen2025.py (Fig2 Fig3_1 Fig5_1 Fig5_2). The files are pickles and are opened only
with qerr.googledata.safe_load (numpy constructors only).
Question (Q53 option 1): can these data give the excitation-number statistics needed for the strong-symmetry diagnostic
of E6b (leakage out of the initial sector, mean drift against variance growth)?
Checks (no test rules; descriptive):
  Fig2  (Z-basis quench from a half-filling product state, 10 sizes x 4 disorder instances x 21 times): does every
        stored distribution hold a single excitation number (post-selection)?
  Fig3_1 (ramps from the Neel state, 36 ramp times x 50,000 shots x 65 qubits): at the shortest ramp (10.4 ns, close to
        the Neel state) a Z or number-conserving measurement gives an excitation number fixed up to readout errors;
        X/Y single-qubit bases give a nearly binomial spread (variance near N/4) and marginals near 1/2.
  Fig5_1, Fig5_2 (dimer quenches, 64 qubits): at t = 0 the dimer partner of every qubit is the qubit with the largest
        |covariance|. The singlet-like dimer (|01> - |10>) is anticorrelated in X, Y and Z; the dimer (|01> + |10>),
        prepared in one half of the device in Fig5_2, is correlated in X and Y but anticorrelated in Z.
"""

import sys

import numpy as np

from common import RESULTS, ROOT, save_json

sys.path.insert(0, str(ROOT / "src"))
from qerr.googledata import safe_load  # noqa: E402

RAW = ROOT / "data" / "raw" / "andersen2025"


def fig2() -> dict:
    obj = safe_load(RAW / "Fig2")
    out = []
    for i, per_size in enumerate(obj["data"]):
        n_q = len(obj["qubits"][i])
        single, shots = [], []
        for per_instance in per_size:
            for dist in per_instance:
                keys = np.fromiter(dist.keys(), dtype=np.uint64, count=len(dist))
                single.append(np.unique(np.bitwise_count(keys)).size == 1)
                shots.append(int(sum(dist.values())))
        out.append({"n_qubits": n_q, "distributions": len(single), "single_excitation_number": int(sum(single)),
                    "shots_min": min(shots), "shots_max": max(shots)})
    return {"evolution_times_ns": obj["evolution_times"].tolist(), "sizes": out,
            "all_post_selected": all(s["single_excitation_number"] == s["distributions"] for s in out)}


def fig3_1() -> dict:
    obj = safe_load(RAW / "Fig3_1")
    x = obj["data"][0].astype(np.int8)  # shortest ramp only
    p = x.mean(axis=0)
    pos = {c: i for i, c in enumerate(obj["qubits"])}
    pairs = [(pos[a], pos[(a[0] + dx, a[1] + dy)]) for a in obj["qubits"] for dx, dy in ((1, 0), (0, 1))
             if (a[0] + dx, a[1] + dy) in pos]
    conn = [float(np.mean(x[:, i] * x[:, j]) - p[i] * p[j]) for i, j in pairs]
    n = x.sum(axis=1)
    return {"shortest_ramp_ns": float(obj["ramp_times"][0]), "shots": int(x.shape[0]), "qubits": int(x.shape[1]),
            "marginal_min": float(p.min()), "marginal_max": float(p.max()),
            "nn_pairs": len(pairs), "nn_connected_mean": float(np.mean(conn)),
            "excitation_mean": float(n.mean()), "excitation_var": float(n.var()), "binomial_var": x.shape[1] / 4}


def fig5(name: str) -> dict:
    obj = safe_load(RAW / name)
    x = obj["data"][0].astype(float)  # t = 0 only
    C = np.cov(x.T)
    np.fill_diagonal(C, 0.0)
    j = np.abs(C).argmax(axis=1)
    best = C[np.arange(C.shape[0]), j]
    return {"qubits": int(x.shape[1]), "shots": int(x.shape[0]), "partner_cov_negative": int((best < 0).sum()),
            "partner_cov_positive": int((best > 0).sum()), "partner_cov_abs_median": float(np.median(np.abs(best))),
            "mutual_pairs": int(sum(j[j[i]] == i for i in range(len(j))))}


def main() -> None:
    res = {"data": "Andersen et al., Nature 638, 79 (2025); Zenodo 10.5281/zenodo.14060446 (CC BY 4.0)",
           "Fig2": fig2(), "Fig3_1": fig3_1(), "Fig5_1": fig5("Fig5_1"), "Fig5_2": fig5("Fig5_2")}
    for k, v in res.items():
        print(k, v if k != "Fig2" else {kk: vv for kk, vv in v.items() if kk != "sizes"})
    for s in res["Fig2"]["sizes"]:
        print("  Fig2", s)
    save_json(RESULTS / "e31_google_structure.json", res)


if __name__ == "__main__":
    main()
