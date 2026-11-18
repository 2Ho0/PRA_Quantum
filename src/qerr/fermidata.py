"""Hardware shots of Phasecraft and Quantinuum, arXiv:2510.26300; Zenodo 10.5281/zenodo.17881394 (CC BY 4.0).

Format (dataset README): shots.h5 has a group "data" with one dataset per evolution time (the name is the time),
holding bit-packed shot arrays (np.packbits along the qubit axis) with attrs["original_shape"] = (shots, qubits), and
a group "metadata" whose attrs include "qubit_order", the label "(x,y,up)" or "(x,y,down)" of every qubit in the
Jordan-Wigner encoding of the Fermi-Hubbard model. A bit 1 means a fermion was detected in that mode (z basis).
"""

import re
from pathlib import Path

import h5py
import numpy as np

_LABEL = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(up|down)\s*\)")


def load_shots(path):
    """(times sorted ascending, shots uint8 of shape (times, shots, qubits), metadata dict)."""
    with h5py.File(Path(path), "r") as h5:
        g = h5["data"]
        items = []
        for key in g.keys():
            d = g[key]
            rows, cols = (int(x) for x in d.attrs["original_shape"])
            bits = np.unpackbits(d[()], axis=-1)[:, :cols].astype(np.uint8)
            if bits.shape != (rows, cols):
                raise ValueError(f"time {key}: shape {bits.shape} != original_shape {(rows, cols)}")
            items.append((float(key), bits))
        meta = {}
        for k, v in h5["metadata"].attrs.items():
            if isinstance(v, np.ndarray) and v.dtype.kind in "OSU":
                v = [x.decode() if isinstance(x, bytes) else str(x) for x in v.tolist()]
            elif isinstance(v, np.generic):
                v = v.item()
            meta[k] = v
    items.sort(key=lambda x: x[0])
    sizes = {b.shape for _, b in items}
    if len(sizes) != 1:
        raise ValueError(f"unequal shot arrays across times: {sizes}")
    times = np.array([t for t, _ in items])
    return times, np.stack([b for _, b in items]), meta


def species_masks(qubit_order):
    """Boolean masks (up, down) over the qubit axis, from labels like "(x,y,up)"."""
    spins = []
    for label in qubit_order:
        m = _LABEL.fullmatch(str(label).strip())
        if m is None:
            raise ValueError(f"unexpected qubit label {label!r}")
        spins.append(m.group(3))
    spins = np.array(spins)
    return spins == "up", spins == "down"


def number_histogram(shots, mask):
    """Shots per particle number of the modes in mask (length mask.sum() + 1)."""
    n = np.asarray(shots)[..., np.asarray(mask, dtype=bool)].sum(axis=-1)
    return np.bincount(n.ravel(), minlength=int(np.sum(mask)) + 1)
