"""Raw data of Joshi et al., PRL 133, 010402 (2024) ("Observing the quantum Mpemba effect in quantum simulations");
Zenodo 10.5281/zenodo.10390690 (CC BY 4.0). Twelve 40Ca+ ions.

Conventions (dataset Readme): every shot is a decimal number whose least significant bit is qubit 1; bit 0 is the +1
eigenvalue of the measured sigma^z, bit 1 the -1 eigenvalue. Before the measurement a random single-qubit unitary is
applied to every ion; the unitaries are stored in AppliedRandomUnitaries/RandUnitaries1to500.mat as a (2, 2, 500, 12)
array. The dephasing-only runs (interactions off) hold, for each tilt angle, files RU1 ... RU300 with 9 rows (times
0 ... 8 ms; first column in microseconds) and 30 shots per row. That their unitaries are the first 300 of the stored
500 was checked on the t = 0 rows (correlation 0.96 between predicted and measured bit frequencies, research note
Q56); the state measured is u rho u^dag.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

BASE = "MpembaDataUpload"


def decode(dec, n: int = 12) -> np.ndarray:
    """Decimal shot records -> bits uint8 (..., n), index k = qubit k + 1 (least significant bit first)."""
    ints = np.asarray(dec).astype(np.int64)
    return ((ints[..., None] >> np.arange(n, dtype=np.int64)) & 1).astype(np.uint8)


def load_unitaries(root) -> np.ndarray:
    """(500, 12, 2, 2) complex: unitary k (0-based) applied to qubit q before the z measurement."""
    m = loadmat(Path(root) / BASE / "AppliedRandomUnitaries" / "RandUnitaries1to500.mat")["UnitaryOpsListed"]
    return np.ascontiguousarray(np.transpose(m, (2, 3, 0, 1)))


def load_dephasing(root, theta_tag: str, n_units: int = 300):
    """times_ms (9,), bits uint8 (9, n_units, 30, 12) of the dephasing-only run at tilt theta_tag ('0p2', '0p33', '0p5')."""
    folder = Path(root) / BASE / "RawData" / "DephasingData" / f"Theta{theta_tag}_JustDephas_VariousTimesR3"
    rows = []
    times = None
    for k in range(1, n_units + 1):
        d = loadmat(folder / f"Theta{theta_tag}_JustDephasR3_RU{k}.mat")["data"]
        if times is None:
            times = d[:, 0].astype(float) / 1000.0
        elif not np.allclose(d[:, 0] / 1000.0, times):
            raise ValueError(f"unit {k}: time grid differs")
        rows.append(decode(d[:, 1:]))
    return times, np.stack(rows, axis=1)
