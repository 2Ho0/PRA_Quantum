"""E13c: the exact density propagator of the dephased XX chain behind Proposition F.1, against E13.

For hopping J and number dephasing g on an infinite chain, the two-point matrix in centre-of-mass momentum k and
relative distance r obeys  dC_r/dt = -a_k (C_{r+1} - C_{r-1}) - g (1 - delta_{r0}) C_r,  a_k = 2 J sin(k/2).
For product initial states only C_0 (the density mode) starts nonzero, and its Laplace transform is
    G_k(p) = 1 / ( sqrt((p + g)^2 + 4 a_k^2) - g ),
with the slow pole p_+ = -g + sqrt(g^2 - 4 a_k^2) = -D k^2 + O(k^4), D = 2 J^2 / g; all other contributions decay as
exp(-g t). Check: the crest density of the E13 density waves (reference run) equals 1/2 + 0.4 G_k(t), with G_k
from the truncated r-chain, and its decay rate in the slow-mode regime (10/g < t < 3/(D k^2)) equals -p_+.
lambda = 160 has a node at the open boundary of the N = 400 chain, so a reflected boundary layer reaches the
centre within t = 1500; the other wavelengths have an extremum there and match the infinite-chain result.
"""

import numpy as np
from scipy.integrate import solve_ivp

from common import RESULTS, save_json
from e13_two_sided_bound import G_DEPH, U

J, R = 1.0, 60


def density_propagator(k, times):
    a = 2.0 * J * np.sin(k / 2.0)
    r = np.arange(-R, R + 1)
    damp = np.where(r == 0, 0.0, G_DEPH)

    def rhs(_, c):
        out = -damp * c
        out[:-1] -= a * c[1:]
        out[1:] += a * c[:-1]
        return out

    c0 = (r == 0).astype(float)
    sol = solve_ivp(rhs, (0.0, times[-1]), c0, t_eval=times, method="DOP853", rtol=1e-11, atol=1e-14)
    return sol.y[R]


if __name__ == "__main__":
    z = np.load(RESULTS / "e13_two_sided_bound.npz")
    out = {}
    for lam in (20, 40, 80, 160):
        k = 2 * np.pi / lam
        n_ref = z[f"state_wave {lam}_0"][:, 0]
        G = density_propagator(k, U)
        dev = float(np.abs(n_ref - (0.5 + 0.4 * G)).max())
        p_plus = -G_DEPH + np.sqrt(G_DEPH**2 - 16 * J**2 * np.sin(k / 2) ** 2)
        dk2 = 2 * J**2 / G_DEPH * k**2
        late = (U > 10.0 / G_DEPH) & (U < 3.0 / dk2) & (np.abs(n_ref - 0.5) > 1e-4)  # slow-mode regime only
        rate = float(-np.polyfit(U[late], np.log(np.abs(n_ref[late] - 0.5)), 1)[0]) if late.sum() > 3 else float("nan")
        out[f"lambda={lam}"] = {"max |crest density - (1/2 + 0.4 G_k)|": dev, "-p_+ (exact slow pole)": float(-p_plus),
                                "fitted late-time decay rate": rate, "D k^2 (diffusive estimate)": float(2 * J**2 / G_DEPH * k**2)}
    save_json(RESULTS / "e13c_kspace_check.json", out)
    for key, v in out.items():
        print(key, v)
