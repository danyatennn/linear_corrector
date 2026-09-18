import _common  # noqa: F401  (sets up sys.path)

import numpy as np

from config import EXP
from recsys_guarantees import plotting


def main():
    n_grid = range(10, 1001, 10)
    p1 = plotting.plot_bound_panels(EXP.deltas, n_grid, e_min=None,
                                    name="exp0_bounds.png")
    p2 = plotting.plot_bound_panels(EXP.deltas, n_grid, e_min=EXP.e_min,
                                    name="exp0_bounds_propensity.png")
    m_grid = [1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]
    p3 = plotting.plot_band_width_vs_m(m_grid, delta=0.9, e_min=EXP.e_min,
                                       name="exp0_bandwidth.png")
    print(f"[exp0] clean DKW bands       -> {p1}")
    print(f"[exp0] propensity overlay R1 -> {p2}")
    print(f"[exp0] band width vs M_-,j   -> {p3}")
    em = EXP.e_min
    vb = np.array([plotting.psi_e(0.9, int(n), em) - plotting.rho_e(0.9, int(n), em)
                   for n in m_grid])
    ho = np.array([plotting.psi_e_hoeffding(0.9, int(n), em)
                   - plotting.rho_e_hoeffding(0.9, int(n), em) for n in m_grid])
    print(f"[exp0] band width at M=1e3: variance-aware {vb[2]:.3f} vs loose {ho[2]:.3f}"
          f"  (sharper by {ho[2] - vb[2]:.3f})")


if __name__ == "__main__":
    main()
