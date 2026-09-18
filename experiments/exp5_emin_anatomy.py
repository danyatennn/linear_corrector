import _common

import numpy as np

from config import EXP
from recsys_guarantees import bounds


def main():
    pools = _common.load_pools()
    n_pool = len(pools["h_cal"])

    emins = np.linspace(0.01, 0.30, 30)
    delta = 0.90
    widths = {}
    for n in [200, 1000, n_pool]:
        w = [bounds.psi_e(delta, n, em) - bounds.rho_e(delta, n, em) for em in emins]
        widths[rf"$n={n}$"] = np.array(w)

    from recsys_guarantees import plotting
    path = plotting.plot_bound_width_vs_emin(
        emins, widths,
        title=rf"Bound width vs $e_{{\min}}$  ($\Delta={delta}$);  "
              rf"variance-aware $n_{{eff}}\sim n\,e_{{\min}}$",
        name="exp5_emin.png")

    w_pool = widths[rf"$n={n_pool}$"]
    print(f"[exp5] saved -> {path}")
    print(f"[exp5] at e_min=0.01 width={w_pool[0]:.3f} (widest); "
          f"at e_min=0.30 width={w_pool[-1]:.3f} "
          f"(variance-aware tail: linear, not quadratic, in e_min)")


if __name__ == "__main__":
    main()
