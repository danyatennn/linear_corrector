import _common

import numpy as np

from config import EXP
from recsys_guarantees import bounds
from recsys_guarantees.precision import precision_decomposition
from recsys_guarantees.simulate import simulate_observation
from recsys_guarantees import plotting


def main():
    pools = _common.load_pools()
    h, R = pools["h_test"], pools["R_test"]
    n = len(h)

    rng = np.random.default_rng(EXP.seed)
    obs = simulate_observation(h, R, e_min=EXP.e_min, strength=EXP.bias_strength, rng=rng)
    C, e = obs["C"], obs["e"]
    w_pos = C / e

    thresholds = np.quantile(h, np.linspace(0.0, 0.98, 40))
    cov, prec_true, prec_naive, prec_ips, prec_lo = [], [], [], [], []
    for t in thresholds:
        acc = h > t
        na = int(acc.sum())
        if na == 0:
            continue
        cov.append(na / n)
        prec_true.append(float(np.mean(R[acc])))
        prec_naive.append(float(np.mean(C[acc])))
        prec_ips.append(float(np.mean(w_pos[acc])))
        p_plus = w_pos[acc].sum() / max(w_pos.sum(), 1e-9)
        w_neg = 1.0 - w_pos
        q_minus = w_neg[acc].sum() / max(w_neg.sum(), 1e-9)
        pi = w_pos.sum() / n
        p_plus_lo = bounds.rho_e(np.clip(p_plus, 0, 1), n, EXP.e_min)
        q_minus_hi = bounds.psi_e(np.clip(q_minus, 0, 1), n, EXP.e_min)
        prec_lo.append(precision_decomposition(p_plus_lo, q_minus_hi, pi))

    path = plotting.plot_risk_coverage(
        curves={
            "oracle (true R)": (cov, prec_true, "k-"),
            "naive (observed clicks)": (cov, prec_naive, "--"),
            "propensity IPS (C/e)  [R2]": (cov, prec_ips, "-"),
            "precision lower bound  [R3]": (cov, prec_lo, ":"),
        },
        title="Risk-coverage: propensity IPS recovers true precision under MNAR",
        name="exp2_risk_coverage.png")

    cov = np.array(cov)
    mask = cov < 0.5
    naive_gap = float(np.mean(np.array(prec_true)[mask] - np.array(prec_naive)[mask]))
    ips_gap = float(np.mean(np.abs(np.array(prec_true)[mask] - np.array(prec_ips)[mask])))
    print(f"[exp2] saved -> {path}")
    print(f"[exp2] mean precision under-estimate by naive (cov<0.5): {naive_gap:+.3f}")
    print(f"[exp2] mean |IPS - oracle| precision gap (cov<0.5): {ips_gap:.3f} "
          f"(small => IPS recovers the true curve)")


if __name__ == "__main__":
    main()
