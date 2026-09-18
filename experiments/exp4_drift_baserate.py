import _common

import numpy as np

from config import EXP
from recsys_guarantees.precision import precision_decomposition
from recsys_guarantees.simulate import reshape_base_rate
from recsys_guarantees.conformal import crc_calibrate, empirical_risk
from recsys_guarantees import plotting


def _unwanted_fraction_topk(h, R, lam, k, rng):
    lists = _common.chunk_pseudo_users(h, R, k, rng)
    return float(np.mean([empirical_risk(s, r, lam, k) for s, r in lists]))


def main():
    pools = _common.load_pools()
    h_cal, R_cal = pools["h_cal"], pools["R_cal"]
    h_test, R_test = pools["h_test"], pools["R_test"]
    alpha, k = 0.10, EXP.top_k

    pos_cal, neg_cal = h_cal[R_cal == 1], h_cal[R_cal == 0]
    p_plus = lambda t: float(np.mean(pos_cal > t))
    q_minus = lambda t: float(np.mean(neg_cal > t))
    theta_grid = np.quantile(h_cal, np.linspace(0.0, 0.99, 200))

    def ours_reported(theta, pi):
        return 1.0 - precision_decomposition(p_plus(theta), q_minus(theta), pi)

    def ours_achieved(h, R, theta):
        acc = h > theta
        return float(np.mean(R[acc] == 0)) if acc.any() else 0.0

    def choose_theta(pi):
        best, best_risk = theta_grid[0], 1.0
        for t in theta_grid:
            r = ours_reported(t, pi)
            if r <= alpha:
                return t
            if r < best_risk:
                best, best_risk = t, r
        return best

    rng = np.random.default_rng(EXP.seed)
    lam = crc_calibrate(_common.chunk_pseudo_users(h_cal, R_cal, k, rng), alpha=alpha, k=k)
    pi_before = float(np.mean(R_test))
    theta = choose_theta(pi_before)

    crc_rep_before, crc_ach_before = alpha, _unwanted_fraction_topk(h_test, R_test, lam, k, rng)
    ours_rep_before = ours_reported(theta, pi_before)
    ours_ach_before = ours_achieved(h_test, R_test, theta)

    idx = reshape_base_rate(R_test, target_pi=pi_before / 2, rng=rng)
    h_d, R_d = h_test[idx], R_test[idx]
    pi_after = float(np.mean(R_d))
    crc_rep_after, crc_ach_after = alpha, _unwanted_fraction_topk(h_d, R_d, lam, k, rng)
    ours_rep_after = ours_reported(theta, pi_after)
    ours_ach_after = ours_achieved(h_d, R_d, theta)
    ours_stale = ours_reported(theta, pi_before)

    path = plotting.plot_drift_bars(
        methods=["De Toni CRC", "Ours (R3)"],
        before=[crc_rep_after, ours_rep_after],
        after=[crc_ach_after, ours_ach_after],
        ylabel="Risk (unwanted fraction)",
        before_label="risk the method reports",
        after_label="true achieved risk",
        hline=alpha, hline_label=rf"pre-drift target $\alpha={alpha}$",
        title=rf"After base-rate drift $\pi$: {pi_before:.2f}$\to${pi_after:.2f}  "
              r"(CRC's promise breaks; R3 re-plugs $\pi$ and stays calibrated)",
        name="exp4_drift.png")

    crc_gap_before = abs(crc_ach_before - crc_rep_before)
    crc_gap_after = abs(crc_ach_after - crc_rep_after)
    ours_gap_before = abs(ours_ach_before - ours_rep_before)
    ours_gap_after = abs(ours_ach_after - ours_rep_after)

    print(f"[exp4] saved -> {path}")
    print(f"[exp4] drift pi: {pi_before:.3f} -> {pi_after:.3f}")
    print(f"[exp4] CRC : reports alpha={alpha:.3f} always | achieved {crc_ach_before:.3f}"
          f" -> {crc_ach_after:.3f} | calibration gap {crc_gap_before:.3f} -> {crc_gap_after:.3f}"
          f"  ({'VIOLATED' if crc_ach_after > alpha else 'ok'}: frozen promise can't track pi)")
    print(f"[exp4] Ours: reports {ours_rep_before:.3f} -> {ours_rep_after:.3f} (re-plug pi) |"
          f" achieved {ours_ach_before:.3f} -> {ours_ach_after:.3f} |"
          f" calibration gap {ours_gap_before:.3f} -> {ours_gap_after:.3f}"
          f"  ({'VALID' if ours_gap_after <= max(0.05, 2 * ours_gap_before) else 'CHECK'}:"
          f" reported tracks achieved)")
    print(f"[exp4] (without re-plugging pi, R3 would have reported a stale {ours_stale:.3f}"
          f" vs the true {ours_ach_after:.3f} -- re-plugging pi is what keeps it honest)")


if __name__ == "__main__":
    main()
