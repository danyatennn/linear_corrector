import _common

import numpy as np

from config import EXP
from recsys_guarantees import bounds
from recsys_guarantees.corrector import Algorithm1
from recsys_guarantees.simulate import simulate_observation
from recsys_guarantees import plotting
from recsys_guarantees.conformal import crc_calibrate, empirical_risk


def _achieved_rejection(h_test, R_test, theta):
    err = h_test[R_test == 0]
    return float(np.mean(err <= theta)) if len(err) else float("nan")


def main():
    pools = _common.load_pools()
    h_cal, R_cal = pools["h_cal"], pools["R_cal"]
    h_test, R_test = pools["h_test"], pools["R_test"]

    deltas = np.linspace(0.50, 0.97, 12)
    n_seeds = 5
    e_min, strength = EXP.e_min, EXP.bias_strength

    claimed = np.array([bounds.rho(d, int((R_cal == 0).sum())) for d in deltas])
    claimed_e = np.array([bounds.rho_e(d, len(h_cal), e_min) for d in deltas])

    naive_ach = np.zeros((n_seeds, len(deltas)))
    prop_ach = np.zeros((n_seeds, len(deltas)))
    for s in range(n_seeds):
        rng = np.random.default_rng(EXP.seed + s)
        obs = simulate_observation(h_cal, R_cal, e_min=e_min, strength=strength, rng=rng)
        C, e = obs["C"], obs["e"]
        clicked, unclicked = C == 1, C == 0
        w_pos, w_neg = C / e, 1.0 - C / e
        for j, d in enumerate(deltas):
            naive = Algorithm1(h_cal[clicked], h_cal[unclicked], d)
            prop = Algorithm1(h_cal, h_cal, d, pos_weights=w_pos, neg_weights=w_neg,
                              e_min=e_min, isotonic=True)
            naive_ach[s, j] = _achieved_rejection(h_test, R_test, naive.threshold)
            prop_ach[s, j] = _achieved_rejection(h_test, R_test, prop.threshold)

    naive_mean, prop_mean = naive_ach.mean(0), prop_ach.mean(0)

    path = plotting.plot_claimed_vs_achieved(
        deltas, claimed,
        achieved={
            "naive corrector (clicks as labels)": (naive_mean, False),
            "propensity corrector (R1/R2)": (prop_mean, True),
        },
        xlabel=r"target $\Delta$",
        ylabel=r"true $P(\mathrm{reject}\mid \mathrm{error})$",
        title="Decisive experiment: naive vs propensity rejection guarantee (MNAR)",
        name="exp1_decisive.png")

    n_violations = int(np.sum(naive_mean < claimed))
    worst_gap = float(np.max(claimed - naive_mean))
    prop_margin = float(np.min(prop_mean - claimed_e))
    print(f"[exp1] saved -> {path}")
    print(f"[exp1] naive: {n_violations}/{len(deltas)} Delta points violate the clean "
          f"DKW bound (worst deficit {worst_gap:.3f})")
    print(f"[exp1] propensity: min margin over its rho_e bound = {prop_margin:+.3f} "
          f"(>=0 means the guarantee holds everywhere)")
    rng = np.random.default_rng(EXP.seed)
    obs_cal = simulate_observation(h_cal, R_cal, e_min=e_min, strength=strength, rng=rng)
    alpha = 0.10
    cal_lists = _common.chunk_pseudo_users(h_cal, obs_cal["C"], EXP.top_k, rng)
    lam = crc_calibrate(cal_lists, alpha=alpha, k=EXP.top_k)
    test_lists = _common.chunk_pseudo_users(h_test, R_test, EXP.top_k, rng)
    true_risk = float(np.mean([empirical_risk(s, r, lam, EXP.top_k) for s, r in test_lists]))
    print(f"[exp1] CRC (alpha={alpha}): true test risk = {true_risk:.3f} "
          f"({'VALID (conservative)' if true_risk <= alpha else 'VIOLATED'}) "
          f"-- see Exp 4 for the drift regime where CRC breaks")


if __name__ == "__main__":
    main()
