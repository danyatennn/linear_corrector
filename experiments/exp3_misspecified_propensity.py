import _common

import numpy as np

from config import EXP
from recsys_guarantees import bounds
from recsys_guarantees.corrector import Algorithm1
from recsys_guarantees.simulate import simulate_observation
from recsys_guarantees.propensity import noisy_propensity
from recsys_guarantees import plotting


def main():
    pools = _common.load_pools()
    h_cal, R_cal = pools["h_cal"], pools["R_cal"]
    h_test, R_test = pools["h_test"], pools["R_test"]
    err_test = h_test[R_test == 0]
    e_min = EXP.e_min

    sigmas = np.linspace(0.0, 0.30, 7)
    n_seeds = 10
    deltas = EXP.deltas
    claimed = {d: bounds.rho_e(d, len(h_cal), e_min) for d in deltas}

    rates = []
    for sigma in sigmas:
        violations, total = 0, 0
        for s in range(n_seeds):
            rng = np.random.default_rng(EXP.seed + s)
            obs = simulate_observation(h_cal, R_cal, e_min=e_min,
                                       strength=EXP.bias_strength, rng=rng)
            C, e_true = obs["C"], obs["e"]
            e_hat = noisy_propensity(e_true, sigma, e_min=e_min, rng=rng)
            w_pos, w_neg = C / e_hat, 1.0 - C / e_hat
            for d in deltas:
                corr = Algorithm1(h_cal, h_cal, d, pos_weights=w_pos, neg_weights=w_neg,
                                  e_min=e_min, isotonic=True)
                achieved = float(np.mean(err_test <= corr.threshold))
                violations += int(achieved < claimed[d])
                total += 1
        rates.append(violations / total)

    path = plotting.plot_violation_vs_sigma(
        sigmas, np.array(rates),
        title="Graceful degradation under mis-specified propensity",
        name="exp3_misspecified.png")
    print(f"[exp3] saved -> {path}")
    print(f"[exp3] violation rate at sigma=0.0: {rates[0]:.2f}; "
          f"at sigma={sigmas[-1]:.2f}: {rates[-1]:.2f}")


if __name__ == "__main__":
    main()
