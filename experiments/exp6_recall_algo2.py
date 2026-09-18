import _common

import numpy as np

from config import EXP
from recsys_guarantees.corrector import Algorithm2
from recsys_guarantees import plotting


def main():
    pools = _common.load_pools()
    h_test, R_test = pools["h_test"], pools["R_test"]

    rng = np.random.default_rng(EXP.seed)
    perm = rng.permutation(len(h_test))
    half = len(perm) // 2
    cal_idx, ev_idx = perm[:half], perm[half:]
    h_cal, R_cal = h_test[cal_idx], R_test[cal_idx]
    h_ev, R_ev = h_test[ev_idx], R_test[ev_idx]
    good_ev = h_ev[R_ev == 1]

    deltas = np.linspace(0.05, 0.40, 10)
    claimed, achieved = [], []
    for d in deltas:
        corr = Algorithm2(h_cal[R_cal == 1], h_cal[R_cal == 0], d)
        lo, _hi = corr.acceptance_bounds()
        claimed.append(lo)
        achieved.append(float(np.mean(good_ev > corr.threshold)))

    claimed, achieved = np.array(claimed), np.array(achieved)
    path = plotting.plot_claimed_vs_achieved(
        deltas, claimed,
        achieved={"Algorithm 2 acceptance (true labels)": (achieved, True)},
        xlabel=r"tolerated good-item miss rate $\Delta$",
        ylabel=r"$P(\mathrm{accept}\mid \mathrm{correct})$",
        title="Recall-side Algorithm 2: acceptance guarantee holds",
        name="exp6_recall_algo2.png")

    n_violations = int(np.sum(achieved < claimed))
    print(f"[exp6] saved -> {path}")
    print(f"[exp6] acceptance below the Type-II lower bound at "
          f"{n_violations}/{len(deltas)} Delta points (0 => guarantee holds)")


if __name__ == "__main__":
    main()
