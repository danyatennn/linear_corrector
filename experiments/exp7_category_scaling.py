import _common

import numpy as np

from recsys_guarantees.corrector import Algorithm1
from recsys_guarantees import plotting


def main():
    pools = _common.load_pools()
    h_cal, R_cal = pools["h_cal"], pools["R_cal"]
    h_test, R_test = pools["h_test"], pools["R_test"]
    delta, n_buckets = 0.90, 4

    edges = np.quantile(h_cal, np.linspace(0.0, 1.0, n_buckets + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    cal_b = np.digitize(h_cal, edges[1:-1])
    test_b = np.digitize(h_test, edges[1:-1])

    g = Algorithm1(h_cal[R_cal == 1], h_cal[R_cal == 0], delta)

    labels, global_ach, bucket_ach = [], [], []
    for b in range(n_buckets):
        err = h_test[(test_b == b) & (R_test == 0)]
        pos_b = h_cal[(cal_b == b) & (R_cal == 1)]
        neg_b = h_cal[(cal_b == b) & (R_cal == 0)]
        if len(err) == 0 or len(neg_b) < 10 or len(pos_b) < 1:
            continue
        local = Algorithm1(pos_b, neg_b, delta)
        labels.append(f"Q{b + 1}")
        global_ach.append(float(np.mean(err <= g.threshold)))
        bucket_ach.append(float(np.mean(err <= local.threshold)))

    path = plotting.plot_drift_bars(
        methods=labels,
        before=global_ach,
        after=bucket_ach,
        ylabel=r"achieved $P(\mathrm{reject}\mid\mathrm{error})$ in bucket",
        hline=delta, hline_label=rf"target $\Delta={delta}$",
        before_label="global threshold", after_label="per-bucket threshold",
        title="Conditional guarantee: global threshold vs per-bucket threshold",
        name="exp7_category.png")

    g_dev = float(np.mean(np.abs(np.array(global_ach) - delta)))
    b_dev = float(np.mean(np.abs(np.array(bucket_ach) - delta)))
    print(f"[exp7] saved -> {path}")
    print(f"[exp7] mean |achieved - Delta|:  global={g_dev:.3f}  per-bucket={b_dev:.3f} "
          f"(smaller per-bucket => tighter conditional guarantee)")


if __name__ == "__main__":
    main()
