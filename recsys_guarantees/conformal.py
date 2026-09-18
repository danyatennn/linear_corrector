import numpy as np


def empirical_risk(scores, relevant, lam, k):
    """Fraction of unwanted items in the top-k of the score-filtered set."""
    scores = np.asarray(scores)
    relevant = np.asarray(relevant).astype(bool)
    keep = scores > lam
    if not np.any(keep):
        return 0.0
    kept_scores = scores[keep]
    kept_rel = relevant[keep]
    top = np.argsort(-kept_scores)[:k]
    chosen_rel = kept_rel[top]
    return float(np.mean(~chosen_rel))


def crc_calibrate(cal_users, alpha, k, n_grid=200):
    all_scores = np.concatenate([np.asarray(s) for s, _ in cal_users])
    grid = np.quantile(all_scores, np.linspace(0.0, 1.0, n_grid))
    n = len(cal_users)
    for lam in grid:
        rhat = np.mean([empirical_risk(s, r, lam, k) for s, r in cal_users])
        if (n / (n + 1)) * rhat + 1.0 / (n + 1) <= alpha:
            return float(lam)
    return float(grid[-1])
