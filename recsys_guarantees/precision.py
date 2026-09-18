import numpy as np


def precision_decomposition(p_plus, q_minus, pi):
    num = p_plus * pi
    den = p_plus * pi + q_minus * (1 - pi)
    return num / den if den > 0 else 0.0


def risk_coverage_curve(scores, relevant, weights=None, n_points=50):
    scores = np.asarray(scores)
    relevant = np.asarray(relevant).astype(float)
    w = np.ones_like(scores, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    thresholds = np.quantile(scores, np.linspace(0.0, 1.0, n_points))
    cov, prec = [], []
    total = w.sum()
    for t in thresholds:
        acc = scores > t
        wa = w[acc].sum()
        cov.append(wa / total if total else 0.0)
        prec.append((w[acc] * relevant[acc]).sum() / wa if wa else 1.0)
    return np.array(cov), np.array(prec)


def violation_rate(achieved, claimed_lower_bound):
    achieved = np.asarray(achieved)
    return float(np.mean(achieved < claimed_lower_bound))
