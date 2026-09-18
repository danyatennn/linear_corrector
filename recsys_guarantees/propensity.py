import numpy as np


def score_propensity(h, e_min=0.1, strength=4.0):
    h = np.asarray(h, dtype=float)
    z = (h - np.median(h)) / (np.std(h) + 1e-9)
    e = 1.0 / (1.0 + np.exp(-strength * z))
    return np.clip(e_min + (1.0 - e_min) * e, e_min, 1.0)


def noisy_propensity(e_true, sigma, e_min=0.1, rng=None):
    rng = rng or np.random.default_rng()
    noisy = e_true + rng.normal(0.0, sigma, size=len(e_true))
    return np.clip(noisy, e_min, 1.0)
