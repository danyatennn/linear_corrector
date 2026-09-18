import numpy as np

from .propensity import score_propensity


def simulate_observation(h, R, e_min=0.1, strength=4.0, rng=None):
    rng = rng or np.random.default_rng()
    h = np.asarray(h, dtype=float)
    R = np.asarray(R, dtype=float)
    e = score_propensity(h, e_min=e_min, strength=strength)
    E = (rng.random(len(h)) < e).astype(float)
    C = E * R
    return dict(h=h, R=R, e=e, E=E, C=C)


def reshape_base_rate(R, target_pi, rng=None):
    rng = rng or np.random.default_rng()
    R = np.asarray(R)
    rel_idx = np.where(R == 1)[0]
    irr_idx = np.where(R == 0)[0]
    n_rel, n_irr = len(rel_idx), len(irr_idx)
    n_irr_target = int(round(n_rel * (1 - target_pi) / max(target_pi, 1e-9)))
    if n_irr_target <= n_irr:
        keep_irr = rng.choice(irr_idx, n_irr_target, replace=False)
        return np.sort(np.concatenate([rel_idx, keep_irr]))
    n_rel_target = int(round(n_irr * target_pi / max(1 - target_pi, 1e-9)))
    keep_rel = rng.choice(rel_idx, min(n_rel_target, n_rel), replace=False)
    return np.sort(np.concatenate([keep_rel, irr_idx]))
