import numpy as np

_EPS_GRID = np.linspace(1e-4, 1.0, 400)


def rho(a, n, eps_grid=_EPS_GRID):
    vals = np.maximum(0.0, a - eps_grid) * (1.0 - 2.0 * np.exp(-2.0 * n * eps_grid ** 2))
    return float(np.clip(np.max(vals), 0.0, 1.0))


def psi(a, n, eps_grid=_EPS_GRID):
    vals = np.minimum(1.0, 2.0 * np.exp(-2.0 * n * eps_grid ** 2) + a + eps_grid)
    return float(np.clip(np.min(vals), 0.0, 1.0))


def variance_terms(e):
    e = np.asarray(e, dtype=float)
    nu = float(np.mean((1.0 - e) / e))
    B = float(np.max(np.maximum(1.0, 1.0 / e - 1.0)))
    return nu, B


def _resolve_nu_B(e_min, nu, B):
    if nu is not None and B is not None:
        return float(nu), float(B)
    if e_min is None:
        raise ValueError("provide either (nu, B) or e_min")
    return 1.0 / e_min - 1.0, 1.0 / e_min


def delta_e_hoeffding(eps, n, e_min, n_split=300):
    eps1 = np.linspace(1e-6, eps, n_split)
    eps2 = eps - eps1
    vals = (2.0 * (n + 1) * np.exp(-2.0 * n * e_min ** 2 * eps1 ** 2)
            + 2.0 * np.exp(-2.0 * n * eps2 ** 2))
    return float(np.min(vals))


def freedman_fr(t, v, B):
    t = np.asarray(t, dtype=float)
    if v <= 0.0:
        return np.where(t > 0.0, 0.0, 1.0)
    u = B * t / v
    return np.exp(-(v / B ** 2) * ((1.0 + u) * np.log1p(u) - u))


def delta_vb(eps, n, e_min=None, nu=None, B=None, n_split=300):
    nu, B = _resolve_nu_B(e_min, nu, B)
    alpha = np.linspace(1e-6, 1.0 - 1e-6, n_split)
    e_sel = alpha * eps
    e_samp = (1.0 - alpha) * eps
    sel = 2.0 * freedman_fr(n * e_sel, n * nu, B)
    samp = 2.0 * np.exp(-2.0 * n * e_samp ** 2)
    return float(min(1.0, np.min(sel + samp)))


def rho_e(a, n, e_min=None, nu=None, B=None, eps_grid=_EPS_GRID):
    vals = [max(0.0, a - e) * (1.0 - delta_vb(e, n, e_min, nu, B)) for e in eps_grid]
    return float(np.clip(np.max(vals), 0.0, 1.0))


def psi_e(a, n, e_min=None, nu=None, B=None, eps_grid=_EPS_GRID):
    vals = [min(1.0, delta_vb(e, n, e_min, nu, B) + a + e) for e in eps_grid]
    return float(np.clip(np.min(vals), 0.0, 1.0))


def rho_e_hoeffding(a, n, e_min, eps_grid=_EPS_GRID):
    vals = [max(0.0, a - e) * (1.0 - delta_e_hoeffding(e, n, e_min)) for e in eps_grid]
    return float(np.clip(np.max(vals), 0.0, 1.0))


def psi_e_hoeffding(a, n, e_min, eps_grid=_EPS_GRID):
    vals = [min(1.0, delta_e_hoeffding(e, n, e_min) + a + e) for e in eps_grid]
    return float(np.clip(np.min(vals), 0.0, 1.0))
