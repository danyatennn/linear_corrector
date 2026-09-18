import numpy as np
from sklearn.isotonic import IsotonicRegression


class EmpiricalCDF:
    def __init__(self, samples, weights=None):
        x = np.asarray(samples, dtype=float)
        order = np.argsort(x)
        self.x = x[order]
        if weights is None:
            weights = np.ones_like(self.x)
        self.w = np.asarray(weights, dtype=float)[order]
        total = self.w.sum()
        self._cum = np.cumsum(self.w) / total if total != 0 else np.cumsum(self.w)
        self.n = len(self.x)

    def cdf(self, s):
        idx = np.searchsorted(self.x, s, side="right") - 1
        if np.isscalar(s):
            return 0.0 if idx < 0 else float(np.clip(self._cum[idx], 0.0, 1.0))
        out = np.where(idx < 0, 0.0, self._cum[np.clip(idx, 0, self.n - 1)])
        return np.clip(out, 0.0, 1.0)

    def inverse(self, q):
        idx = np.searchsorted(self._cum, q, side="left")
        return float(self.x[min(idx, self.n - 1)])

    def isotonic(self):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True)
        fitted = iso.fit_transform(self.x, np.clip(self._cum, 0.0, 1.0))
        new = EmpiricalCDF.__new__(EmpiricalCDF)
        new.x, new.w, new._cum, new.n = self.x, self.w, fitted, self.n
        return new
