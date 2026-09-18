import numpy as np

from . import bounds
from .cdf import EmpiricalCDF


class _BaseCorrector:
    def __init__(self, pos_scores, neg_scores, delta, pos_weights=None,
                 neg_weights=None, e_min=None, isotonic=False):
        self.delta = delta
        self.e_min = e_min
        F_pos = EmpiricalCDF(pos_scores, pos_weights)
        F_neg = EmpiricalCDF(neg_scores, neg_weights)
        self.F_pos = F_pos.isotonic() if isotonic else F_pos
        self.F_neg = F_neg.isotonic() if isotonic else F_neg
        self.M_pos = len(np.asarray(pos_scores))
        self.M_neg = len(np.asarray(neg_scores))
        self.threshold = self._compute_threshold()

    def _compute_threshold(self):
        raise NotImplementedError

    def predict(self, h):
        return "accept" if h > self.threshold else "reject"

    def accept_mask(self, h):
        return np.asarray(h) > self.threshold

    def _band(self, a, n):
        if self.e_min is None:
            return bounds.rho(a, n), bounds.psi(a, n)
        return bounds.rho_e(a, n, self.e_min), bounds.psi_e(a, n, self.e_min)


class Algorithm1(_BaseCorrector):

    def _compute_threshold(self):
        return self.F_neg.inverse(self.delta)

    def rejection_bounds(self):
        return self._band(self.delta, self.M_neg)

    def acceptance_bounds(self):
        a = self.F_pos.cdf(self.threshold)
        lo, hi = self._band(a, self.M_pos)
        return 1 - hi, 1 - lo


class Algorithm2(_BaseCorrector):

    def _compute_threshold(self):
        return self.F_pos.inverse(self.delta)

    def acceptance_bounds(self):
        lo, hi = self._band(self.delta, self.M_pos)
        return 1 - hi, 1 - lo
