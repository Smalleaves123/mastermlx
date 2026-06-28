from __future__ import annotations

import numpy as np


class VariationalEstimator:
    """Common helpers for variational inference models."""

    def __init__(self):
        self.lower_bound_ = []
        self.n_iter_ = 0
        self.converged_ = False

    def elbo_trace(self):
        return np.asarray(self.lower_bound_, dtype=float).copy()

    def final_elbo(self):
        if not self.lower_bound_:
            raise RuntimeError("Model has not been fit yet")
        return float(self.lower_bound_[-1])

    def posterior_summary(self):
        if not self.lower_bound_:
            raise RuntimeError("Model has not been fit yet")
        summary = {
            "model": self.__class__.__name__,
            "n_iter": int(self.n_iter_),
            "converged": bool(self.converged_),
            "final_elbo": self.final_elbo(),
        }
        summary.update(self._posterior_summary())
        return summary

    def sample_posterior_weights(self, n_samples=1, random_state=None):
        mean = getattr(self, "posterior_mean_", None)
        cov = getattr(self, "posterior_cov_", None)
        if mean is None or cov is None:
            raise RuntimeError("Model has not been fit yet")

        rng = np.random.default_rng(random_state)
        samples = rng.multivariate_normal(np.asarray(mean, dtype=float), np.asarray(cov, dtype=float), size=int(n_samples))
        return samples[0] if int(n_samples) == 1 else samples

    def _posterior_summary(self):
        return {}


VarEst = VariationalEstimator
