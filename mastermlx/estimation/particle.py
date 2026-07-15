from __future__ import annotations

import numpy as np

from ..config import get_backend

try:
    from ._particle_ops import normalize_weights as _cy_normalize_weights
    from ._particle_ops import systematic_resample as _cy_systematic_resample
except ImportError:  # pragma: no cover - fallback when Cython extensions are unavailable
    _cy_normalize_weights = None
    _cy_systematic_resample = None


def systematic_resample(weights, rng=None):
    if get_backend() != "numpy" and _cy_systematic_resample is not None:
        return _cy_systematic_resample(weights, rng=rng)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size == 0:
        raise ValueError("weights cannot be empty")
    total = np.sum(weights)
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    weights = weights / total
    rng = np.random.default_rng() if rng is None else rng

    positions = (rng.random() + np.arange(weights.size)) / weights.size
    cumulative = np.cumsum(weights)
    return np.searchsorted(cumulative, positions, side="left").astype(np.intp, copy=False)


class ParticleFilter:
    """Bootstrap particle filter for nonlinear state estimation."""

    def __init__(self, particles, weights=None, transition=None, likelihood=None, rng=None):
        self.particles_ = np.asarray(particles, dtype=float)
        if self.particles_.ndim != 2:
            raise ValueError("particles must have shape (n_particles, state_dim)")
        self.n_particles_, self.state_dim_ = self.particles_.shape
        if weights is None:
            self.weights_ = np.full(self.n_particles_, 1.0 / self.n_particles_, dtype=float)
        else:
            self.weights_ = np.asarray(weights, dtype=float).reshape(-1)
            if self.weights_.size != self.n_particles_:
                raise ValueError("weights must match the number of particles")
            total = float(np.sum(self.weights_))
            if total <= 0:
                self.weights_ = np.full(self.n_particles_, 1.0 / self.n_particles_, dtype=float)
            elif get_backend() != "numpy" and _cy_normalize_weights is not None:
                self.weights_ = _cy_normalize_weights(self.weights_)
            else:
                self.weights_ = self.weights_ / total
        self.transition_ = transition
        self.likelihood_ = likelihood
        self.rng_ = np.random.default_rng() if rng is None else rng

    @property
    def estimate(self):
        return np.sum(self.particles_ * self.weights_[:, None], axis=0)

    @property
    def ess(self):
        return 1.0 / np.sum(self.weights_ ** 2)

    def predict(self, control=None, noise=None):
        if self.transition_ is None:
            raise ValueError("transition function is required for prediction")
        updated = []
        for particle in self.particles_:
            next_state = self.transition_(particle, control)
            if noise is not None:
                next_state = np.asarray(next_state, dtype=float) + np.asarray(noise(self.rng_), dtype=float)
            updated.append(next_state)
        self.particles_ = np.asarray(updated, dtype=float)
        return self.particles_

    def update(self, measurement):
        if self.likelihood_ is None:
            raise ValueError("likelihood function is required for update")
        weights = np.array([self.likelihood_(particle, measurement) for particle in self.particles_], dtype=float)
        weights = np.maximum(weights, 0.0)
        total = float(np.sum(weights))
        if total <= 0:
            self.weights_ = np.full(self.n_particles_, 1.0 / self.n_particles_, dtype=float)
        elif get_backend() != "numpy" and _cy_normalize_weights is not None:
            self.weights_ = _cy_normalize_weights(weights)
        else:
            self.weights_ = weights / total
        return self.weights_

    def resample(self):
        idx = systematic_resample(self.weights_, rng=self.rng_)
        self.particles_ = self.particles_[idx]
        self.weights_ = np.full(self.n_particles_, 1.0 / self.n_particles_, dtype=float)
        return self.particles_

    def step(self, measurement, control=None, noise=None, resample_threshold=None):
        self.predict(control=control, noise=noise)
        self.update(measurement)
        threshold = self.n_particles_ / 2.0 if resample_threshold is None else float(resample_threshold)
        if self.ess < threshold:
            self.resample()
        return self.estimate
