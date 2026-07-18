from __future__ import annotations

import numpy as np
from typing import cast

from ..utils.math import log_sum_exp


class HMM:
    """Discrete hidden Markov model."""

    def __init__(self, n_states, n_obs, random_state=None):
        self.n_states = n_states
        self.n_obs = n_obs
        self.random_state = random_state
        self.start_ = None
        self.trans_ = None
        self.emit_ = None
        self.loglik_ = []

    def _init_params(self):
        rng = np.random.default_rng(self.random_state)
        self.start_ = rng.random(self.n_states)
        self.start_ /= self.start_.sum()
        self.trans_ = rng.random((self.n_states, self.n_states))
        self.trans_ /= self.trans_.sum(axis=1, keepdims=True)
        self.emit_ = rng.random((self.n_states, self.n_obs))
        self.emit_ /= self.emit_.sum(axis=1, keepdims=True)

    def _check_seq(self, seq):
        seq = np.asarray(seq, dtype=int)
        if seq.ndim != 1 or seq.size == 0:
            raise ValueError("Expected a non-empty 1D observation sequence")
        if np.any((seq < 0) | (seq >= self.n_obs)):
            raise ValueError("Observation index out of range")
        return seq

    def _forward(self, seq):
        start = cast(np.ndarray, self.start_)
        trans = cast(np.ndarray, self.trans_)
        emit = cast(np.ndarray, self.emit_)
        t = seq.size
        log_start = np.log(start + 1e-12)
        log_trans = np.log(trans + 1e-12)
        log_emit = np.log(emit + 1e-12)

        a = np.empty((t, self.n_states))
        a[0] = log_start + log_emit[:, seq[0]]
        for i in range(1, t):
            a[i] = log_emit[:, seq[i]] + log_sum_exp(a[i - 1][:, None] + log_trans, axis=0)
        return a

    def _backward(self, seq):
        trans = cast(np.ndarray, self.trans_)
        emit = cast(np.ndarray, self.emit_)
        t = seq.size
        log_trans = np.log(trans + 1e-12)
        log_emit = np.log(emit + 1e-12)
        b = np.empty((t, self.n_states))
        b[-1] = 0.0
        for i in range(t - 2, -1, -1):
            b[i] = log_sum_exp(log_trans + log_emit[:, seq[i + 1]] + b[i + 1], axis=1)
        return b

    def fit(self, sequences, n_iter=50, tol=1e-4):
        if self.start_ is None:
            self._init_params()

        seqs = [self._check_seq(seq) for seq in sequences]
        prev = None
        self.loglik_ = []

        for _ in range(n_iter):
            start = np.zeros(self.n_states)
            trans = np.zeros((self.n_states, self.n_states))
            emit = np.zeros((self.n_states, self.n_obs))
            total = 0.0

            for seq in seqs:
                a = self._forward(seq)
                b = self._backward(seq)
                logp = log_sum_exp(a[-1], axis=0)
                total += logp

                gamma = np.exp(a + b - logp)
                start += gamma[0]
                for state in range(self.n_states):
                    np.add.at(emit[state], seq, gamma[:, state])

                for t in range(seq.size - 1):
                    tmp = (
                        a[t][:, None]
                        + np.log(cast(np.ndarray, self.trans_) + 1e-12)
                        + np.log(cast(np.ndarray, self.emit_)[:, seq[t + 1]] + 1e-12)[None, :]
                        + b[t + 1][None, :]
                        - logp
                    )
                    trans += np.exp(tmp)

            self.start_ = start / start.sum()
            self.trans_ = trans / np.maximum(trans.sum(axis=1, keepdims=True), 1e-12)
            self.emit_ = emit / np.maximum(emit.sum(axis=1, keepdims=True), 1e-12)

            self.loglik_.append(total)
            if prev is not None and abs(total - prev) < tol:
                break
            prev = total

        return self

    def score(self, seq):
        seq = self._check_seq(seq)
        a = self._forward(seq)
        return float(log_sum_exp(a[-1], axis=0))

    def predict(self, seq):
        seq = self._check_seq(seq)
        start = cast(np.ndarray, self.start_)
        trans = cast(np.ndarray, self.trans_)
        emit = cast(np.ndarray, self.emit_)
        log_start = np.log(start + 1e-12)
        log_trans = np.log(trans + 1e-12)
        log_emit = np.log(emit + 1e-12)

        t = seq.size
        delta = np.empty((t, self.n_states))
        psi = np.empty((t, self.n_states), dtype=int)

        delta[0] = log_start + log_emit[:, seq[0]]
        psi[0] = 0
        for i in range(1, t):
            scores = delta[i - 1][:, None] + log_trans
            psi[i] = np.argmax(scores, axis=0)
            delta[i] = np.max(scores, axis=0) + log_emit[:, seq[i]]

        path = np.empty(t, dtype=int)
        path[-1] = np.argmax(delta[-1])
        for i in range(t - 2, -1, -1):
            path[i] = psi[i + 1, path[i + 1]]
        return path
