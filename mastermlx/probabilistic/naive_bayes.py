from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils import accuracy, as_2d, check_1d_array, check_2d_array


class _BaseNB(BaseEstimator):
    def _check_fitted(self):
        if self.classes_ is None:
            raise RuntimeError("Model has not been fit yet")

    def _predict_from_log_joint(self, logp):
        idx = np.argmax(logp, axis=1)
        pred = self.classes_[idx]
        return pred[0] if pred.shape[0] == 1 else pred

    def _predict_proba_from_log_joint(self, logp):
        logp = logp - np.max(logp, axis=1, keepdims=True)
        p = np.exp(logp)
        p = p / np.sum(p, axis=1, keepdims=True)
        return p[0] if p.shape[0] == 1 else p


class GaussianNB(_BaseNB):
    """Gaussian naive Bayes classifier."""

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.mean_ = None
        self.var_ = None
        self.prior_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        self.classes_ = np.unique(y)
        n_class = self.classes_.shape[0]
        n_feat = X.shape[1]
        self.mean_ = np.zeros((n_class, n_feat))
        self.var_ = np.zeros((n_class, n_feat))
        self.prior_ = np.zeros(n_class)

        for i, c in enumerate(self.classes_):
            x = X[y == c]
            self.mean_[i] = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            self.var_[i] = var + self.var_smoothing
            self.prior_[i] = x.shape[0] / X.shape[0]

        return self

    def _joint_log_prob(self, X):
        eps = 1e-12
        out = []
        for i in range(self.classes_.shape[0]):
            mean = self.mean_[i]
            var = self.var_[i]
            log_prior = np.log(self.prior_[i] + eps)
            log_det = -0.5 * np.sum(np.log(2.0 * np.pi * var))
            quad = -0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            out.append(log_prior + log_det + quad)
        return np.column_stack(out)

    def predict(self, X):
        self._check_fitted()
        X = as_2d(X)
        if X.shape[1] != self.mean_.shape[1]:
            raise ValueError("X has a different number of features than the fitted data")
        logp = self._joint_log_prob(X)
        return self._predict_from_log_joint(logp)

    def predict_proba(self, X):
        self._check_fitted()
        X = as_2d(X)
        logp = self._joint_log_prob(X)
        return self._predict_proba_from_log_joint(logp)

    def score(self, X, y):
        return accuracy(y, self.predict(X))


class BernoulliNB(_BaseNB):
    """Bernoulli naive Bayes classifier for binary/discrete features."""

    def __init__(self, alpha=1.0, binarize=0.0):
        self.alpha = float(alpha)
        self.binarize = binarize
        self.classes_ = None
        self.class_count_ = None
        self.feature_count_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.feature_log_prob_neg_ = None

    def _bin(self, X):
        X = as_2d(X).astype(float, copy=False)
        if self.binarize is None:
            return X
        return (X > float(self.binarize)).astype(float)

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        X = self._bin(X)
        if np.any((X != 0.0) & (X != 1.0)):
            raise ValueError("BernoulliNB expects binary features after binarization")

        self.classes_ = np.unique(y)
        n_class = self.classes_.shape[0]
        n_feat = X.shape[1]
        self.class_count_ = np.zeros(n_class, dtype=float)
        self.feature_count_ = np.zeros((n_class, n_feat), dtype=float)

        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.class_count_[i] = Xc.shape[0]
            self.feature_count_[i] = np.sum(Xc, axis=0)

        alpha = float(self.alpha)
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = self.class_count_[:, None] + 2.0 * alpha
        self.feature_log_prob_ = np.log(smoothed_fc / smoothed_cc)
        self.feature_log_prob_neg_ = np.log((self.class_count_[:, None] - self.feature_count_ + alpha) / smoothed_cc)
        self.class_log_prior_ = np.log(self.class_count_ / np.sum(self.class_count_))
        return self

    def _joint_log_prob(self, X):
        X = self._bin(X)
        if X.shape[1] != self.feature_log_prob_.shape[1]:
            raise ValueError("X has a different number of features than the fitted data")
        return (
            self.class_log_prior_[None, :]
            + X @ self.feature_log_prob_.T
            + (1.0 - X) @ self.feature_log_prob_neg_.T
        )

    def predict(self, X):
        self._check_fitted()
        return self._predict_from_log_joint(self._joint_log_prob(X))

    def predict_proba(self, X):
        self._check_fitted()
        return self._predict_proba_from_log_joint(self._joint_log_prob(X))

    def score(self, X, y):
        return accuracy(y, self.predict(X))


class MultinomialNB(_BaseNB):
    """Multinomial naive Bayes classifier for count-like features."""

    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.classes_ = None
        self.class_count_ = None
        self.feature_count_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if np.any(X < 0):
            raise ValueError("MultinomialNB expects non-negative features")
        X = X.astype(float, copy=False)

        self.classes_ = np.unique(y)
        n_class = self.classes_.shape[0]
        n_feat = X.shape[1]
        self.class_count_ = np.zeros(n_class, dtype=float)
        self.feature_count_ = np.zeros((n_class, n_feat), dtype=float)

        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.class_count_[i] = Xc.shape[0]
            self.feature_count_[i] = np.sum(Xc, axis=0)

        alpha = float(self.alpha)
        smoothed_fc = self.feature_count_ + alpha
        smoothed_totals = smoothed_fc.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_fc / smoothed_totals)
        self.class_log_prior_ = np.log(self.class_count_ / np.sum(self.class_count_))
        return self

    def _joint_log_prob(self, X):
        X = as_2d(X).astype(float, copy=False)
        if X.shape[1] != self.feature_log_prob_.shape[1]:
            raise ValueError("X has a different number of features than the fitted data")
        if np.any(X < 0):
            raise ValueError("MultinomialNB expects non-negative features")
        return self.class_log_prior_[None, :] + X @ self.feature_log_prob_.T

    def predict(self, X):
        self._check_fitted()
        return self._predict_from_log_joint(self._joint_log_prob(X))

    def predict_proba(self, X):
        self._check_fitted()
        return self._predict_proba_from_log_joint(self._joint_log_prob(X))

    def score(self, X, y):
        return accuracy(y, self.predict(X))
