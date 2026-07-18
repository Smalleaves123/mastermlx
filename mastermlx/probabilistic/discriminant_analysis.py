from __future__ import annotations

import numpy as np
from typing import cast

from ..base import BaseEstimator
from ..utils import accuracy, as_2d, check_1d_array, check_2d_array


def _regularize_cov(cov, reg=1e-9):
    d = cov.shape[0]
    return cov + reg * np.eye(d)


class _BaseDiscriminantAnalysis(BaseEstimator):
    def __init__(self, reg_param=1e-9):
        self.reg_param = float(reg_param)
        self.classes_ = None
        self.prior_ = None
        self.means_ = None

    def _joint_log_likelihood(self, X):
        raise NotImplementedError

    def _class_stats(self, X, y):
        classes = np.unique(y)
        n_classes = classes.size
        n_features = X.shape[1]
        means = np.zeros((n_classes, n_features), dtype=float)
        prior = np.zeros(n_classes, dtype=float)
        covs = []
        for i, c in enumerate(classes):
            Xc = X[y == c]
            means[i] = np.mean(Xc, axis=0)
            prior[i] = Xc.shape[0] / X.shape[0]
            covs.append(np.cov(Xc, rowvar=False, bias=False))
        return classes, means, prior, covs

    def predict(self, X):
        if self.classes_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = as_2d(X)
        means = cast(np.ndarray, self.means_)
        if X.shape[1] != means.shape[1]:
            raise ValueError("X has a different number of features than the fitted data")
        scores = self._joint_log_likelihood(X)
        idx = np.argmax(scores, axis=1)
        pred = self.classes_[idx]
        return pred[0] if pred.shape[0] == 1 else pred

    def score(self, X, y):
        return accuracy(y, self.predict(X))


BaseDA = _BaseDiscriminantAnalysis


class LDA(_BaseDiscriminantAnalysis):
    """Linear discriminant analysis classifier."""

    def __init__(self, reg_param=1e-9):
        super().__init__(reg_param=reg_param)
        self.covariance_ = None
        self.inv_covariance_ = None
        self.log_det_covariance_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        classes, means, prior, covs = self._class_stats(X, y)
        cov = np.zeros((X.shape[1], X.shape[1]), dtype=float)
        for i, c in enumerate(classes):
            Xc = X[y == c] - means[i]
            cov += Xc.T @ Xc
        cov /= max(X.shape[0] - classes.size, 1)
        cov = _regularize_cov(cov, self.reg_param)

        self.classes_ = classes
        self.means_ = means
        self.prior_ = prior
        self.covariance_ = cov
        self.inv_covariance_ = np.linalg.inv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError("Covariance matrix must be positive definite")
        self.log_det_covariance_ = logdet
        return self

    def _joint_log_likelihood(self, X):
        classes = cast(np.ndarray, self.classes_)
        means = cast(np.ndarray, self.means_)
        prior = cast(np.ndarray, self.prior_)
        inv_covariance = cast(np.ndarray, self.inv_covariance_)
        log_det_covariance = cast(float, self.log_det_covariance_)
        out = []
        for i in range(classes.size):
            mean = means[i]
            log_prior = np.log(prior[i] + 1e-12)
            quad = np.sum((X - mean) @ inv_covariance * (X - mean), axis=1)
            score = log_prior - 0.5 * (log_det_covariance + quad)
            out.append(score)
        return np.column_stack(out)


class QDA(_BaseDiscriminantAnalysis):
    """Quadratic discriminant analysis classifier."""

    def __init__(self, reg_param=1e-9):
        super().__init__(reg_param=reg_param)
        self.covariances_ = None
        self.inv_covariances_ = None
        self.log_det_covariances_ = None

    def fit(self, X, y=None):
        X = check_2d_array(X)
        y = check_1d_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        classes, means, prior, covs = self._class_stats(X, y)
        covariances = []
        inv_covariances = []
        log_det_covariances = []
        for cov in covs:
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])
            cov = _regularize_cov(cov, self.reg_param)
            covariances.append(cov)
            inv_covariances.append(np.linalg.inv(cov))
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                raise ValueError("Covariance matrix must be positive definite")
            log_det_covariances.append(logdet)

        self.classes_ = classes
        self.means_ = means
        self.prior_ = prior
        self.covariances_ = np.asarray(covariances)
        self.inv_covariances_ = np.asarray(inv_covariances)
        self.log_det_covariances_ = np.asarray(log_det_covariances)
        return self

    def _joint_log_likelihood(self, X):
        classes = cast(np.ndarray, self.classes_)
        means = cast(np.ndarray, self.means_)
        prior = cast(np.ndarray, self.prior_)
        inv_covariances = cast(np.ndarray, self.inv_covariances_)
        log_det_covariances = cast(np.ndarray, self.log_det_covariances_)
        out = []
        for i in range(classes.size):
            mean = means[i]
            inv_cov = inv_covariances[i]
            logdet = log_det_covariances[i]
            log_prior = np.log(prior[i] + 1e-12)
            diff = X - mean
            quad = np.sum(diff @ inv_cov * diff, axis=1)
            score = log_prior - 0.5 * (logdet + quad)
            out.append(score)
        return np.column_stack(out)
