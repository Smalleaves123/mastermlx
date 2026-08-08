from __future__ import annotations

import numpy as np

from ..utils.random import resolve_rng
from ..variational.utils import digamma, log_gamma


class LDA:
    """Latent Dirichlet allocation via batch variational Bayes.

    Parameters
    ----------
    n_topics : int
        Number of topics.
    alpha : float
        Symmetric Dirichlet prior on document-topic distributions.
    eta : float
        Symmetric Dirichlet prior on topic-word distributions.
    max_iter : int
        Maximum variational EM iterations.
    tol : float
        Convergence tolerance on the maximum topic-word probability change.
        Set to zero to always run ``max_iter`` iterations.
    random_state : int, numpy.random.Generator, or None
        Random seed or generator used to initialize the topic-word posterior.
    """

    def __init__(
        self,
        n_topics=10,
        alpha=0.1,
        eta=0.1,
        max_iter=100,
        tol=1e-4,
        random_state=None,
    ):
        self.n_topics = int(n_topics)
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.components_ = None
        self.exp_dirichlet_component_ = None
        self.doc_topic_ = None
        self.n_features_in_ = None
        self.n_iter_ = 0
        self.bound_ = None
        self.bound_trace_ = []
        self._lambda = None

    def _validate_parameters(self):
        if self.n_topics < 1:
            raise ValueError("n_topics must be at least 1")
        if not np.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("alpha must be positive and finite")
        if not np.isfinite(self.eta) or self.eta <= 0.0:
            raise ValueError("eta must be positive and finite")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if not np.isfinite(self.tol) or self.tol < 0.0:
            raise ValueError("tol must be non-negative and finite")

    @staticmethod
    def _validate_counts(X, *, n_features=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D document-term matrix")
        if n_features is not None and X.shape[1] != n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but the fitted model expects {n_features}"
            )
        if not np.all(np.isfinite(X)) or np.any(X < 0.0):
            raise ValueError("X must contain finite non-negative counts")
        if not np.allclose(X, np.rint(X), rtol=0.0, atol=1e-12):
            raise ValueError("X must contain integer document-term counts")
        return X

    @staticmethod
    def _expected_log_dirichlet(parameters):
        parameters = np.asarray(parameters, dtype=float)
        return digamma(parameters) - digamma(np.sum(parameters, axis=-1))[..., None]

    def _infer_document(self, counts, expected_log_beta):
        word_ids = np.flatnonzero(counts)
        total = float(np.sum(counts))
        if word_ids.size == 0:
            gamma = np.full(self.n_topics, self.alpha, dtype=float)
            return gamma, word_ids, np.empty((self.n_topics, 0), dtype=float), 0.0

        word_counts = counts[word_ids]
        gamma = np.full(
            self.n_topics,
            self.alpha + total / self.n_topics,
            dtype=float,
        )
        phi = np.empty((self.n_topics, word_ids.size), dtype=float)
        inner_tol = max(self.tol, 1e-8)
        for _ in range(100):
            expected_log_theta = self._expected_log_dirichlet(gamma[None, :])[0]
            log_phi = expected_log_theta[:, None] + expected_log_beta[:, word_ids]
            log_phi -= np.max(log_phi, axis=0, keepdims=True)
            phi = np.exp(log_phi)
            phi /= np.sum(phi, axis=0, keepdims=True)
            gamma_new = self.alpha + phi @ word_counts
            delta = float(np.max(np.abs(gamma_new - gamma)))
            gamma = gamma_new
            if delta < inner_tol:
                break

        # Recompute phi from the final gamma.  In particular, this keeps the
        # sufficient statistics aligned when the inner loop exits early.
        expected_log_theta = self._expected_log_dirichlet(gamma[None, :])[0]
        log_phi = expected_log_theta[:, None] + expected_log_beta[:, word_ids]
        log_phi -= np.max(log_phi, axis=0, keepdims=True)
        phi = np.exp(log_phi)
        phi /= np.sum(phi, axis=0, keepdims=True)

        entropy = -np.log(np.maximum(phi, 1e-300))
        word_bound = float(
            np.sum(
                word_counts[None, :]
                * phi
                * (expected_log_theta[:, None] + expected_log_beta[:, word_ids] + entropy)
            )
        )
        topic_prior = float(
            log_gamma(np.array([self.n_topics * self.alpha]))[0]
            - self.n_topics * log_gamma(np.array([self.alpha]))[0]
            + np.sum((self.alpha - 1.0) * expected_log_theta)
        )
        topic_entropy = float(
            log_gamma(np.array([np.sum(gamma)]))[0]
            - np.sum(log_gamma(gamma))
            + np.sum((gamma - 1.0) * expected_log_theta)
        )
        return gamma, word_ids, phi, word_bound + topic_prior - topic_entropy

    def _beta_bound(self, parameters, expected_log_beta):
        prior = (
            log_gamma(np.array([parameters.shape[1] * self.eta]))[0]
            - parameters.shape[1] * log_gamma(np.array([self.eta]))[0]
            + np.sum((self.eta - 1.0) * expected_log_beta, axis=1)
        )
        posterior = (
            log_gamma(np.sum(parameters, axis=1))
            - np.sum(log_gamma(parameters), axis=1)
            + np.sum((parameters - 1.0) * expected_log_beta, axis=1)
        )
        return float(np.sum(prior - posterior))

    def _e_step(self, X, parameters):
        expected_log_beta = self._expected_log_dirichlet(parameters)
        stats = np.zeros_like(parameters)
        doc_topic = np.zeros((X.shape[0], self.n_topics), dtype=float)
        bound = self._beta_bound(parameters, expected_log_beta)
        for document_index, counts in enumerate(X):
            gamma, word_ids, phi, document_bound = self._infer_document(
                counts, expected_log_beta
            )
            doc_topic[document_index] = gamma / np.sum(gamma)
            if word_ids.size:
                stats[:, word_ids] += phi * counts[word_ids][None, :]
            bound += document_bound
        return doc_topic, stats, float(bound)

    def fit(self, X, y=None):
        self._validate_parameters()
        X = self._validate_counts(X)
        self.n_features_in_ = int(X.shape[1])
        rng = resolve_rng(self.random_state)

        parameters = self.eta + rng.gamma(
            shape=1.0,
            scale=1.0,
            size=(self.n_topics, self.n_features_in_),
        )
        previous_components = parameters / np.sum(parameters, axis=1, keepdims=True)
        self.bound_trace_ = []

        for iteration in range(1, self.max_iter + 1):
            doc_topic, stats, bound = self._e_step(X, parameters)
            # The M-step is a replacement, not an accumulation into the
            # normalized topic probabilities from a previous EM iteration.
            parameters = self.eta + stats
            components = parameters / np.sum(parameters, axis=1, keepdims=True)
            change = float(np.max(np.abs(components - previous_components)))
            self.bound_trace_.append(bound)
            self.n_iter_ = iteration
            previous_components = components
            if self.tol > 0.0 and change < self.tol:
                break

        # Run one final E-step so doc_topic_ and bound_ correspond to the final
        # M-step parameters rather than the previous topic-word iterate.
        self.doc_topic_, _, self.bound_ = self._e_step(X, parameters)
        self._lambda = parameters
        self.components_ = parameters / np.sum(parameters, axis=1, keepdims=True)
        self.exp_dirichlet_component_ = np.exp(
            self._expected_log_dirichlet(parameters)
        )
        return self

    def transform(self, X):
        """Return document-topic distributions for new documents."""

        if self._lambda is None or self.n_features_in_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = self._validate_counts(X, n_features=self.n_features_in_)
        doc_topic, _, _ = self._e_step(X, self._lambda)
        return doc_topic

    def score(self, X, y=None):
        """Return the variational evidence lower bound for ``X``."""

        if self._lambda is None or self.n_features_in_ is None:
            raise RuntimeError("Model has not been fit yet")
        X = self._validate_counts(X, n_features=self.n_features_in_)
        _, _, bound = self._e_step(X, self._lambda)
        return float(bound)

    def perplexity(self, X):
        """Return an ELBO-based approximate perplexity for ``X``."""

        X = self._validate_counts(X, n_features=self.n_features_in_)
        tokens = float(np.sum(X))
        if tokens <= 0.0:
            raise ValueError("perplexity requires at least one observed token")
        return float(np.exp(-self.score(X) / tokens))

    def fit_transform(self, X, y=None):
        return self.fit(X, y).doc_topic_
