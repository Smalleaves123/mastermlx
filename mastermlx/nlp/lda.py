from __future__ import annotations

import numpy as np

from ..utils.random import resolve_rng


class LDA:
    """Latent Dirichlet Allocation via online variational Bayes.

    Parameters
    ----------
    n_topics : int
        Number of topics.
    alpha : float
        Dirichlet prior on document-topic distributions.
    eta : float
        Dirichlet prior on topic-word distributions.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on per-word bound.
    random_state : int or None
        Random seed.
    """

    def __init__(self, n_topics=10, alpha=0.1, eta=0.1, max_iter=100,
                 tol=1e-4, random_state=None):
        self.n_topics = int(n_topics)
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.components_ = None
        self.exp_dirichlet_component_ = None
        self.doc_topic_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D document-term matrix")
        if np.any(X < 0):
            raise ValueError("X must be non-negative")
        n_docs, n_words = X.shape
        k = self.n_topics
        rng = resolve_rng(self.random_state)

        # Initialize topic-word matrix (k, n_words) as exp(E[log beta])
        self.components_ = rng.gamma(1.0, 1.0, size=(k, n_words))
        self.components_ /= self.components_.sum(axis=1, keepdims=True)
        E_log_beta = np.log(self.components_ + 1e-12)

        self.doc_topic_ = np.zeros((n_docs, k), dtype=float)
        bound_prev = -np.inf

        for it in range(self.max_iter):
            bound = 0.0
            for d in range(n_docs):
                doc = X[d]
                doc_sum = int(doc.sum())
                if doc_sum == 0:
                    self.doc_topic_[d] = 1.0 / k
                    continue

                # E-step: variational inference for doc d
                gamma = np.full(k, self.alpha + doc_sum / k, dtype=float)
                phi = np.zeros((doc_sum, k), dtype=float) if doc_sum > 0 else np.zeros((0, k))

                if doc_sum > 0:
                    word_ids = np.repeat(np.arange(n_words), doc.astype(int))
                    n_tokens = len(word_ids)
                    if n_tokens > 0:
                        phi = rng.dirichlet(np.ones(k), size=n_tokens)
                        for _ in range(20):  # inner VI loop
                            phi_prev = phi.copy()
                            # Update phi
                            Elog_theta = np.log(gamma + 1e-12) - np.log(np.sum(gamma))
                            log_phi = Elog_theta + E_log_beta[:, word_ids].T
                            log_phi -= np.max(log_phi, axis=1, keepdims=True)
                            phi = np.exp(log_phi)
                            phi /= phi.sum(axis=1, keepdims=True)
                            # Update gamma
                            gamma_new = self.alpha + phi.sum(axis=0)
                            if np.max(np.abs(phi - phi_prev)) < 1e-4:
                                break
                            gamma = gamma_new

                self.doc_topic_[d] = gamma / gamma.sum()

                # M-step: accumulate statistics
                if doc_sum > 0:
                    word_ids = np.repeat(np.arange(n_words), doc.astype(int))
                    for w in range(n_words):
                        mask = word_ids == w
                        if np.any(mask):
                            self.components_[:, w] += self.eta + phi[mask].sum(axis=0)
                else:
                    self.components_ += self.eta

            # Normalize components
            self.components_ /= self.components_.sum(axis=1, keepdims=True)
            E_log_beta = np.log(self.components_ + 1e-12)

            # Simple bound check
            if abs(bound - bound_prev) < self.tol and it > 2:
                break
            bound_prev = bound

        self.exp_dirichlet_component_ = self.components_
        return self

    def transform(self, X):
        """Return document-topic distributions for new documents."""
        X = np.asarray(X, dtype=float)
        if self.components_ is None:
            raise RuntimeError("Model has not been fit yet")
        n_docs, n_words = X.shape
        k = self.n_topics
        E_log_beta = np.log(self.components_ + 1e-12)
        doc_topic = np.zeros((n_docs, k))

        for d in range(n_docs):
            doc = X[d]
            doc_sum = int(doc.sum())
            if doc_sum == 0:
                doc_topic[d] = 1.0 / k
                continue
                continue
            gamma = np.full(k, self.alpha + doc_sum / k, dtype=float)
            word_ids = np.repeat(np.arange(n_words), doc.astype(int))
            phi = np.ones((len(word_ids), k)) / k
            for _ in range(20):
                Elog_theta = np.log(gamma + 1e-12) - np.log(np.sum(gamma))
                log_phi = Elog_theta + E_log_beta[:, word_ids].T
                log_phi -= np.max(log_phi, axis=1, keepdims=True)
                phi = np.exp(log_phi)
                phi /= phi.sum(axis=1, keepdims=True)
                gamma = self.alpha + phi.sum(axis=0)
            doc_topic[d] = gamma / gamma.sum()
        return doc_topic

    def fit_transform(self, X, y=None):
        return self.fit(X).doc_topic_
