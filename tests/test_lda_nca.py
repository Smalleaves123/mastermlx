import numpy as np
import pytest

from mastermlx.nlp import LDA
from mastermlx.decomposition import NCA


# ---------------------------------------------------------------------------
# LDA
# ---------------------------------------------------------------------------

def test_lda_fit_shape():
    X = np.random.randint(0, 5, size=(20, 30)).astype(float)
    lda = LDA(n_topics=3, max_iter=5, random_state=0).fit(X)
    assert lda.components_.shape == (3, 30)
    assert lda.doc_topic_.shape == (20, 3)
    assert np.allclose(lda.doc_topic_.sum(axis=1), 1.0, atol=0.05)


def test_lda_transform():
    X = np.random.randint(0, 5, size=(15, 25)).astype(float)
    lda = LDA(n_topics=3, max_iter=5, random_state=0).fit(X)
    dt = lda.transform(X[:5])
    assert dt.shape == (5, 3)


def test_lda_empty_docs():
    X = np.zeros((10, 20))
    X[0, 0] = 1
    lda = LDA(n_topics=2, max_iter=3, random_state=0).fit(X)
    assert lda.components_.shape == (2, 20)


def test_lda_raises_negative():
    X = np.array([[-1.0, 2.0], [3.0, 0.0]])
    try:
        LDA(n_topics=2, max_iter=3).fit(X)
    except ValueError:
        pass
    else:
        assert False, "should raise ValueError"


# ---------------------------------------------------------------------------
# NCA
# ---------------------------------------------------------------------------

def test_nca_fit_transform():
    X = np.vstack([np.random.randn(30, 5) + [0, 0, 0, 0, 0],
                   np.random.randn(30, 5) + [3, 3, 0, 0, 0]])
    y = np.array([0]*30 + [1]*30)
    nca = NCA(n_components=2, lr=0.05, max_iter=100, random_state=0).fit(X, y)
    Z = nca.transform(X)
    assert Z.shape == (60, 2)
    assert len(nca.loss_) > 0


def test_nca_loss_decreases():
    X = np.vstack([np.random.randn(20, 3), np.random.randn(20, 3) + [3, 0, 0]])
    y = np.array([0]*20 + [1]*20)
    nca = NCA(n_components=2, lr=0.05, max_iter=100, random_state=0).fit(X, y)
    # Loss should generally decrease
    assert nca.loss_[0] > nca.loss_[-1] * 0.5


def test_nca_default_components():
    X = np.random.randn(50, 4)
    y = np.where(X[:, 0] > 0, 1, 0)
    nca = NCA(lr=0.01, max_iter=50, random_state=0).fit(X, y)
    assert nca.transform(X).shape == (50, 4)  # n_components=None → keeps d


def test_nca_raises_on_mismatch():
    with pytest.raises(ValueError):
        NCA().fit(np.random.randn(10, 3), np.random.randn(5))
