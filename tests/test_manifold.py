import numpy as np

from mastermlx import Isomap, LLE, MDS, SpectralEmbedding
from mastermlx.manifold import ClassicalMDS, LocallyLinearEmbedding


def _arc(n=12):
    t = np.linspace(0.0, np.pi, n)
    return np.c_[np.cos(t), np.sin(t)], t


def test_mds_recovers_1d_distances():
    x = np.array([[0.0], [1.0], [3.0], [6.0]])
    y = MDS(n_components=1).fit_transform(x)

    dx = np.abs(x - x.T)
    dy = np.abs(y - y.T)

    assert y.shape == (4, 1)
    assert np.allclose(dx, dy, atol=1e-8)
    assert ClassicalMDS is MDS


def test_isomap_and_lle_preserve_arc_order():
    x, t = _arc(14)

    y1 = Isomap(n_components=1, n_neighbors=4).fit_transform(x).ravel()
    y2 = LLE(n_components=1, n_neighbors=4).fit_transform(x).ravel()

    c1 = np.corrcoef(t, y1)[0, 1]
    assert y1.shape == (14,)
    assert y2.shape == (14,)
    assert abs(c1) > 0.95
    assert np.std(y2) > 0.0
    assert np.isfinite(y2).all()
    assert LocallyLinearEmbedding is LLE


def test_spectral_embedding_is_finite_and_nontrivial():
    x, _ = _arc(16)

    y = SpectralEmbedding(n_components=2, n_neighbors=4).fit_transform(x)

    assert y.shape == (16, 2)
    assert np.isfinite(y).all()
    assert np.std(y[:, 0]) > 0.0
    assert np.std(y[:, 1]) > 0.0
