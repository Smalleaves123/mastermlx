import numpy as np

from mastermlx import Isomap, LLE, MDS, SpectralEmbedding, get_backend, set_backend
from mastermlx.manifold import ClassicalMDS, LocallyLinearEmbedding
from mastermlx.manifold._core import all_pairs_shortest


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


def test_isomap_geodesic_cpp_and_numpy_paths_match(monkeypatch):
    W = np.full((8, 8), np.inf)
    for i in range(8):
        W[i, i] = 0.0
        if i > 0:
            W[i, i - 1] = 1.0
        if i + 1 < 8:
            W[i, i + 1] = 1.0
    old = get_backend()
    try:
        set_backend("auto")
        accelerated = all_pairs_shortest(W)
        set_backend("numpy")
        fallback = all_pairs_shortest(W)
    finally:
        set_backend(old)

    assert np.allclose(accelerated, fallback)
    assert np.allclose(accelerated, np.abs(np.arange(8)[:, None] - np.arange(8)[None, :]))

    import mastermlx.manifold._core as core

    monkeypatch.setattr(core, "_load_cpp", lambda backend: object())
    assert np.allclose(core.all_pairs_shortest(W), fallback)
