import numpy as np

from mastermlx.decomposition import FastICA


def test_fastica_shapes_are_consistent():
    t = np.linspace(0.0, 8.0 * np.pi, 200)
    s1 = np.sin(t)
    s2 = np.sign(np.sin(3.0 * t))
    S = np.column_stack([s1, s2])
    A = np.array([[1.0, 0.5], [0.3, 1.2]])
    X = S @ A.T

    ica = FastICA(n_components=2, max_iter=1000, tol=1e-5, random_state=0)
    Z = ica.fit_transform(X)
    X2 = ica.inverse_transform(Z)

    assert Z.shape == X.shape
    assert ica.components_.shape == (2, 2)
    assert X2.shape == X.shape
