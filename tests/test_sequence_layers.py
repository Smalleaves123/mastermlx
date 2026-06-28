import numpy as np

from mastermlx.neural_net import Dense, Embedding, GlobalAveragePooling1D, Sequential


def test_embedding_layer_accumulates_repeated_ids():
    layer = Embedding(n_tokens=6, dim=3, random_state=0, padding_idx=0)
    X = np.array([[1, 2, 1], [0, 3, 4]])

    out = layer.forward(X)
    assert out.shape == (2, 3, 3)
    assert np.allclose(out[1, 0], 0.0)

    grad = np.ones_like(out)
    back = layer.backward(grad)
    assert back.shape == X.shape
    before = layer.W_.copy()
    layer.step(lr=0.1)
    assert not np.allclose(layer.W_, before)
    assert np.allclose(layer.W_[0], 0.0)


def test_global_average_pooling_reduces_sequence_axis():
    layer = GlobalAveragePooling1D()
    X = np.arange(24, dtype=float).reshape(2, 4, 3)

    out = layer.forward(X)
    assert out.shape == (2, 3)
    assert np.allclose(out[0], np.mean(X[0], axis=0))

    grad = np.ones_like(out)
    back = layer.backward(grad)
    assert back.shape == X.shape
    assert np.allclose(back[0, 0], np.ones(3) / 4.0)


def test_sequential_can_fit_a_tiny_sequence_problem():
    X = np.array([
        [1, 1, 2],
        [1, 2, 2],
        [3, 3, 4],
        [3, 4, 4],
    ])
    y = np.array([0, 0, 1, 1])

    model = Sequential(
        layers=[
            Embedding(n_tokens=5, dim=4, random_state=0, padding_idx=0),
            GlobalAveragePooling1D(),
            Dense(4, 6, random_state=1),
            Dense(6, 2, random_state=2),
        ],
        optimizer="adam",
        lr=0.05,
        n_iter=2500,
        random_state=0,
        task="classification",
    )
    model.fit(X, y)

    pred = model.predict(X)
    assert np.array_equal(pred, y)
