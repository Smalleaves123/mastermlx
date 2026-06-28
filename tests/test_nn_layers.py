import numpy as np

from mastermlx.neural_net.layers import AttentionPooling1D, BatchNorm, Dense, Dropout, MultiHeadAttention, ReLU
from mastermlx.neural_net import Sequential


def test_dropout_switches_between_train_and_eval():
    X = np.ones((8, 4))
    layer = Dropout(rate=0.5, random_state=0)

    out_train = layer.forward(X)
    assert out_train.shape == X.shape
    assert np.any(out_train == 0.0)

    layer.eval()
    out_eval = layer.forward(X)
    assert np.allclose(out_eval, X)


def test_batchnorm_normalizes_in_train_mode():
    X = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
    ])

    bn = BatchNorm(n_features=3)
    out = bn.forward(X)

    assert out.shape == X.shape
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-7)
    assert np.allclose(out.var(axis=0), 1.0, atol=1e-5)

    bn.eval()
    out_eval = bn.forward(X)
    assert out_eval.shape == X.shape


def test_sequential_can_fit_xor_with_batchnorm_and_dropout():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([0, 1, 1, 0])

    model = Sequential(
        layers=[
            Dense(2, 8, random_state=0),
            BatchNorm(8),
            ReLU(),
            Dropout(rate=0.0, random_state=0),
            Dense(8, 2, random_state=1),
        ],
        optimizer="adam",
        lr=0.05,
        n_iter=2000,
        random_state=0,
        task="classification",
    )
    model.fit(X, y)
    pred = model.predict(X)

    assert np.array_equal(pred, y)
    assert model.score(X, y) == 1.0


def test_attention_pooling_forward_backward_shapes():
    X = np.array([
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        [[0.5, 0.2], [0.1, 0.9], [0.4, 0.4]],
    ])

    layer = AttentionPooling1D(hidden_size=4, random_state=0)
    out = layer.forward(X)
    grad = np.ones_like(out)
    back = layer.backward(grad)

    assert out.shape == (2, 2)
    assert back.shape == X.shape
    assert np.isfinite(out).all()
    assert np.isfinite(back).all()


def test_multi_head_attention_forward_backward_shapes():
    X = np.array([
        [[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, -0.5, 0.5], [1.0, 1.0, 0.2, 0.2]],
        [[0.3, 0.1, 0.7, -0.2], [0.2, 0.8, -0.1, 0.4], [0.9, 0.4, 0.0, 0.1]],
    ])

    layer = MultiHeadAttention(n_features=4, n_heads=2, random_state=0)
    out = layer.forward(X)
    back = layer.backward(np.ones_like(out))

    assert out.shape == X.shape
    assert back.shape == X.shape
    assert np.isfinite(out).all()
    assert np.isfinite(back).all()


def test_sequential_can_fit_sequence_task_with_attention_pooling():
    X = np.array([
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
    ])
    y = np.array([0, 1, 1, 0])

    model = Sequential(
        layers=[
            AttentionPooling1D(hidden_size=6, random_state=0),
            Dense(2, 2, random_state=1),
        ],
        optimizer="adam",
        lr=0.05,
        n_iter=1200,
        random_state=0,
        task="classification",
    )
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert model.score(X, y) >= 0.75
