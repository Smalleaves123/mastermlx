import numpy as np

from mastermlx.neural_net.layers import (
    AvgPool1D, AvgPool2D, Conv1D, Conv2D, Flatten, GELU, GlobalAveragePooling2D,
    GRU, LayerNorm, LeakyReLU, LSTM, MaxPool2D, SimpleRNN,
)


# --- Conv2D ---
def test_conv2d_forward_shape():
    X = np.random.randn(4, 8, 8, 3)
    c = Conv2D(n_filters=5, kernel_size=3).forward(X)
    assert c.shape == (4, 6, 6, 5)


def test_conv2d_backward_shape():
    X = np.random.randn(2, 6, 6, 2)
    layer = Conv2D(n_filters=3, kernel_size=3)
    out = layer.forward(X)
    dX = layer.backward(np.ones_like(out))
    assert dX.shape == X.shape


def test_conv2d_backward_requires_forward():
    layer = Conv2D(n_filters=2, kernel_size=3)
    try:
        layer.backward(np.ones((1, 1, 1, 2)))
    except RuntimeError as exc:
        assert "forward must be called before backward" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_conv2d_forward_known_values():
    X = np.arange(1, 10, dtype=float).reshape(1, 3, 3, 1)
    layer = Conv2D(n_filters=1, kernel_size=2, stride=1)
    layer.W_ = np.ones((4, 1), dtype=float)
    layer.b_ = np.zeros(1, dtype=float)
    out = layer.forward(X)

    expected = np.array([[[[12.0], [16.0]], [[24.0], [28.0]]]])
    assert np.allclose(out, expected)


def test_conv2d_forward_stride_two():
    X = np.arange(1, 17, dtype=float).reshape(1, 4, 4, 1)
    layer = Conv2D(n_filters=1, kernel_size=2, stride=2)
    layer.W_ = np.ones((4, 1), dtype=float)
    layer.b_ = np.zeros(1, dtype=float)
    out = layer.forward(X)

    expected = np.array([[[[14.0], [22.0]], [[46.0], [54.0]]]])
    assert np.allclose(out, expected)


# --- MaxPool2D ---
def test_maxpool_shape():
    p = MaxPool2D(kernel_size=2)
    X = np.random.randn(4, 8, 8, 3)
    out = p.forward(X)
    assert out.shape == (4, 4, 4, 3)


def test_maxpool_backward_routes_gradient_to_max():
    X = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    p = MaxPool2D(kernel_size=2)
    out = p.forward(X)
    back = p.backward(np.ones_like(out))

    assert out.shape == (1, 1, 1, 1)
    assert np.allclose(out, np.array([[[[4.0]]]]))
    assert np.allclose(back, np.array([[[[0.0], [0.0]], [[0.0], [1.0]]]]))


def test_maxpool_backward_requires_forward():
    p = MaxPool2D(kernel_size=2)
    try:
        p.backward(np.ones((1, 1, 1, 1)))
    except RuntimeError as exc:
        assert "forward must be called before backward" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


# --- Flatten ---
def test_flatten_shape():
    f = Flatten()
    out = f.forward(np.random.randn(2, 4, 4, 8))
    assert out.shape == (2, 128)
    assert f.backward(np.ones_like(out)).shape == (2, 4, 4, 8)


def test_flatten_backward_requires_forward():
    f = Flatten()
    try:
        f.backward(np.ones((2, 8)))
    except RuntimeError as exc:
        assert "forward must be called before backward" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


# --- GELU / LeakyReLU ---
def test_gelu():
    g = GELU()
    out = g.forward(np.array([[-1.0, 0.0, 1.0]]))
    assert out[0, 0] < out[0, 2]
    assert g.backward(np.ones_like(out)).shape == (1, 3)


def test_leaky_relu():
    lr = LeakyReLU(alpha=0.1)
    X = np.array([[-1.0, 2.0]])
    out = lr.forward(X)
    assert out[0, 0] == -0.1
    assert out[0, 1] == 2.0


# --- LayerNorm ---
def test_layer_norm():
    X = np.random.randn(4, 5, 8)
    ln = LayerNorm(8)
    out = ln.forward(X)
    assert out.shape == (4, 5, 8)
    assert np.allclose(np.mean(out, axis=-1), 0, atol=1e-6)
    assert np.allclose(np.var(out, axis=-1), 1, atol=0.05)


# --- RNN ---
def test_rnn_forward_shape():
    X = np.random.randn(3, 10, 5)
    rnn = SimpleRNN(n_units=8)
    out = rnn.forward(X)
    assert out.shape == (3, 8)
    out_seq = SimpleRNN(n_units=8, return_sequences=True).forward(X)
    assert out_seq.shape == (3, 10, 8)


# --- LSTM ---
def test_lstm_forward():
    X = np.random.randn(2, 5, 4)
    lstm = LSTM(n_units=6)
    out = lstm.forward(X)
    assert out.shape == (2, 6)
    lstm.backward(np.ones_like(out))
    lstm.step(lr=0.01)


# --- GRU ---
def test_gru_forward():
    X = np.random.randn(3, 7, 4)
    gru = GRU(n_units=5, return_sequences=True)
    out = gru.forward(X)
    assert out.shape == (3, 7, 5)
    gru.backward(np.ones_like(out))
    gru.step(lr=0.01)


# --- Conv1D ---
def test_conv1d_forward_shape():
    X = np.random.randn(4, 10, 3)
    c = Conv1D(n_filters=5, kernel_size=3)
    out = c.forward(X)
    assert out.shape == (4, 8, 5)


def test_conv1d_backward_shape():
    from mastermlx.neural_net.layers.conv1d import Conv1D
    X = np.random.randn(2, 8, 3)
    c = Conv1D(n_filters=4, kernel_size=3)
    out = c.forward(X)
    dX = c.backward(np.ones_like(out))
    assert dX.shape == X.shape


# --- AvgPool ---
def test_avgpool1d_shape():
    from mastermlx.neural_net.layers.conv1d import AvgPool1D
    p = AvgPool1D(kernel_size=2)
    X = np.random.randn(3, 8, 4)
    assert p.forward(X).shape == (3, 4, 4)


def test_avgpool2d_shape():
    from mastermlx.neural_net.layers.pooling import AvgPool2D
    p = AvgPool2D(kernel_size=2)
    X = np.random.randn(2, 6, 6, 3)
    assert p.forward(X).shape == (2, 3, 3, 3)


def test_global_avgpool2d_shape():
    from mastermlx.neural_net.layers.pooling import GlobalAveragePooling2D
    g = GlobalAveragePooling2D()
    X = np.random.randn(2, 4, 4, 8)
    assert g.forward(X).shape == (2, 8)
