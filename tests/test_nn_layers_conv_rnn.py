import numpy as np

from mastermlx.neural_net.layers import (
    Conv2D, Flatten, GELU, GRU, LayerNorm, LeakyReLU,
    LSTM, MaxPool2D, SimpleRNN,
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


# --- MaxPool2D ---
def test_maxpool_shape():
    p = MaxPool2D(kernel_size=2)
    X = np.random.randn(4, 8, 8, 3)
    out = p.forward(X)
    assert out.shape == (4, 4, 4, 3)


# --- Flatten ---
def test_flatten_shape():
    f = Flatten()
    out = f.forward(np.random.randn(2, 4, 4, 8))
    assert out.shape == (2, 128)
    assert f.backward(np.ones_like(out)).shape == (2, 4, 4, 8)


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
