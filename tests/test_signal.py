import numpy as np

from mastermlx.math_tools import autocorrelation, autocorrelation_function
from mastermlx.signal import (
    convolve1d,
    frame_signal,
    hamming_window,
    hann_window,
    istft,
    moving_average,
    stft,
)


def test_signal_windows_and_convolution():
    assert np.isclose(hann_window(5)[0], 0.0)
    assert hamming_window(5).shape == (5,)

    x = np.array([1.0, 2.0, 3.0])
    kernel = np.array([1.0, 0.0, -1.0])
    y = convolve1d(x, kernel, mode="full")
    assert y.shape == (5,)


def test_signal_moving_average_and_autocorrelation():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    ma = moving_average(x, 2)
    assert np.allclose(ma, np.array([1.5, 2.5, 3.5]))

    ac = autocorrelation_function(x, max_lag=2)
    assert ac.shape == (3,)
    assert np.isclose(ac[0], 1.0)


def test_signal_frame_and_stft_roundtrip():
    x = np.sin(np.linspace(0, 2 * np.pi, 64, endpoint=False))
    frames = frame_signal(x, frame_length=16, hop_length=8, pad_end=True)
    assert frames.ndim == 2

    spec = stft(x, frame_length=16, hop_length=8, window="hann", n_fft=16, pad_end=True)
    y = istft(spec, frame_length=16, hop_length=8, window="hann", length=x.shape[0])

    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_signal_istft_roundtrip_without_window_is_exact():
    x = np.sin(np.linspace(0, 2 * np.pi, 37, endpoint=False))

    spec = stft(x, frame_length=8, hop_length=8, window=None, n_fft=8, pad_end=True)
    y = istft(spec, frame_length=8, hop_length=8, window=None, length=x.shape[0])

    assert y.shape == x.shape
    assert np.allclose(y, x, atol=1e-12)
