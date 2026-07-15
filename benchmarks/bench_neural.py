"""Compare NumPy and optional compiled hot paths used by neural and signal code."""

from __future__ import annotations

import time

import numpy as np

from mastermlx import set_backend
from mastermlx.accel import backend_report
from mastermlx.accel.conv1d_ops import im2col1d
from mastermlx.accel.rnn_ops import gru_forward, lstm_forward, simple_rnn_forward
from mastermlx.accel.signal_ops import iir_filter_1d
from mastermlx.accel.timefreq_ops import ridge_path


def bench(fn, runs=5):
    fn()
    elapsed = []
    for _ in range(runs):
        start = time.perf_counter()
        result = fn()
        elapsed.append(time.perf_counter() - start)
    return float(np.mean(elapsed)), result


def compare(name, fn, runs=5):
    set_backend("numpy")
    numpy_time, reference = bench(fn, runs)
    set_backend("auto")
    compiled_time, result = bench(fn, runs)
    if isinstance(reference, tuple):
        same = all(np.allclose(a, b, atol=1e-10, rtol=1e-10) for a, b in zip(reference, result))
    else:
        same = np.allclose(reference, result, atol=1e-10, rtol=1e-10)
    speedup = numpy_time / compiled_time if compiled_time else float("inf")
    print(f"  {name:18s} NumPy={numpy_time:8.5f}s  auto={compiled_time:8.5f}s  "
          f"speedup={speedup:6.2f}x  equal={same}")


def main():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 96, 8))
    W_xh = rng.normal(scale=0.15, size=(8, 16))
    W_hh = rng.normal(scale=0.05, size=(16, 16))
    b_rnn = rng.normal(scale=0.05, size=16)
    W_lstm = rng.normal(scale=0.15, size=(8, 64))
    U_lstm = rng.normal(scale=0.05, size=(16, 64))
    b_lstm = rng.normal(scale=0.05, size=64)
    W_zr = rng.normal(scale=0.15, size=(8, 32))
    W_h = rng.normal(scale=0.15, size=(8, 16))
    U_zr = rng.normal(scale=0.05, size=(16, 32))
    U_h = rng.normal(scale=0.05, size=(16, 16))
    b_zr = rng.normal(scale=0.05, size=32)
    b_h = rng.normal(scale=0.05, size=16)
    x = rng.normal(size=20_000)
    b_iir = np.array([0.2, 0.1, -0.04])
    a_iir = np.array([1.0, -0.35, 0.08])
    score = rng.normal(size=(128, 400))

    print("Backend report:", backend_report())
    print("\nNeural and signal hot paths")
    compare("Conv1D packing", lambda: im2col1d(X, 7, 1, 0), runs=3)
    compare("SimpleRNN", lambda: simple_rnn_forward(X, W_xh, W_hh, b_rnn), runs=3)
    compare("LSTM", lambda: lstm_forward(X, W_lstm, U_lstm, b_lstm, 16), runs=3)
    compare("GRU", lambda: gru_forward(X, W_zr, W_h, U_zr, U_h, b_zr, b_h, 16), runs=3)
    compare("IIR filter", lambda: iir_filter_1d(x, b_iir, a_iir), runs=3)
    compare("Ridge path", lambda: ridge_path(score, 0.25, 3), runs=3)
    set_backend("auto")


if __name__ == "__main__":
    main()
