import numpy as np

from mastermlx import get_backend, set_backend
from mastermlx.accel import backend_report
from mastermlx.accel import rnn_ops, signal_ops, timefreq_ops
from mastermlx.signal import extract_ridge, iir_filter


def test_backend_report_has_compiled_capabilities():
    old = get_backend()
    try:
        set_backend("numpy")
        report = backend_report()
        assert report["requested"] == "numpy"
        assert report["active"] == "numpy"
        assert report["cython"] is False
        assert report["cpp_distance"] is False
    finally:
        set_backend(old)


def test_recurrent_kernels_match_numpy_fallback():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(2, 5, 3))
    W_rnn = rng.normal(scale=0.1, size=(3, 3))
    U_rnn = rng.normal(scale=0.05, size=(3, 3))
    b_rnn = rng.normal(scale=0.05, size=3)
    W = rng.normal(size=(3, 12))
    U = rng.normal(size=(3, 12))
    b = rng.normal(size=12)
    W_zr = rng.normal(size=(3, 6))
    W_h = rng.normal(size=(3, 3))
    U_zr = rng.normal(size=(3, 6))
    U_h = rng.normal(size=(3, 3))
    b_zr = rng.normal(size=6)
    b_h = rng.normal(size=3)
    old = get_backend()
    try:
        set_backend("numpy")
        rnn_ref = rnn_ops.simple_rnn_forward(X, W_rnn, U_rnn, b_rnn)
        lstm_ref = rnn_ops.lstm_forward(X, W, U, b, 3)
        gru_ref = rnn_ops.gru_forward(X, W_zr, W_h, U_zr, U_h, b_zr, b_h, 3)
        set_backend("auto")
        rnn = rnn_ops.simple_rnn_forward(X, W_rnn, U_rnn, b_rnn)
        lstm = rnn_ops.lstm_forward(X, W, U, b, 3)
        assert np.allclose(rnn_ref, rnn, atol=1e-12)
        gru = rnn_ops.gru_forward(X, W_zr, W_h, U_zr, U_h, b_zr, b_h, 3)
        for expected, actual in zip(lstm_ref, lstm):
            assert np.allclose(expected, actual, atol=1e-12)
        for expected, actual in zip(gru_ref, gru):
            assert np.allclose(expected, actual, atol=1e-12)
    finally:
        set_backend(old)


def test_iir_and_ridge_kernels_match_numpy_fallback():
    rng = np.random.default_rng(5)
    x = rng.normal(size=256)
    b = np.array([0.2, 0.1, -0.04])
    a = np.array([1.0, -0.35, 0.08])
    score = rng.normal(size=(30, 64))
    old = get_backend()
    try:
        set_backend("numpy")
        iir_ref = iir_filter(x, b, a)
        ridge_ref = timefreq_ops.ridge_path(score, 0.25, 3)
        set_backend("auto")
        iir = iir_filter(x, b, a)
        ridge = timefreq_ops.ridge_path(score, 0.25, 3)
        assert np.allclose(iir_ref, iir, atol=1e-12)
        assert np.array_equal(ridge_ref, ridge)
    finally:
        set_backend(old)


def test_extract_ridge_preserves_path_score():
    score = np.array([[1.0, 1.0], [0.0, 2.0]])
    result = extract_ridge(score, [10.0, 20.0], smoothness=0.5, log_power=False)
    expected = score[result["indices"][0], 0]
    expected += score[result["indices"][1], 1]
    expected -= 0.5 * (result["indices"][1] - result["indices"][0]) ** 2
    assert np.isclose(result["score"], expected)


def test_numpy_backend_skips_new_extension_imports(monkeypatch):
    old = get_backend()
    try:
        set_backend("numpy")

        def fail_import(*args, **kwargs):
            raise AssertionError("compiled extension import was attempted")

        monkeypatch.setattr(rnn_ops.importlib, "import_module", fail_import)
        monkeypatch.setattr(signal_ops.importlib, "import_module", fail_import)
        monkeypatch.setattr(timefreq_ops.importlib, "import_module", fail_import)
        X = np.zeros((1, 2, 1))
        rnn_ops.simple_rnn_forward(X, np.zeros((1, 1)), np.zeros((1, 1)), np.zeros(1))
        signal_ops.iir_filter_1d(np.ones(4), np.ones(1), np.ones(1))
        timefreq_ops.ridge_path(np.ones((2, 3)), 0.0, None)
    finally:
        set_backend(old)
