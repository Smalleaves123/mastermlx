import numpy as np

from mastermlx.preprocessing import (
    Binarizer,
    MaxAbsScaler,
    PowerTransform,
    QuantileTransform,
    TargetEncoder,
)


# ---------------------------------------------------------------------------
# MaxAbsScaler
# ---------------------------------------------------------------------------


def test_maxabs_fit_transform():
    X = np.array([[1.0, -2.0], [3.0, -4.0]])
    s = MaxAbsScaler().fit(X)
    Xt = s.transform(X)
    assert np.allclose(Xt, [[1.0 / 3, -2.0 / 4], [3.0 / 3, -4.0 / 4]])
    assert np.allclose(s.inverse_transform(Xt), X)


def test_maxabs_zero_col():
    X = np.array([[0.0, 1.0], [0.0, 2.0]])
    s = MaxAbsScaler().fit(X)
    Xt = s.transform(X)
    assert np.allclose(Xt[:, 0], [0.0, 0.0])


# ---------------------------------------------------------------------------
# Binarizer
# ---------------------------------------------------------------------------


def test_binarizer_default():
    X = np.array([[1.0, -1.0], [2.0, 0.0]])
    B = Binarizer().fit(X)
    Xt = B.transform(X)
    assert np.array_equal(Xt, [[1, 0], [1, 0]])


def test_binarizer_custom_thresh():
    X = np.array([[1.0, 3.0, 5.0]])
    B = Binarizer(threshold=2.5).fit(X)
    Xt = B.transform(X)
    assert np.array_equal(Xt, [[0, 1, 1]])


# ---------------------------------------------------------------------------
# PowerTransform
# ---------------------------------------------------------------------------


def test_power_yj_ranges():
    np.random.seed(42)
    X = np.random.exponential(scale=3.0, size=(200, 2))
    pt = PowerTransform().fit(X)
    Xt = pt.transform(X)
    assert Xt.shape == X.shape
    assert np.all(np.isfinite(Xt))
    assert np.std(Xt[:, 0]) > 0


def test_power_inverse_roundtrip():
    np.random.seed(42)
    X = np.random.exponential(scale=2.0, size=(100, 2))
    pt = PowerTransform().fit(X)
    Xt = pt.transform(X)
    Xi = pt.inverse_transform(Xt)
    assert np.allclose(X, Xi, atol=1e-6)


# ---------------------------------------------------------------------------
# QuantileTransform
# ---------------------------------------------------------------------------


def test_quantile_uniform():
    np.random.seed(42)
    X = np.random.exponential(scale=3.0, size=(200, 2))
    qt = QuantileTransform(output_distribution="uniform").fit(X)
    Xt = qt.transform(X)
    assert Xt.shape == X.shape
    assert Xt.min() >= -0.01
    assert Xt.max() <= 1.01


def test_quantile_normal():
    np.random.seed(42)
    X = np.random.exponential(scale=3.0, size=(200, 2))
    qt = QuantileTransform(output_distribution="normal").fit(X)
    Xt = qt.transform(X)
    assert Xt.shape == X.shape
    assert np.all(np.isfinite(Xt))


# ---------------------------------------------------------------------------
# TargetEncoder
# ---------------------------------------------------------------------------


def test_target_encoder_basic():
    X = np.array([["a"], ["b"], ["a"], ["c"]])
    y = np.array([1.0, 3.0, 1.0, 5.0])
    te = TargetEncoder(smoothing=0.0).fit(X, y)
    Xt = te.transform(X)
    # a: mean=(1+1)/2=1.0, b: mean=3.0, c: mean=5.0
    assert np.allclose(Xt.ravel(), [1.0, 3.0, 1.0, 5.0])


def test_target_encoder_smoothing():
    X = np.array([["a"], ["a"], ["b"]])
    y = np.array([1.0, 2.0, 10.0])
    te = TargetEncoder(smoothing=2.0).fit(X, y)
    Xt = te.transform(X)
    # global_mean = (1+2+10)/3 = 13/3 ≈ 4.333
    # a: smoothed = (1+2 + 2*4.333) / (2+2) = (3 + 8.667) / 4 = 2.917
    global_mean = np.mean(y)
    a_smoothed = (3.0 + 2.0 * global_mean) / (2.0 + 2.0)
    assert np.isclose(Xt[0, 0], a_smoothed)
