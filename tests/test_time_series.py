import numpy as np

from mastermlx.math_tools import (
    ARModel,
    autocorrelation,
    autocorrelation_function,
    cusum_change_points,
    dtw_distance,
    dtw_path,
    difference,
    exponential_smoothing,
    partial_autocorrelation,
    rolling_mean,
)


def test_time_series_basic_transforms():
    x = np.array([1.0, 2.0, 4.0, 7.0, 11.0])

    d1 = difference(x)
    rm = rolling_mean(x, window=2)
    acf = autocorrelation_function(x, max_lag=2)
    esm = exponential_smoothing(x, alpha=0.5)

    assert np.array_equal(d1, np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.array_equal(rm, np.array([1.5, 3.0, 5.5, 9.0]))
    assert acf.shape == (3,)
    assert np.isclose(acf[0], 1.0)
    assert esm.shape == x.shape


def test_dtw_distance_and_path_are_consistent():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 2.5, 3.0])

    path, dist = dtw_path(x, y)
    dist2 = dtw_distance(x, y)

    assert dist >= 0.0
    assert np.isclose(dist, dist2)
    assert path[0] == (0, 0)
    assert path[-1] == (2, 3)


def test_ar_model_fits_and_forecasts_linear_series():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    model = ARModel(order=1).fit(x)
    pred = model.predict(x)
    forecast = model.forecast(steps=2)

    assert pred.shape == (4,)
    assert np.allclose(pred, np.array([2.0, 3.0, 4.0, 5.0]), atol=1e-8)
    assert forecast.shape == (2,)
    assert forecast[0] > x[-1]


def test_cusum_change_point_detector_flags_shift():
    x = np.array([0.0] * 20 + [3.0] * 20)
    cps = cusum_change_points(x, threshold=2.0, drift=0.0, direction="positive")

    assert cps
    assert cps[0] >= 18


def test_autocorrelation_helpers_are_stable():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    assert np.isclose(autocorrelation(x, lag=0), 1.0)
    assert np.isclose(partial_autocorrelation(x, lag=1), autocorrelation(x, lag=1))
