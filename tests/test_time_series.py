import numpy as np

from mastermlx import get_backend, set_backend
from mastermlx.math_tools import (
    ARModel,
    autocorrelation,
    autocorrelation_function,
    backtest,
    cusum_change_points,
    compare_time_series_models,
    dtw_distance,
    dtw_path,
    difference,
    exponential_smoothing,
    lagged_matrix,
    TimeSeriesExperiment,
    TimeSeriesPipeline,
    ForecastMetrics,
    partial_autocorrelation,
    rolling_mean,
    rolling_backtest,
)
from mastermlx.linear_models import LinearRegression


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


def test_dtw_distance_rejects_negative_window_and_matches_path():
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.8, 0.1, 1.0])

    with np.testing.assert_raises(ValueError):
        dtw_distance(x, y, window=-1)
    path_distance = dtw_path(x, y, window=1)[1]
    assert np.isclose(dtw_distance(x, y, window=1), path_distance)


def test_dtw_narrow_window_matches_numpy_and_compiled_paths():
    x = np.array([0.0, 3.0, 1.0, 4.0, 2.0])
    y = np.array([0.5, 2.5, 1.5, 3.5, 2.5])
    old = get_backend()
    try:
        set_backend("numpy")
        numpy_dist = dtw_distance(x, y, window=1)
        numpy_path, numpy_path_dist = dtw_path(x, y, window=1)
        set_backend("auto")
        compiled_dist = dtw_distance(x, y, window=1)
        compiled_path, compiled_path_dist = dtw_path(x, y, window=1)
    finally:
        set_backend(old)

    assert np.isclose(numpy_dist, numpy_path_dist)
    assert np.isclose(compiled_dist, compiled_path_dist)
    assert np.isclose(numpy_dist, compiled_dist)
    assert numpy_path[0] == compiled_path[0] == (0, 0)
    assert numpy_path[-1] == compiled_path[-1] == (4, 4)


def test_dtw_rejects_non_finite_values():
    with np.testing.assert_raises(ValueError):
        dtw_distance([0.0, np.nan], [0.0, 1.0])


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


def test_lagged_matrix_builds_supervised_windows():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    X, y = lagged_matrix(x, lags=2, horizon=1)

    assert X.shape == (3, 2)
    assert np.array_equal(X, np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]))
    assert np.array_equal(y, np.array([3.0, 4.0, 5.0]))


def test_time_series_pipeline_fits_and_forecasts():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    pipeline = TimeSeriesPipeline(
        model=LinearRegression(fit_intercept=True),
        lags=2,
        horizon=1,
    )
    pipeline.fit(x)
    pred = pipeline.predict(x)
    forecast = pipeline.forecast(steps=2)

    assert pred.shape == (4,)
    assert np.allclose(pred, np.array([3.0, 4.0, 5.0, 6.0]), atol=1e-8)
    assert np.allclose(forecast, np.array([7.0, 8.0]), atol=1e-8)
    assert pipeline.score(x) > 0.99


def test_time_series_experiment_searches_and_refits():
    x = np.arange(1.0, 13.0)

    exp = TimeSeriesExperiment(
        model=LinearRegression(fit_intercept=False),
        lags=2,
        search="grid",
        param_grid={"model__fit_intercept": [True, False]},
        cv=3,
        scoring="r2",
        random_state=0,
    )
    exp.fit(x)
    pred = exp.forecast(steps=2, history=x)

    assert exp.best_params_ is not None
    assert exp.best_score_ is not None
    assert exp.summary()["lags"] == 2
    assert pred.shape == (2,)
    assert np.allclose(pred, np.array([13.0, 14.0]), atol=1e-8)
    assert exp.score(x[:10], x[10:]) > 0.99


def test_compare_time_series_models_returns_leaderboard():
    x = np.arange(1.0, 13.0)

    out = compare_time_series_models(
        [
            ("lr_a", LinearRegression(fit_intercept=True)),
            ("lr_b", LinearRegression(fit_intercept=False)),
        ],
        x,
        lags=2,
        scoring="r2",
    )

    assert out["leaderboard"]
    assert out["best_name"] in {"lr_a", "lr_b"}
    assert out["best_experiment"] is not None
    assert out["best_score"] >= out["leaderboard"][-1][1]


def test_backtest_returns_fold_scores_and_predictions():
    x = np.arange(1.0, 21.0)

    out = backtest(
        LinearRegression(fit_intercept=False),
        x,
        lags=2,
        cv=3,
        scoring="r2",
    )

    assert out["scores"].shape == (3,)
    assert out["true"].shape == out["pred"].shape
    assert len(out["folds"]) == 3
    assert out["mean"] > 0.99
    assert np.allclose(out["true"], out["pred"], atol=1e-8)


def test_rolling_backtest_and_forecast_metrics():
    x = np.arange(1.0, 16.0)
    out = rolling_backtest(
        LinearRegression(fit_intercept=True),
        x,
        lags=1,
        test_size=2,
        step=2,
        scoring="r2",
    )
    report = ForecastMetrics.report(out["true"], out["pred"])

    assert len(out["folds"]) > 1
    assert out["metrics"] == report
    assert report["mae"] < 1e-8
    assert report["rmse"] < 1e-8
