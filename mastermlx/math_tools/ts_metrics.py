from __future__ import annotations

import numpy as np

from ..utils.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


def _arrays(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must be non-empty arrays with the same shape")
    return y_true, y_pred


class ForecastMetrics:
    """Common point-forecast metrics with one compact report method."""

    @staticmethod
    def mae(y_true, y_pred):
        return mean_absolute_error(*_arrays(y_true, y_pred))

    @staticmethod
    def mse(y_true, y_pred):
        return float(mean_squared_error(*_arrays(y_true, y_pred)))

    @staticmethod
    def rmse(y_true, y_pred):
        return float(root_mean_squared_error(*_arrays(y_true, y_pred)))

    @staticmethod
    def mape(y_true, y_pred):
        return mean_absolute_percentage_error(*_arrays(y_true, y_pred))

    @staticmethod
    def smape(y_true, y_pred, eps=1e-12):
        y_true, y_pred = _arrays(y_true, y_pred)
        denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
        return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))

    @staticmethod
    def r2(y_true, y_pred):
        return float(r2_score(*_arrays(y_true, y_pred)))

    @classmethod
    def report(cls, y_true, y_pred):
        return {
            "mae": cls.mae(y_true, y_pred),
            "mse": cls.mse(y_true, y_pred),
            "rmse": cls.rmse(y_true, y_pred),
            "mape": cls.mape(y_true, y_pred),
            "smape": cls.smape(y_true, y_pred),
            "r2": cls.r2(y_true, y_pred),
        }


__all__ = ["ForecastMetrics"]
