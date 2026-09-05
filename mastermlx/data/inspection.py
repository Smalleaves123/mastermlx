"""Model-agnostic inspection helpers for fitted estimators."""

from __future__ import annotations

import numbers
from typing import Any, cast

import numpy as np

from ..base.results import BaseReport
from ..utils.random import resolve_rng
from .model_selection import _single_scorer


def _score_estimator(estimator, X, y, scoring):
    scorers = _single_scorer(scoring)
    if scorers is None:
        if not callable(getattr(estimator, "score", None)):
            raise AttributeError("estimator does not define score() and no scoring was provided")
        return float(estimator.score(X, y))
    scorer = scorers["score"]
    method = getattr(scorer, "method", "predict")
    if not callable(getattr(estimator, method, None)):
        raise AttributeError(f"estimator does not define {method}() required by scoring")
    return float(scorer(y, getattr(estimator, method)(X)))


def permutation_importance(
    estimator,
    X,
    y,
    *,
    scoring=None,
    n_repeats=5,
    random_state=None,
    feature_names=None,
):
    """Measure score decrease after independently permuting each feature.

    The estimator must already be fitted. Scores follow the library convention
    that larger is better, including negative loss scorers, so positive values
    consistently indicate useful features.
    """

    X_array = np.asarray(X)
    y_array = np.asarray(y)
    if X_array.ndim != 2 or X_array.shape[0] == 0 or X_array.shape[1] == 0:
        raise ValueError("X must be a non-empty 2D feature matrix")
    if y_array.ndim != 1 or y_array.shape[0] != X_array.shape[0]:
        raise ValueError("y must be 1D with one value per sample")
    n_repeats = int(n_repeats)
    if n_repeats < 1:
        raise ValueError("n_repeats must be positive")

    if feature_names is None:
        columns = getattr(X, "columns", None)
        feature_names = (
            [f"x{index}" for index in range(X_array.shape[1])]
            if columns is None
            else [str(name) for name in columns]
        )
    else:
        feature_names = [str(name) for name in feature_names]
    if len(feature_names) != X_array.shape[1] or len(feature_names) != len(set(feature_names)):
        raise ValueError("feature_names must be unique with one name per feature")

    baseline = _score_estimator(estimator, X_array, y_array, scoring)
    rng = resolve_rng(random_state)
    importances = np.empty((X_array.shape[1], n_repeats), dtype=float)
    permuted = np.array(X_array, copy=True)
    for feature_index in range(X_array.shape[1]):
        original = np.array(X_array[:, feature_index], copy=True)
        for repeat in range(n_repeats):
            permuted[:, feature_index] = original[rng.permutation(X_array.shape[0])]
            score = _score_estimator(estimator, permuted, y_array, scoring)
            importances[feature_index, repeat] = baseline - score
        permuted[:, feature_index] = original

    means = np.mean(importances, axis=1)
    stds = np.std(importances, axis=1)
    order = np.argsort(-means, kind="stable")
    return BaseReport(
        {
            "kind": "permutation",
            "scoring": "estimator.score" if scoring is None else str(scoring),
            "baseline_score": baseline,
            "n_repeats": n_repeats,
            "feature_names": list(feature_names),
            "importances": importances,
            "importances_mean": means,
            "importances_std": stds,
            "items": [
                {
                    "feature": feature_names[index],
                    "importance": float(means[index]),
                    "std": float(stds[index]),
                }
                for index in order
            ],
        }
    )


def _resolve_feature(X, feature):
    X_array = np.asarray(X)
    columns = getattr(X, "columns", None)
    if isinstance(feature, str):
        if columns is None:
            raise TypeError("string feature names require X.columns")
        names = [str(name) for name in columns]
        if feature not in names:
            raise ValueError(f"unknown feature {feature!r}")
        return X_array, names.index(feature), feature
    index = int(feature)
    if index < 0 or index >= X_array.shape[1]:
        raise ValueError(f"feature index must be in [0, {X_array.shape[1]})")
    name = str(columns[index]) if columns is not None else f"x{index}"
    return X_array, index, name


def _unique_in_order(values):
    unique = []
    keys = set()
    for value in values:
        item = value.item() if isinstance(value, np.generic) else value
        try:
            key = (type(item).__name__, item)
            is_new = key not in keys
        except TypeError:
            key = (type(item).__name__, repr(item))
            is_new = key not in keys
        if is_new:
            keys.add(key)
            unique.append(item)
    return unique


def _is_missing_number(value):
    if value is None:
        return True
    if not isinstance(value, numbers.Number):
        return False
    try:
        return bool(np.isnan(cast(Any, value)))
    except TypeError:
        return False


def _feature_grid(column, grid_values, grid_resolution, percentiles):
    if grid_values is not None:
        values = np.asarray(list(grid_values))
        if values.ndim != 1 or values.size == 0:
            raise ValueError("grid_values must be a non-empty 1D sequence")
        return values

    grid_resolution = int(grid_resolution)
    if grid_resolution < 2:
        raise ValueError("grid_resolution must be at least 2")
    low, high = tuple(float(value) for value in percentiles)
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 1")

    raw_values = [value.item() if isinstance(value, np.generic) else value for value in column]
    finite_values = [
        value
        for value in raw_values
        if not _is_missing_number(value)
    ]
    is_numeric = bool(finite_values) and all(
        isinstance(value, numbers.Number) and not isinstance(value, (bool, np.bool_))
        for value in finite_values
    )
    if not is_numeric:
        unique = _unique_in_order(column)
        if len(unique) > grid_resolution:
            raise ValueError("categorical feature has more values than grid_resolution")
        return np.asarray(unique, dtype=object)

    numeric = np.asarray(finite_values, dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("numeric feature values must be finite")

    unique = np.unique(numeric)
    if unique.size <= grid_resolution:
        return unique
    bounds = np.asarray(np.quantile(numeric, [low, high]), dtype=float).ravel()
    start, stop = float(bounds[0]), float(bounds[1])
    if start == stop:
        return np.asarray([start])
    return np.linspace(start, stop, grid_resolution)


def _prediction_response(estimator, X, response_method, target):
    method = response_method
    if method == "auto":
        if callable(getattr(estimator, "predict_proba", None)):
            try:
                response = np.asarray(estimator.predict_proba(X))
                method = "predict_proba"
            except AttributeError:
                method = "predict"
                response = np.asarray(estimator.predict(X))
        else:
            method = "predict"
            response = np.asarray(estimator.predict(X))
    else:
        response = None
    if method not in {"predict", "predict_proba", "decision_function"}:
        raise ValueError("response_method must be 'auto', 'predict', 'predict_proba', or 'decision_function'")
    if not callable(getattr(estimator, method, None)):
        raise AttributeError(f"estimator does not define {method}()")
    if response is None:
        response = np.asarray(getattr(estimator, method)(X))
    if response.ndim == 1:
        if target is not None:
            raise ValueError("target is only valid for multi-output responses")
        return np.asarray(response, dtype=float), method, None
    if response.ndim != 2:
        raise ValueError("estimator response must be 1D or 2D")
    if target is None:
        if response.shape[1] == 1:
            target = 0
        elif response.shape[1] == 2:
            target = 1
        else:
            raise ValueError("target is required for multiclass or multi-output responses")
    target = int(target)
    if target < 0 or target >= response.shape[1]:
        raise ValueError(f"target must be in [0, {response.shape[1]})")
    return np.asarray(response[:, target], dtype=float), method, target


def partial_dependence(
    estimator,
    X,
    feature,
    *,
    grid_values=None,
    grid_resolution=20,
    percentiles=(0.05, 0.95),
    kind="average",
    response_method="auto",
    target=None,
    center=False,
):
    """Compute one-feature partial dependence and optional ICE curves.

    ``kind="average"`` returns only the population mean, ``"individual"``
    returns ICE curves, and ``"both"`` includes both representations.
    """

    X_array = np.asarray(X)
    if X_array.ndim != 2 or X_array.shape[0] == 0 or X_array.shape[1] == 0:
        raise ValueError("X must be a non-empty 2D feature matrix")
    if kind not in {"average", "individual", "both"}:
        raise ValueError("kind must be 'average', 'individual', or 'both'")
    X_array, feature_index, feature_name = _resolve_feature(X, feature)
    values = _feature_grid(
        X_array[:, feature_index], grid_values, grid_resolution, percentiles
    )
    working = np.array(
        X_array,
        dtype=float if np.issubdtype(X_array.dtype, np.number) else object,
        copy=True,
    )
    curves = np.empty((values.size, X_array.shape[0]), dtype=float)
    resolved_method = response_method
    resolved_target = target
    for grid_index, value in enumerate(values):
        working[:, feature_index] = value
        curves[grid_index], resolved_method, resolved_target = _prediction_response(
            estimator, working, resolved_method, resolved_target
        )
    if center:
        curves = curves - curves[0]
    average = np.mean(curves, axis=1)
    result = BaseReport(
        {
            "kind": kind,
            "feature": feature_name,
            "feature_index": feature_index,
            "grid_values": values,
            "response_method": resolved_method,
            "target": resolved_target,
            "centered": bool(center),
        }
    )
    if kind in {"average", "both"}:
        result["average"] = average
    if kind in {"individual", "both"}:
        result["individual"] = curves
    return result


__all__ = ["partial_dependence", "permutation_importance"]
