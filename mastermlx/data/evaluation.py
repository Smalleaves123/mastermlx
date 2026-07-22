"""Unified, leakage-aware evaluation reports for tabular estimators."""

from __future__ import annotations

import copy
from itertools import combinations
from typing import Any

import numpy as np

from ..utils.metrics import (
    accuracy,
    f1_score,
    log_loss,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from ..utils.random import resolve_rng
from .cv import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit
from .model_selection import _split_cv, cross_validate, learning_curve


def _as_array(X, y, groups=None):
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim < 1 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")
    if groups is not None:
        groups = np.asarray(groups)
        if groups.ndim != 1 or groups.shape[0] != y.shape[0]:
            raise ValueError("groups must be a 1D array with one value per sample")
    return X, y, groups


def _resolve_cv(cv, task, n_samples, groups, random_state):
    if isinstance(cv, str):
        key = cv.lower()
        if key in {"stratified", "stratifiedkfold"}:
            cv = "stratified"
        elif key in {"group", "groupkfold"}:
            cv = "group"
        elif key in {"timeseries", "time_series", "timeseriessplit"}:
            cv = "timeseries"
        elif key in {"kfold", "ordinary"}:
            cv = "kfold"
        else:
            raise ValueError("cv must be a splitter, integer, or one of: stratified, group, timeseries, kfold")

    if cv is None or isinstance(cv, (int, np.integer)):
        requested = 5 if cv is None else int(cv)
        if requested < 2:
            raise ValueError("cv must contain at least two splits")
        if groups is not None:
            n_splits = min(requested, np.unique(groups).size)
            if n_splits < 2:
                raise ValueError("at least two groups are required for grouped CV")
            return GroupKFold(n_splits=n_splits)
        n_splits = min(requested, n_samples)
        if n_splits < 2:
            raise ValueError("at least two samples are required for cross-validation")
        if task == "classification":
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if cv == "stratified":
        if task != "classification":
            raise ValueError("stratified CV requires task='classification'")
        n_splits = min(5, n_samples)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if cv == "group":
        if groups is None:
            raise ValueError("groups are required for grouped CV")
        n_splits = min(5, np.unique(groups).size)
        return GroupKFold(n_splits=n_splits)
    if cv == "timeseries":
        return TimeSeriesSplit(n_splits=5)
    if cv == "kfold":
        return KFold(n_splits=min(5, n_samples), shuffle=True, random_state=random_state)
    if not hasattr(cv, "split"):
        raise TypeError("cv must be a splitter, integer, or supported string")
    return cv


def _scoring_config(scoring, task):
    if scoring is None:
        if task == "classification":
            return ["accuracy"], "accuracy"
        return ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"], "r2"
    if isinstance(scoring, str):
        return scoring, scoring
    if callable(scoring):
        return scoring, "score"
    if isinstance(scoring, (list, tuple)):
        values = list(scoring)
        if not values or not all(isinstance(value, str) for value in values):
            raise ValueError("scoring sequences must contain metric names")
        return values, values[0]
    if isinstance(scoring, dict):
        if not scoring:
            raise ValueError("scoring must not be empty")
        return scoring, next(iter(scoring))
    raise ValueError("Unsupported scoring value")


def _learning_scoring(scoring, task, primary_metric):
    if callable(scoring):
        return scoring
    if isinstance(scoring, dict):
        return next(iter(scoring.values()))
    if scoring is not None:
        return primary_metric
    return "accuracy" if task == "classification" else "r2"


def _metric_functions(task, y, prediction, probability) -> dict[str, Any]:
    if task == "classification":
        average = "binary" if np.unique(y).size <= 2 else "macro"
        functions: dict[str, Any] = {
            "accuracy": lambda yt, yp, _: float(accuracy(yt, yp)),
            "f1": lambda yt, yp, _: float(f1_score(yt, yp, average=average)),
        }
        if probability is not None:
            functions["log_loss"] = lambda yt, _, proba: float(log_loss(yt, proba))
        return functions
    return {
        "r2": lambda yt, yp, _: float(r2_score(yt, yp)),
        "mae": lambda yt, yp, _: float(mean_absolute_error(yt, yp)),
        "rmse": lambda yt, yp, _: float(root_mean_squared_error(yt, yp)),
    }


def _metric_values(functions, y, prediction, probability):
    values = {}
    for name, function in functions.items():
        try:
            values[name] = float(function(y, prediction, probability))
        except (TypeError, ValueError, ZeroDivisionError):
            values[name] = float("nan")
    return values


def _bootstrap(functions, y, prediction, probability, n_resamples, confidence_level, random_state):
    n_resamples = int(n_resamples)
    if n_resamples < 0:
        raise ValueError("n_bootstrap must be non-negative")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if n_resamples == 0:
        return {}

    rng = resolve_rng(random_state)
    n_samples = y.shape[0]
    draws: dict[str, list[float]] = {name: [] for name in functions}
    for _ in range(n_resamples):
        indices = rng.integers(0, n_samples, size=n_samples)
        for name, function in functions.items():
            try:
                value = float(function(y[indices], prediction[indices], None if probability is None else probability[indices]))
            except (TypeError, ValueError, ZeroDivisionError):
                value = float("nan")
            if np.isfinite(value):
                draws[name].append(value)

    alpha = (1.0 - confidence_level) / 2.0
    result = {}
    for name, values in draws.items():
        finite = np.asarray(values, dtype=float)
        estimate = float(functions[name](y, prediction, probability))
        if finite.size == 0:
            lower = upper = float("nan")
        else:
            lower, upper = np.quantile(finite, [alpha, 1.0 - alpha])
        result[name] = {
            "estimate": estimate,
            "lower": float(lower),
            "upper": float(upper),
            "confidence_level": float(confidence_level),
            "n_resamples": int(finite.size),
        }
    return result


def _curve_summary(sizes, train_scores, test_scores):
    return {
        "train_sizes": np.asarray(sizes, dtype=int).tolist(),
        "train_scores": np.asarray(train_scores, dtype=float).tolist(),
        "test_scores": np.asarray(test_scores, dtype=float).tolist(),
        "train_mean": np.mean(train_scores, axis=1).astype(float).tolist(),
        "train_std": np.std(train_scores, axis=1).astype(float).tolist(),
        "test_mean": np.mean(test_scores, axis=1).astype(float).tolist(),
        "test_std": np.std(test_scores, axis=1).astype(float).tolist(),
    }


def _partial_oof_predict(estimator, X, y, splitter, groups=None, method="predict"):
    """Generate OOF predictions while allowing chronological warm-up samples."""

    out = None
    seen = np.zeros(X.shape[0], dtype=bool)
    for train_idx, test_idx in _split_cv(splitter, X, y, groups=groups):
        model = copy.deepcopy(estimator)
        model.fit(X[train_idx], y[train_idx])
        if not hasattr(model, method):
            raise AttributeError(f"Estimator does not define method '{method}'")
        prediction = np.asarray(getattr(model, method)(X[test_idx]))
        if out is None:
            if prediction.ndim == 1:
                if np.issubdtype(prediction.dtype, np.number):
                    out = np.full(X.shape[0], np.nan, dtype=float)
                else:
                    out = np.empty(X.shape[0], dtype=object)
                    out[:] = None
            else:
                out = np.full((X.shape[0],) + prediction.shape[1:], np.nan, dtype=float)
        if np.any(seen[test_idx]):
            raise ValueError("Cross-validator assigned a sample to more than one test fold")
        out[test_idx] = prediction
        seen[test_idx] = True
    if not np.any(seen):
        raise ValueError("Cross-validator did not assign any OOF predictions")
    return out, seen


def _to_report_list(values):
    array = np.asarray(values)
    if array.ndim == 0:
        value = array.item()
        return None if value is None or (isinstance(value, float) and not np.isfinite(value)) else value
    return [_to_report_list(value) for value in array]


class EvaluationReport:
    """Build one reproducible report for OOF, CV, uncertainty, and learning curves."""

    def __init__(self, estimator, *, task="classification", scoring=None, random_state=0):
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        self.estimator = estimator
        self.task = task
        self.scoring = scoring
        self.random_state = random_state

    def run(
        self,
        X,
        y,
        *,
        cv=None,
        groups=None,
        n_bootstrap=1000,
        confidence_level=0.95,
        train_sizes=None,
        include_learning_curve=True,
    ):
        """Run all report sections using the same cross-validation definition."""

        X, y, groups = _as_array(X, y, groups)
        splitter = _resolve_cv(cv, self.task, X.shape[0], groups, self.random_state)
        effective_scoring, primary_metric = _scoring_config(self.scoring, self.task)

        cv_result = cross_validate(
            self.estimator,
            X,
            y,
            cv=splitter,
            scoring=effective_scoring,
            return_train_score=True,
            groups=groups,
        )
        test_scores = {}
        train_scores = {}
        for key, values in cv_result.items():
            if key.startswith("test_"):
                name = primary_metric if key == "test_score" else key[5:]
                test_scores[name] = np.asarray(values, dtype=float).tolist()
            elif key.startswith("train_"):
                name = primary_metric if key == "train_score" else key[6:]
                train_scores[name] = np.asarray(values, dtype=float).tolist()

        prediction, seen = _partial_oof_predict(self.estimator, X, y, splitter, groups=groups)
        probability = None
        if self.task == "classification" and hasattr(self.estimator, "predict_proba"):
            probability, probability_seen = _partial_oof_predict(
                self.estimator, X, y, splitter, groups=groups, method="predict_proba"
            )
            if not np.array_equal(seen, probability_seen):
                raise ValueError("predict and predict_proba did not produce matching OOF coverage")

        metric_y = y[seen]
        metric_prediction = prediction[seen]
        metric_probability = None if probability is None else probability[seen]
        functions = _metric_functions(self.task, metric_y, metric_prediction, metric_probability)
        oof_metrics = _metric_values(functions, metric_y, metric_prediction, metric_probability)
        if callable(self.scoring):
            oof_metrics[primary_metric] = float(self.scoring(metric_y, metric_prediction))

        curve = None
        if include_learning_curve:
            curve_scoring = _learning_scoring(self.scoring, self.task, primary_metric)
            sizes, curve_train, curve_test = learning_curve(
                self.estimator,
                X,
                y,
                train_sizes=train_sizes,
                cv=splitter,
                scoring=curve_scoring,
                groups=groups,
                shuffle=False,
                random_state=self.random_state,
            )
            curve = _curve_summary(sizes, curve_train, curve_test)

        return {
            "task": self.task,
            "primary_metric": primary_metric,
            "cv": {
                "name": splitter.__class__.__name__,
                "n_splits": int(len(next(iter(test_scores.values())))) if test_scores else 0,
                "test_scores": test_scores,
                "train_scores": train_scores,
                "fit_time": np.asarray(cv_result["fit_time"], dtype=float).tolist(),
                "score_time": np.asarray(cv_result["score_time"], dtype=float).tolist(),
            },
            "oof": {
                "prediction": _to_report_list(prediction),
                "probability": None if probability is None else _to_report_list(probability),
                "covered_indices": np.flatnonzero(seen).astype(int).tolist(),
                "uncovered_indices": np.flatnonzero(~seen).astype(int).tolist(),
                "metrics": oof_metrics,
            },
            "bootstrap": _bootstrap(
                functions,
                metric_y,
                metric_prediction,
                metric_probability,
                n_bootstrap,
                confidence_level,
                self.random_state,
            ),
            "learning_curve": curve,
        }


def _paired_significance(left, right, n_permutations=2000, random_state=0):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError("paired model scores must be 1D arrays with the same length and at least two folds")
    differences = left - right
    observed = float(np.mean(differences))
    rng = resolve_rng(random_state)
    permutations = int(n_permutations)
    if permutations < 1:
        raise ValueError("n_permutations must be positive")
    signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, differences.size))
    null = np.mean(signs * differences[None, :], axis=1)
    p_value = float((1.0 + np.sum(np.abs(null) >= abs(observed))) / (permutations + 1.0))
    return {
        "mean_difference": observed,
        "p_value": p_value,
        "significant_at_0_05": bool(p_value < 0.05),
        "n_permutations": permutations,
    }


def compare_estimators(
    models,
    X,
    y,
    *,
    task="classification",
    scoring=None,
    cv=None,
    groups=None,
    random_state=0,
    n_bootstrap=0,
    n_permutations=2000,
):
    """Compare models on identical folds with paired significance tests."""

    if not models:
        raise ValueError("models must be non-empty")
    reports = {}
    for name, estimator in models:
        reports[str(name)] = EvaluationReport(
            copy.deepcopy(estimator), task=task, scoring=scoring, random_state=random_state
        ).run(
            X,
            y,
            cv=cv,
            groups=groups,
            n_bootstrap=n_bootstrap,
            include_learning_curve=False,
        )

    primary = next(iter(reports.values()))["primary_metric"]
    leaderboard = []
    for name, report in reports.items():
        scores = np.asarray(report["cv"]["test_scores"][primary], dtype=float)
        leaderboard.append(
            {
                "name": name,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "oof_metric": report["oof"]["metrics"].get(primary),
            }
        )
    leaderboard.sort(key=lambda item: item["mean"], reverse=True)

    pairs = []
    rng = resolve_rng(random_state)
    for left_name, right_name in combinations(reports, 2):
        left_scores = reports[left_name]["cv"]["test_scores"][primary]
        right_scores = reports[right_name]["cv"]["test_scores"][primary]
        pairs.append(
            {
                "model_a": left_name,
                "model_b": right_name,
                "metric": primary,
                **_paired_significance(
                    left_scores,
                    right_scores,
                    n_permutations=n_permutations,
                    random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
                ),
            }
        )
    return {"primary_metric": primary, "leaderboard": leaderboard, "pairwise": pairs, "reports": reports}


__all__ = ["EvaluationReport", "compare_estimators"]
