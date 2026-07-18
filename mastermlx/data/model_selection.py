from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

from ..utils.metrics import (
    accuracy,
    explained_variance_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from ..utils.random import resolve_rng
from .cv import KFold


class _Scorer:
    def __init__(self, func, method="predict"):
        self.func = func
        self.method = method

    def __call__(self, y_true, y_pred):
        return self.func(y_true, y_pred)


def _positive_score(y_true, y_pred):
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 2:
        if y_pred.shape[1] != 2:
            raise ValueError("roc_auc scoring requires binary probabilities")
        y_pred = y_pred[:, 1]
    return roc_auc_score(y_true, y_pred)


def _named_scorer(name):
    scorers = {
        "accuracy": _Scorer(accuracy),
        "precision": _Scorer(precision_score),
        "recall": _Scorer(recall_score),
        "f1": _Scorer(f1_score),
        "r2": _Scorer(r2_score),
        "explained_variance": _Scorer(explained_variance_score),
        "neg_mean_absolute_error": _Scorer(lambda yt, yp: -mean_absolute_error(yt, yp)),
        "neg_mean_squared_error": _Scorer(lambda yt, yp: -mean_squared_error(yt, yp)),
        "neg_root_mean_squared_error": _Scorer(lambda yt, yp: -root_mean_squared_error(yt, yp)),
        "neg_log_loss": _Scorer(lambda yt, yp: -log_loss(yt, yp), method="predict_proba"),
        "roc_auc": _Scorer(_positive_score, method="predict_proba"),
    }
    if name not in scorers:
        raise ValueError(f"Unsupported scoring value: {name}")
    return scorers[name]


def _resolve_scorers(scoring):
    if scoring is None:
        return None, False
    if callable(scoring):
        return {"score": scoring}, False
    if isinstance(scoring, str):
        return {"score": _named_scorer(scoring)}, False
    if isinstance(scoring, (list, tuple)):
        out = {}
        for item in scoring:
            if not isinstance(item, str):
                raise ValueError("scoring lists must contain only string metric names")
            out[item] = _named_scorer(item)
        return out, True
    if isinstance(scoring, dict):
        out = {}
        for name, scorer in scoring.items():
            out[name] = scorer if callable(scorer) else _named_scorer(scorer)
        return out, True
    raise ValueError("Unsupported scoring value")


def _split_cv(splitter, X, y, groups=None):
    if isinstance(splitter, int):
        splitter = KFold(n_splits=int(splitter), shuffle=True, random_state=0)
    try:
        return splitter.split(X, y, groups=groups)
    except TypeError:
        return splitter.split(X, y)


def _single_scorer(scoring):
    if isinstance(scoring, (list, tuple, dict)):
        raise ValueError("This function only supports a single scoring metric")
    scorers, _ = _resolve_scorers(scoring)
    return scorers


def _error_value(error_score):
    if error_score == "raise":
        return None
    try:
        return float(error_score)
    except (TypeError, ValueError) as exc:
        raise ValueError("error_score must be 'raise' or a numeric value") from exc


def _run_cv(
    estimator,
    X,
    y,
    cv=None,
    scoring=None,
    return_train_score=False,
    groups=None,
    error_score="raise",
    return_estimator=False,
):
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")

    splitter = cv if cv is not None else KFold(n_splits=5, shuffle=True, random_state=0)
    scorers, multi = _resolve_scorers(scoring)
    error_value = _error_value(error_score)
    test_scores: dict[str, list[float]] = {}
    train_scores: dict[str, list[float]] = {}
    fit_times = []
    score_times = []
    estimators = []
    errors: list[str | None] = []

    for train_idx, test_idx in _split_cv(splitter, X, y, groups=groups):
        model = None
        try:
            model = copy.deepcopy(estimator)
            t0 = time.perf_counter()
            model.fit(X[train_idx], y[train_idx])
            fit_times.append(time.perf_counter() - t0)

            t1 = time.perf_counter()
            if scorers is None:
                test_score = model.score(X[test_idx], y[test_idx])
                test_scores.setdefault("score", []).append(float(test_score))
                if return_train_score:
                    train_score = model.score(X[train_idx], y[train_idx])
                    train_scores.setdefault("score", []).append(float(train_score))
            else:
                test_pred = {}
                train_pred = {}
                for name, scorer in scorers.items():
                    method = getattr(scorer, "method", "predict")
                    if method not in test_pred:
                        test_pred[method] = getattr(model, method)(X[test_idx])
                    test_score = scorer(y[test_idx], test_pred[method])
                    test_scores.setdefault(name, []).append(float(test_score))
                    if return_train_score:
                        if method not in train_pred:
                            train_pred[method] = getattr(model, method)(X[train_idx])
                        train_score = scorer(y[train_idx], train_pred[method])
                        train_scores.setdefault(name, []).append(float(train_score))
            score_times.append(time.perf_counter() - t1)
            errors.append(None)
        except Exception as exc:
            if error_value is None:
                raise
            fit_times.append(np.nan)
            score_times.append(np.nan)
            errors.append(f"{type(exc).__name__}: {exc}")
            if scorers is None:
                test_scores.setdefault("score", []).append(error_value)
                if return_train_score:
                    train_scores.setdefault("score", []).append(error_value)
            else:
                for name in scorers:
                    test_scores.setdefault(name, []).append(error_value)
                    if return_train_score:
                        train_scores.setdefault(name, []).append(error_value)
        if return_estimator:
            estimators.append(model)

    out: dict[str, Any] = {
        "fit_time": np.asarray(fit_times, dtype=float),
        "score_time": np.asarray(score_times, dtype=float),
    }
    for name, values in test_scores.items():
        key = "test_score" if name == "score" and not multi else f"test_{name}"
        out[key] = np.asarray(values, dtype=float)
    if return_train_score:
        for name, values in train_scores.items():
            key = "train_score" if name == "score" and not multi else f"train_{name}"
            out[key] = np.asarray(values, dtype=float)
    if return_estimator:
        out["estimator"] = estimators
    if any(error is not None for error in errors):
        out["errors"] = errors
    return out


def cross_val_score(estimator, X, y, cv=None, scoring=None, groups=None, error_score="raise"):
    """Evaluate an estimator by cross-validation."""

    if isinstance(scoring, (list, tuple, dict)):
        raise ValueError("cross_val_score only supports a single scoring metric")
    return _run_cv(estimator, X, y, cv=cv, scoring=scoring, groups=groups, error_score=error_score)["test_score"]


def cross_validate(
    estimator,
    X,
    y,
    cv=None,
    scoring=None,
    return_train_score=False,
    groups=None,
    error_score="raise",
    return_estimator=False,
):
    """Run cross-validation and return scores with timing info."""

    return _run_cv(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=return_train_score,
        groups=groups,
        error_score=error_score,
        return_estimator=return_estimator,
    )


def cross_val_predict(estimator, X, y, cv=None, groups=None, method="predict"):
    """Generate out-of-fold predictions for each sample."""

    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")

    splitter = cv if cv is not None else KFold(n_splits=5, shuffle=True, random_state=0)
    out = None
    seen = np.zeros(X.shape[0], dtype=bool)

    for train_idx, test_idx in _split_cv(splitter, X, y, groups=groups):
        model = copy.deepcopy(estimator)
        model.fit(X[train_idx], y[train_idx])
        if not hasattr(model, method):
            raise AttributeError(f"Estimator does not define method '{method}'")
        pred = getattr(model, method)(X[test_idx])
        pred = np.asarray(pred)

        if out is None:
            if pred.ndim == 1:
                out = np.empty(X.shape[0], dtype=pred.dtype)
            else:
                out = np.empty((X.shape[0],) + pred.shape[1:], dtype=pred.dtype)

        out[test_idx] = pred
        seen[test_idx] = True

    if not np.all(seen):
        raise ValueError("Cross-validator did not assign predictions to every sample")
    return out


def learning_curve(estimator, X, y, train_sizes=None, cv=None, scoring=None, shuffle=False, random_state=None, groups=None):
    """Compute learning curves for different training set sizes."""

    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes = np.asarray(train_sizes, dtype=float)
    if train_sizes.ndim != 1 or train_sizes.size == 0:
        raise ValueError("train_sizes must be a non-empty 1D sequence")

    splitter = cv if cv is not None else KFold(n_splits=5, shuffle=True, random_state=0)
    scorer = _single_scorer(scoring)

    splits = list(_split_cv(splitter, X, y, groups=groups))
    max_train = min(len(train_idx) for train_idx, _ in splits)
    if max_train < 1:
        raise ValueError("Cross-validator produced an empty training split")

    sizes_list: list[int] = []
    for value in train_sizes:
        if value <= 0:
            raise ValueError("train_sizes must be positive")
        if value <= 1.0:
            size = max(1, int(np.ceil(value * max_train)))
        else:
            size = int(value)
        sizes_list.append(min(size, max_train))
    sizes = np.unique(np.asarray(sizes_list, dtype=int))

    train_scores = np.zeros((sizes.shape[0], len(splits)), dtype=float)
    test_scores = np.zeros((sizes.shape[0], len(splits)), dtype=float)
    rng = resolve_rng(random_state)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        base_train_idx = np.asarray(train_idx, dtype=int)
        if shuffle:
            base_train_idx = rng.permutation(base_train_idx)
        for size_idx, size in enumerate(sizes):
            sub_train = base_train_idx[:size]
            model = copy.deepcopy(estimator)
            model.fit(X[sub_train], y[sub_train])

            if scorer is None:
                train_score = model.score(X[sub_train], y[sub_train])
                test_score = model.score(X[test_idx], y[test_idx])
            else:
                pred_train = model.predict(X[sub_train])
                pred_test = model.predict(X[test_idx])
                fn = scorer["score"]
                train_score = fn(y[sub_train], pred_train)
                test_score = fn(y[test_idx], pred_test)

            train_scores[size_idx, fold_idx] = float(train_score)
            test_scores[size_idx, fold_idx] = float(test_score)

    return sizes, train_scores, test_scores


def validation_curve(
    estimator,
    X,
    y,
    param_name,
    param_range,
    cv=None,
    scoring=None,
    groups=None,
):
    """Compute train and test scores for different values of one parameter."""

    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")

    values = list(param_range)
    if not values:
        raise ValueError("param_range must be non-empty")

    splitter = cv if cv is not None else KFold(n_splits=5, shuffle=True, random_state=0)
    scorer = _single_scorer(scoring)
    splits = list(_split_cv(splitter, X, y, groups=groups))

    train_scores = np.zeros((len(values), len(splits)), dtype=float)
    test_scores = np.zeros((len(values), len(splits)), dtype=float)

    for value_idx, value in enumerate(values):
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            model = copy.deepcopy(estimator)
            if hasattr(model, "set_params"):
                model.set_params(**{param_name: value})
            else:
                setattr(model, param_name, value)
            model.fit(X[train_idx], y[train_idx])

            if scorer is None:
                train_score = model.score(X[train_idx], y[train_idx])
                test_score = model.score(X[test_idx], y[test_idx])
            else:
                pred_train = model.predict(X[train_idx])
                pred_test = model.predict(X[test_idx])
                fn = scorer["score"]
                train_score = fn(y[train_idx], pred_train)
                test_score = fn(y[test_idx], pred_test)

            train_scores[value_idx, fold_idx] = float(train_score)
            test_scores[value_idx, fold_idx] = float(test_score)

    return np.asarray(values, dtype=object), train_scores, test_scores
