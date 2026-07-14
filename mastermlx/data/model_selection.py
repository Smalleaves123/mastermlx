from __future__ import annotations

import copy
import time

import numpy as np

from ..utils.metrics import accuracy, mean_squared_error, r2_score
from ..utils.random import resolve_rng
from .cv import KFold


def _named_scorer(name):
    if name == "accuracy":
        return lambda y_true, y_pred: accuracy(y_true, y_pred)
    if name == "r2":
        return lambda y_true, y_pred: r2_score(y_true, y_pred)
    if name == "neg_mean_squared_error":
        return lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
    raise ValueError("Unsupported scoring value")


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


def _run_cv(estimator, X, y, cv=None, scoring=None, return_train_score=False, groups=None):
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")

    splitter = cv if cv is not None else KFold(n_splits=5, shuffle=True, random_state=0)
    scorers, multi = _resolve_scorers(scoring)
    test_scores = {}
    train_scores = {}
    fit_times = []
    score_times = []

    for train_idx, test_idx in _split_cv(splitter, X, y, groups=groups):
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
            pred_test = None
            pred_train = None
            for name, scorer in scorers.items():
                if pred_test is None:
                    pred_test = model.predict(X[test_idx])
                test_score = scorer(y[test_idx], pred_test)
                test_scores.setdefault(name, []).append(float(test_score))
                if return_train_score:
                    if pred_train is None:
                        pred_train = model.predict(X[train_idx])
                    train_score = scorer(y[train_idx], pred_train)
                    train_scores.setdefault(name, []).append(float(train_score))
        score_times.append(time.perf_counter() - t1)

    out = {
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
    return out


def cross_val_score(estimator, X, y, cv=None, scoring=None, groups=None):
    """Evaluate an estimator by cross-validation."""

    if isinstance(scoring, (list, tuple, dict)):
        raise ValueError("cross_val_score only supports a single scoring metric")
    return _run_cv(estimator, X, y, cv=cv, scoring=scoring, groups=groups)["test_score"]


def cross_validate(estimator, X, y, cv=None, scoring=None, return_train_score=False, groups=None):
    """Run cross-validation and return scores with timing info."""

    return _run_cv(estimator, X, y, cv=cv, scoring=scoring, return_train_score=return_train_score, groups=groups)


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

    sizes = []
    for value in train_sizes:
        if value <= 0:
            raise ValueError("train_sizes must be positive")
        if value <= 1.0:
            size = max(1, int(np.ceil(value * max_train)))
        else:
            size = int(value)
        sizes.append(min(size, max_train))
    sizes = np.unique(np.asarray(sizes, dtype=int))

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
