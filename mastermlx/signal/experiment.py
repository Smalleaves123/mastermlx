from __future__ import annotations

import numpy as np

from ..data.search import GridSearchCV, RandomizedSearchCV
from ..preprocessing import Pipeline as PreprocessingPipeline
from ..utils.estimator import clone
from .transforms import SignalPipeline


def _normalize_signal_steps(signal_transform):
    if signal_transform is None:
        return []
    if isinstance(signal_transform, SignalPipeline):
        return list(signal_transform.steps)
    if isinstance(signal_transform, (list, tuple)):
        if not signal_transform:
            raise ValueError("signal_transform steps must be non-empty")
        first = signal_transform[0]
        if isinstance(first, tuple) and len(first) == 2:
            return list(signal_transform)
        return [(f"signal_{idx}", step) for idx, step in enumerate(signal_transform)]
    if hasattr(signal_transform, "fit") and hasattr(signal_transform, "transform"):
        return [("signal", signal_transform)]
    if callable(signal_transform):
        return [("signal", _CallableSignalTransform(signal_transform))]
    raise TypeError(
        "signal_transform must be None, a callable, a transformer, a signal pipeline, or a list of steps"
    )


def _as_signal_list(X):
    if isinstance(X, np.ndarray):
        if X.dtype == object:
            return [np.asarray(item, dtype=float).ravel() for item in X]
        if X.ndim == 1:
            return [np.asarray(X, dtype=float).ravel()]
        if X.ndim == 2:
            return [np.asarray(row, dtype=float).ravel() for row in X]
        raise ValueError("X must be a 1D signal or a batch of 1D signals")

    if isinstance(X, (list, tuple)):
        if not X:
            return []
        if all(np.isscalar(item) for item in X):
            return [np.asarray(X, dtype=float).ravel()]
        return [np.asarray(item, dtype=float).ravel() for item in X]

    arr = np.asarray(X)
    if arr.ndim == 0:
        return [np.asarray([arr.item()], dtype=float)]
    if arr.ndim == 1:
        return [arr.astype(float).ravel()]
    if arr.ndim == 2:
        return [np.asarray(row, dtype=float).ravel() for row in arr]
    raise ValueError("X must be a 1D signal or a batch of 1D signals")


def _first_signal(X):
    signals = _as_signal_list(X)
    if not signals:
        raise ValueError("X must be non-empty")
    return signals[0]


def _prepare_signal_batch_input(X):
    if isinstance(X, np.ndarray):
        return X
    if isinstance(X, (list, tuple)):
        if not X:
            return np.asarray(X)
        if all(np.isscalar(item) for item in X):
            return np.asarray(X, dtype=float)
        return np.asarray(X, dtype=object)
    return X


def _pool_features(features, feature_aggregator=None, flatten_output=True):
    arr = np.asarray(features, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if feature_aggregator is not None:
        pooled = feature_aggregator(arr)
        pooled = np.asarray(pooled, dtype=float)
        return pooled.ravel()
    if flatten_output:
        return arr.ravel()
    raise ValueError("signal transform produced a 2D feature map; set flatten_output=True or provide feature_aggregator")


class _CallableSignalTransform:
    def __init__(self, fn):
        self.fn = fn

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.fn(X)

    def get_params(self, deep=True):
        return {"fn": self.fn}

    def set_params(self, **params):
        if "fn" in params:
            self.fn = params["fn"]
        return self


class SignalFeatureTransformer:
    """Convert raw 1D signals into a fixed-width feature matrix."""

    def __init__(self, signal_transform=None, feature_aggregator=None, flatten_output=True):
        self.signal_transform = signal_transform
        self.feature_aggregator = feature_aggregator
        self.flatten_output = bool(flatten_output)
        self.signal_transform_ = None

    def _resolve_transform(self):
        if self.signal_transform is None:
            return None
        if isinstance(self.signal_transform, SignalPipeline):
            return self.signal_transform
        if isinstance(self.signal_transform, (list, tuple)):
            if not self.signal_transform:
                raise ValueError("signal_transform steps must be non-empty")
            first = self.signal_transform[0]
            if isinstance(first, tuple) and len(first) == 2:
                return SignalPipeline(self.signal_transform)
            return SignalPipeline([(f"signal_{idx}", step) for idx, step in enumerate(self.signal_transform)])
        if hasattr(self.signal_transform, "fit") and hasattr(self.signal_transform, "transform"):
            return self.signal_transform
        if callable(self.signal_transform):
            return _CallableSignalTransform(self.signal_transform)
        raise TypeError(
            "signal_transform must be None, a callable, a transformer, a signal pipeline, or a list of steps"
        )

    def fit(self, X, y=None):
        transform = self._resolve_transform()
        if transform is None:
            self.signal_transform_ = None
            return self

        self.signal_transform_ = clone(transform)
        first_signal = _first_signal(X)
        if hasattr(self.signal_transform_, "fit"):
            self.signal_transform_.fit(first_signal, y)
        return self

    def transform(self, X):
        signals = _as_signal_list(X)
        if not signals:
            return np.empty((0, 0), dtype=float)

        rows = []
        transform = self.signal_transform_ if self.signal_transform_ is not None else None
        for signal in signals:
            features = signal
            if transform is not None:
                features = transform.transform(signal)
            pooled = _pool_features(
                features,
                feature_aggregator=self.feature_aggregator,
                flatten_output=self.flatten_output,
            )
            rows.append(np.asarray(pooled, dtype=float).ravel())

        widths = {row.size for row in rows}
        if len(widths) != 1:
            raise ValueError("All transformed signals must produce the same feature width")
        return np.vstack(rows)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_params(self, deep=True):
        params = {
            "signal_transform": self.signal_transform,
            "feature_aggregator": self.feature_aggregator,
            "flatten_output": self.flatten_output,
        }
        if deep and hasattr(self.signal_transform, "get_params"):
            for key, value in self.signal_transform.get_params().items():
                params[f"signal_transform__{key}"] = value
        return params

    def set_params(self, **params):
        for key, value in params.items():
            if key == "signal_transform":
                self.signal_transform = value
            elif key == "feature_aggregator":
                self.feature_aggregator = value
            elif key == "flatten_output":
                self.flatten_output = bool(value)
            elif key.startswith("signal_transform__") and hasattr(self.signal_transform, "set_params"):
                self.signal_transform.set_params(**{key.split("__", 1)[1]: value})
            else:
                setattr(self, key, value)
        return self


class SignalExperiment:
    """High-level workflow for raw signals, feature extraction, search, and scoring."""

    def __init__(
        self,
        model,
        signal_transform=None,
        feature_aggregator=None,
        flatten_output=True,
        search="grid",
        param_grid=None,
        param_distributions=None,
        n_iter=10,
        cv=None,
        scoring=None,
        refit=True,
        return_train_score=False,
        random_state=None,
        task="classification",
    ):
        self.model = model
        self.signal_transform = signal_transform
        self.feature_aggregator = feature_aggregator
        self.flatten_output = bool(flatten_output)
        self.search = search
        self.param_grid = param_grid
        self.param_distributions = param_distributions
        self.n_iter = int(n_iter)
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.return_train_score = return_train_score
        self.random_state = random_state
        self.task = task
        self.pipeline_ = None
        self.searcher_ = None
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.feature_transformer_ = None

    def _build_feature_transformer(self):
        transformer = SignalFeatureTransformer(
            signal_transform=self.signal_transform,
            feature_aggregator=self.feature_aggregator,
            flatten_output=self.flatten_output,
        )
        return transformer

    def _resolve_searcher(self, pipeline):
        if self.search is None:
            return None
        if self.search == "grid":
            if self.param_grid is None:
                return None
            return GridSearchCV(
                pipeline,
                self.param_grid,
                cv=self.cv,
                scoring=self.scoring,
                refit=self.refit,
                return_train_score=self.return_train_score,
            )
        if self.search == "random":
            if self.param_distributions is None:
                raise ValueError("param_distributions must be provided when search='random'")
            return RandomizedSearchCV(
                pipeline,
                self.param_distributions,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=self.scoring,
                refit=self.refit,
                return_train_score=self.return_train_score,
                random_state=self.random_state,
            )
        raise ValueError("search must be one of: None, 'grid', 'random'")

    def fit(self, X, y, groups=None):
        y = np.asarray(y)
        X = _prepare_signal_batch_input(X)
        feature_transformer = self._build_feature_transformer()
        pipeline = PreprocessingPipeline([("features", feature_transformer), ("model", clone(self.model))])
        self.pipeline_ = pipeline
        self.feature_transformer_ = feature_transformer

        searcher = self._resolve_searcher(pipeline)
        if searcher is None:
            self.best_estimator_ = pipeline.fit(X, y)
            self.best_params_ = pipeline.get_params()
            self.best_score_ = None
            self.cv_results_ = None
            self.searcher_ = None
            return self

        self.searcher_ = searcher
        self.searcher_.fit(X, y, groups=groups)
        self.best_estimator_ = self.searcher_.best_estimator_
        self.best_params_ = self.searcher_.best_params_
        self.best_score_ = self.searcher_.best_score_
        self.cv_results_ = self.searcher_.cv_results_
        return self

    def _require_fitted(self):
        if self.best_estimator_ is None:
            raise RuntimeError("SignalExperiment has not been fit yet")

    def transform_features(self, X):
        self._require_fitted()
        transformer = self.best_estimator_.named_steps.get("features")
        if transformer is None:
            raise RuntimeError("SignalExperiment does not contain a features step")
        return transformer.transform(_prepare_signal_batch_input(X))

    def predict(self, X):
        self._require_fitted()
        return self.best_estimator_.predict(_prepare_signal_batch_input(X))

    def predict_proba(self, X):
        self._require_fitted()
        if not hasattr(self.best_estimator_, "predict_proba"):
            raise AttributeError("Best estimator does not define predict_proba")
        return self.best_estimator_.predict_proba(_prepare_signal_batch_input(X))

    def score(self, X, y):
        self._require_fitted()
        return self.best_estimator_.score(_prepare_signal_batch_input(X), y)

    def summary(self):
        self._require_fitted()
        return {
            "task": self.task,
            "search": self.search,
            "best_params": self.best_params_,
            "best_score": self.best_score_,
            "model": self.model.__class__.__name__,
            "has_signal_transform": self.signal_transform is not None,
            "flatten_output": self.flatten_output,
        }


def compare_signal_models(
    models,
    X,
    y,
    signal_transform=None,
    feature_aggregator=None,
    flatten_output=True,
    cv=None,
    scoring=None,
    task="classification",
):
    """Fit several signal candidates and return a simple leaderboard."""

    if not models:
        raise ValueError("models must be non-empty")

    leaderboard = []
    best_name = None
    best_score = -np.inf
    best_experiment = None

    for name, model in models:
        experiment = SignalExperiment(
            model=model,
            signal_transform=signal_transform,
            feature_aggregator=feature_aggregator,
            flatten_output=flatten_output,
            search=None,
            cv=cv,
            scoring=scoring,
            task=task,
        )
        experiment.fit(X, y)
        score = experiment.score(X, y)
        leaderboard.append((name, float(score)))
        if score > best_score:
            best_score = float(score)
            best_name = name
            best_experiment = experiment

    leaderboard.sort(key=lambda item: item[1], reverse=True)
    return {
        "leaderboard": leaderboard,
        "best_name": best_name,
        "best_score": best_score,
        "best_experiment": best_experiment,
    }


__all__ = [
    "SignalExperiment",
    "SignalFeatureTransformer",
    "compare_signal_models",
]
