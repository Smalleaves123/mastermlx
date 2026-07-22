"""High-level tabular training workflows."""

from __future__ import annotations

import numpy as np
from typing import Any

from ..data.search import GridSearchCV, RandomizedSearchCV
from ..data.cv import KFold
from ..data.drift import drift_report
from ..data.model_selection import cross_val_score
from ..data.quality import quality_report
from ..math_tools.calibration import brier_score, expected_calibration_error, maximum_calibration_error
from ..preprocessing import AutoPreprocessor, Pipeline as PreprocessingPipeline
from ..utils.estimator import clone
from ..utils.metrics import accuracy, mean_absolute_error, r2_score, root_mean_squared_error


def _normalize_steps(preprocessing):
    if preprocessing is None:
        return []
    if isinstance(preprocessing, str):
        if preprocessing != "auto":
            raise ValueError("preprocessing string must be 'auto'")
        return [("preprocess", AutoPreprocessor())]
    if isinstance(preprocessing, PreprocessingPipeline):
        return list(preprocessing.steps)
    if isinstance(preprocessing, (list, tuple)):
        if not preprocessing:
            raise ValueError("preprocessing steps must be non-empty")
        first = preprocessing[0]
        if isinstance(first, tuple) and len(first) == 2:
            return list(preprocessing)
        return [(f"preprocess_{idx}", step) for idx, step in enumerate(preprocessing)]
    if hasattr(preprocessing, "fit") and hasattr(preprocessing, "transform"):
        return [("preprocess", preprocessing)]
    raise TypeError("preprocessing must be None, a transformer, a pipeline, or a list of steps")


def _build_pipeline(model, preprocessing):
    steps = _normalize_steps(preprocessing)
    steps.append(("model", model))
    return PreprocessingPipeline(steps)


class TabularExperiment:
    """Business-friendly tabular workflow for preprocessing, search, and evaluation."""

    def __init__(
        self,
        model,
        preprocessing=None,
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
        self.preprocessing = preprocessing
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
        self.best_estimator_: Any | None = None
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.cv_scores_ = None
        self.reference_X_ = None
        self.reference_y_ = None

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
        X = np.asarray(X)
        y = np.asarray(y)
        self.reference_X_ = np.array(X, copy=True)
        self.reference_y_ = np.array(y, copy=True)
        pipeline = _build_pipeline(clone(self.model), self.preprocessing)
        self.pipeline_ = pipeline
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
            raise RuntimeError("TabularExperiment has not been fit yet")
        return self.best_estimator_

    def predict(self, X):
        best = self._require_fitted()
        return best.predict(X)

    def predict_proba(self, X):
        best = self._require_fitted()
        if not hasattr(best, "predict_proba"):
            raise AttributeError("Best estimator does not define predict_proba")
        return best.predict_proba(X)

    def score(self, X, y):
        best = self._require_fitted()
        return best.score(X, y)

    def cv_score(self, X, y, groups=None):
        """Return cross-validated scores for the configured workflow."""

        self._require_fitted()
        cv = self.cv
        if cv is None:
            n_splits = min(5, np.asarray(X).shape[0])
            if n_splits < 2:
                raise ValueError("at least two samples are required for cross-validation")
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        pipe = _build_pipeline(clone(self.model), self.preprocessing)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=self.scoring, groups=groups)
        self.cv_scores_ = np.asarray(scores, dtype=float)
        return self.cv_scores_

    def _feature_importance_report(self):
        pipeline = self._require_fitted()
        named_steps = pipeline.named_steps
        model = named_steps.get("model", pipeline)
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_, dtype=float).ravel()
            kind = "split_frequency"
        elif hasattr(model, "coef_"):
            values = np.asarray(model.coef_, dtype=float)
            values = np.abs(values).mean(axis=0) if values.ndim > 1 else np.abs(values).ravel()
            kind = "absolute_coefficient"
        else:
            return None

        names = np.asarray([f"x{idx}" for idx in range(pipeline.n_features_in_ or values.size)], dtype=object)
        for name, step in list(named_steps.items())[:-1]:
            if not hasattr(step, "get_feature_names_out"):
                continue
            try:
                names = np.asarray(step.get_feature_names_out(names), dtype=object).ravel()
            except (TypeError, ValueError, RuntimeError):
                names = np.asarray(step.get_feature_names_out(), dtype=object).ravel()
        if names.size != values.size:
            names = np.asarray([f"feature_{idx}" for idx in range(values.size)], dtype=object)
        order = np.argsort(-values, kind="stable")
        return {
            "kind": kind,
            "items": [
                {"feature": str(names[index]), "importance": float(values[index])}
                for index in order
            ],
        }

    def report(self, X=None, y=None, *, drift_reference=None, n_bins=10):
        """Build a compact quality, drift, performance, and calibration report.

        With no arguments, the report describes the training reference data.
        Passing new ``X`` compares it against that reference for drift.
        """

        self._require_fitted()
        if X is None:
            X = self.reference_X_
        if y is None and X is self.reference_X_:
            y = self.reference_y_
        if X is None:
            raise RuntimeError("TabularExperiment has no reference data")
        X = np.asarray(X)
        y_array = None if y is None else np.asarray(y)
        reference = self.reference_X_ if drift_reference is None else np.asarray(drift_reference)
        result = {
            "summary": self.summary(),
            "quality": quality_report(X, y_array),
            "drift": None if reference is None else drift_report(reference, X, bins=n_bins),
            "feature_importance": self._feature_importance_report(),
        }
        if y_array is None:
            result["performance"] = None
            return result

        prediction = self.predict(X)
        performance: dict[str, Any] = {"score": float(self.score(X, y_array))}
        if self.task == "classification":
            performance["accuracy"] = float(accuracy(y_array, prediction))
            try:
                probabilities = self.predict_proba(X)
            except AttributeError:
                probabilities = None
            if probabilities is not None:
                probabilities = np.asarray(probabilities, dtype=float)
                performance["confidence"] = {
                    "mean": float(np.mean(np.max(probabilities, axis=1))),
                    "min": float(np.min(np.max(probabilities, axis=1))),
                    "max": float(np.max(np.max(probabilities, axis=1))),
                }
                calibration = {"brier_score": float(brier_score(y_array, probabilities))}
                if np.unique(y_array).size == 2:
                    calibration.update(
                        ece=float(expected_calibration_error(y_array, probabilities, n_bins=n_bins)),
                        mce=float(maximum_calibration_error(y_array, probabilities, n_bins=n_bins)),
                    )
                performance["calibration"] = calibration
        else:
            performance.update(
                r2=float(r2_score(y_array, prediction)),
                mae=float(mean_absolute_error(y_array, prediction)),
                rmse=float(root_mean_squared_error(y_array, prediction)),
            )
        result["performance"] = performance
        return result

    def summary(self):
        self._require_fitted()
        return {
            "task": self.task,
            "search": self.search,
            "best_params": self.best_params_,
            "best_score": self.best_score_,
            "model": self.model.__class__.__name__,
            "has_preprocessing": self.preprocessing is not None,
        }


def compare_tabular_models(models, X, y, preprocessing=None, cv=None, scoring=None, task="classification"):
    """Fit several tabular candidates and return a simple leaderboard."""

    if not models:
        raise ValueError("models must be non-empty")

    leaderboard = []
    best_name = None
    best_score = -np.inf
    best_experiment = None

    for name, model in models:
        experiment = TabularExperiment(
            model=model,
            preprocessing=preprocessing,
            search=None,
            cv=cv,
            scoring=scoring,
            task=task,
        )
        experiment.fit(X, y)
        scores = experiment.cv_score(X, y)
        score = float(np.mean(scores))
        experiment.best_score_ = score
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


__all__ = ["TabularExperiment", "compare_tabular_models"]
