from __future__ import annotations

from itertools import product

import numpy as np

from .model_selection import cross_validate
from ..utils.estimator import clone, set_params as _set_params


def _param_grid_iter(param_grid):
    if not isinstance(param_grid, dict) or not param_grid:
        raise ValueError("param_grid must be a non-empty dict")
    keys = list(param_grid)
    values = []
    for key in keys:
        cur = list(param_grid[key])
        if not cur:
            raise ValueError(f"Parameter grid for '{key}' must be non-empty")
        values.append(cur)
    for combo in product(*values):
        yield dict(zip(keys, combo))


def _set_params(estimator, params):
    if hasattr(estimator, "set_params"):
        return estimator.set_params(**params)
    return _set_params(estimator, **params)


def _sample_param_distributions(param_distributions, n_iter, random_state=None):
    if not isinstance(param_distributions, dict) or not param_distributions:
        raise ValueError("param_distributions must be a non-empty dict")
    rng = np.random.default_rng(random_state)
    keys = list(param_distributions)
    out = []
    for _ in range(int(n_iter)):
        params = {}
        for key in keys:
            values = list(param_distributions[key])
            if not values:
                raise ValueError(f"Parameter distribution for '{key}' must be non-empty")
            pick = int(rng.integers(0, len(values)))
            params[key] = values[pick]
        out.append(params)
    return out


sample_params = _sample_param_distributions


def _summarize_search(estimator, candidates, X, y, cv=None, scoring=None, refit=True, return_train_score=False, groups=None):
    params_list = []
    summary = {}
    mean_fit = []
    mean_score = []
    refit_name = None

    for params in candidates:
        model = _set_params(clone(estimator), params)
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=return_train_score,
            groups=groups,
        )
        params_list.append(params)
        mean_fit.append(float(np.mean(scores["fit_time"])))
        mean_score.append(float(np.mean(scores["score_time"])))
        for key, values in scores.items():
            if key in {"fit_time", "score_time"}:
                continue
            summary.setdefault(key, {"mean": [], "std": []})
            summary[key]["mean"].append(float(np.mean(values)))
            summary[key]["std"].append(float(np.std(values)))

    test_keys = sorted([key for key in summary if key.startswith("test_")])
    if not test_keys:
        raise RuntimeError("No test scores were produced during search")
    if len(test_keys) == 1:
        refit_name = test_keys[0]
    elif isinstance(refit, str):
        refit_name = f"test_{refit}" if not refit.startswith("test_") else refit
    elif refit:
        refit_name = test_keys[0]

    best_idx = int(np.argmax(summary[refit_name]["mean"])) if refit_name is not None else 0
    best_params = params_list[best_idx]
    best_score = summary[refit_name]["mean"][best_idx] if refit_name is not None else None

    results = {
        "params": params_list,
        "mean_fit_time": np.asarray(mean_fit, dtype=float),
        "mean_score_time": np.asarray(mean_score, dtype=float),
    }
    for key, values in summary.items():
        results[f"mean_{key}"] = np.asarray(values["mean"], dtype=float)
        results[f"std_{key}"] = np.asarray(values["std"], dtype=float)

    best_estimator = None
    if refit:
        best_estimator = _set_params(clone(estimator), best_params)
        best_estimator.fit(X, y)
    return best_estimator, best_params, best_score, results


class GridSearchCV:
    """Grid search over a parameter grid with cross-validation."""

    def __init__(self, estimator, param_grid, cv=None, scoring=None, refit=True, return_train_score=False):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.return_train_score = return_train_score
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None

    def fit(self, X, y, groups=None):
        X = np.asarray(X)
        y = np.asarray(y)
        candidates = list(_param_grid_iter(self.param_grid))
        self.best_estimator_, self.best_params_, self.best_score_, self.cv_results_ = _summarize_search(
            self.estimator,
            candidates,
            X,
            y,
            cv=self.cv,
            scoring=self.scoring,
            refit=self.refit,
            return_train_score=self.return_train_score,
            groups=groups,
        )
        return self

    def predict(self, X):
        if self.best_estimator_ is None:
            raise RuntimeError("GridSearchCV has not been fit yet")
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        if self.best_estimator_ is None:
            raise RuntimeError("GridSearchCV has not been fit yet")
        return self.best_estimator_.score(X, y)


class RandomizedSearchCV:
    """Randomized parameter search with cross-validation."""

    def __init__(
        self,
        estimator,
        param_distributions,
        n_iter=10,
        cv=None,
        scoring=None,
        refit=True,
        return_train_score=False,
        random_state=None,
    ):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = int(n_iter)
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.return_train_score = return_train_score
        self.random_state = random_state
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None

    def fit(self, X, y, groups=None):
        X = np.asarray(X)
        y = np.asarray(y)
        if self.n_iter < 1:
            raise ValueError("n_iter must be at least 1")
        candidates = _sample_param_distributions(
            self.param_distributions,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.best_estimator_, self.best_params_, self.best_score_, self.cv_results_ = _summarize_search(
            self.estimator,
            candidates,
            X,
            y,
            cv=self.cv,
            scoring=self.scoring,
            refit=self.refit,
            return_train_score=self.return_train_score,
            groups=groups,
        )
        return self

    def predict(self, X):
        if self.best_estimator_ is None:
            raise RuntimeError("RandomizedSearchCV has not been fit yet")
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        if self.best_estimator_ is None:
            raise RuntimeError("RandomizedSearchCV has not been fit yet")
        return self.best_estimator_.score(X, y)
