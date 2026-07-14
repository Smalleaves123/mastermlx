from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils.estimator import clone
from .ts_core import _as_1d_series, _score, lagged_matrix
from .ts_metrics import ForecastMetrics
from .ts_models import TimeSeriesPipeline


def _grid(params):
    if not isinstance(params, dict) or not params:
        raise ValueError("param_grid must be a non-empty dict")
    keys = list(params)
    vals = []
    for key in keys:
        cur = list(params[key])
        if not cur:
            raise ValueError(f"Parameter grid for '{key}' must be non-empty")
        vals.append(cur)
    out = [{}]
    for key, cur in zip(keys, vals):
        nxt = []
        for base in out:
            for val in cur:
                item = dict(base)
                item[key] = val
                nxt.append(item)
        out = nxt
    return out


def _sample(params, n_iter, seed=None):
    if not isinstance(params, dict) or not params:
        raise ValueError("param_distributions must be a non-empty dict")
    rng = np.random.default_rng(seed)
    keys = list(params)
    out = []
    for _ in range(int(n_iter)):
        item = {}
        for key in keys:
            cur = list(params[key])
            if not cur:
                raise ValueError(f"Parameter distribution for '{key}' must be non-empty")
            item[key] = cur[int(rng.integers(0, len(cur)))]
        out.append(item)
    return out


class TimeSeriesExperiment(BaseEstimator):
    """Fit, tune, and score a forecasting model on ordered series splits."""

    def __init__(
        self,
        model,
        lags=12,
        horizon=1,
        preprocessing=None,
        search="grid",
        param_grid=None,
        param_distributions=None,
        n_iter=10,
        cv=None,
        scoring="neg_mean_squared_error",
        refit=True,
        random_state=None,
    ):
        self.model = model
        self.lags = int(lags)
        self.horizon = int(horizon)
        self.preprocessing = preprocessing
        self.search = search
        self.param_grid = param_grid
        self.param_distributions = param_distributions
        self.n_iter = int(n_iter)
        self.cv = cv
        self.scoring = scoring
        self.refit = bool(refit)
        self.random_state = random_state
        self.pipeline_ = None
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.history_ = None

    def _split(self, x):
        if self.cv is None:
            from ..data.cv import TimeSeriesSplit

            # Keep the default expanding-window split valid for short series.
            # Each training fold must contain enough observations to build at
            # least one lagged sample.
            min_train = self.lags + self.horizon
            n_splits = 5
            while n_splits >= 2 and x.size // (n_splits + 1) < min_train:
                n_splits -= 1
            if n_splits < 2:
                raise ValueError(
                    "Not enough samples for time-series cross-validation with "
                    f"lags={self.lags} and horizon={self.horizon}"
                )
            cv = TimeSeriesSplit(n_splits=n_splits)
        elif isinstance(self.cv, int):
            from ..data.cv import TimeSeriesSplit

            cv = TimeSeriesSplit(n_splits=int(self.cv))
        else:
            cv = self.cv
        try:
            splits = list(cv.split(x))
        except TypeError:
            splits = list(cv.split(x, x))
        min_train = self.lags + self.horizon
        if any(len(train) < min_train for train, _ in splits):
            raise ValueError(
                "Each time-series training fold must contain at least "
                f"lags + horizon = {min_train} samples"
            )
        return splits

    def _cands(self):
        if self.search is None:
            return [{}]
        if self.search == "grid":
            return _grid(self.param_grid)
        if self.search == "random":
            return _sample(self.param_distributions, self.n_iter, seed=self.random_state)
        raise ValueError("search must be one of: None, 'grid', 'random'")

    def _make_pipe(self, params=None):
        pipe = TimeSeriesPipeline(
            model=clone(self.model),
            lags=self.lags,
            horizon=self.horizon,
            preprocessing=self.preprocessing,
        )
        if params:
            pipe.set_params(**params)
        return pipe

    def fit(self, X, y=None):
        x = _as_1d_series(X)
        cand = self._cands()
        splits = self._split(x)
        rows = []
        best = None
        best_s = -np.inf

        for prm in cand:
            scores = []
            for tr, te in splits:
                p = self._make_pipe(prm)
                hist = x[tr]
                test = x[te]
                p.fit(hist)
                pred = p.forecast(steps=test.size, history=hist)
                scores.append(_score(test, pred, self.scoring))
            s = float(np.mean(scores))
            row = {"params": prm, "mean_test_score": s, "test_scores": np.asarray(scores, dtype=float)}
            rows.append(row)
            if s > best_s:
                best_s = s
                best = prm

        self.cv_results_ = rows
        self.best_params_ = best
        self.best_score_ = best_s
        if self.refit:
            self.best_estimator_ = self._make_pipe(best)
            self.best_estimator_.fit(x)
            self.pipeline_ = self.best_estimator_
        self.history_ = x
        return self

    def _need(self):
        if self.best_estimator_ is None:
            raise RuntimeError("TimeSeriesExperiment has not been fit yet")

    def forecast(self, steps=1, history=None):
        self._need()
        return self.best_estimator_.forecast(steps=steps, history=history)

    def predict(self, X=None):
        self._need()
        if X is None:
            if self.history_ is None:
                raise RuntimeError("No training history available")
            X = self.history_
        return self.best_estimator_.predict(X)

    def score(self, X, y=None):
        self._need()
        if y is None:
            if self.history_ is None:
                raise RuntimeError("No training history available")
            hist = self.history_
            feat, tgt = lagged_matrix(hist, lags=self.lags, horizon=self.horizon)
            Xt = self.best_estimator_._transform_preprocessing(feat)
            if hasattr(self.best_estimator_.model_, "score"):
                return float(self.best_estimator_.model_.score(Xt, tgt))
            pred = self.best_estimator_.model_.predict(Xt)
            return _score(tgt, pred, self.scoring)
        hist = _as_1d_series(X)
        tgt = np.asarray(y, dtype=float).ravel()
        pred = self.forecast(steps=tgt.size, history=hist)
        return _score(tgt, pred, self.scoring)

    def summary(self):
        self._need()
        return {
            "lags": self.lags,
            "horizon": self.horizon,
            "search": self.search,
            "best_params": self.best_params_,
            "best_score": self.best_score_,
            "model": self.model.__class__.__name__,
            "has_preprocessing": self.preprocessing is not None,
        }


def compare_time_series_models(models, X, lags=12, horizon=1, preprocessing=None, cv=None, scoring="r2"):
    """Fit several forecasting models and return a small leaderboard."""

    if not models:
        raise ValueError("models must be non-empty")

    x = _as_1d_series(X)
    board = []
    best_name = None
    best_score = -np.inf
    best_model = None

    for name, model in models:
        exp = TimeSeriesExperiment(
            model=model,
            lags=lags,
            horizon=horizon,
            preprocessing=preprocessing,
            search=None,
            cv=cv,
            scoring=scoring,
        )
        splits = exp._split(x)
        scores = []
        for tr, te in splits:
            p = exp._make_pipe()
            hist = x[tr]
            test = x[te]
            p.fit(hist)
            pred = p.forecast(steps=test.size, history=hist)
            scores.append(_score(test, pred, scoring))
        score = float(np.mean(scores))
        board.append((name, score))
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    board.sort(key=lambda item: item[1], reverse=True)
    best_exp = TimeSeriesExperiment(
        model=best_model,
        lags=lags,
        horizon=horizon,
        preprocessing=preprocessing,
        search=None,
        cv=cv,
        scoring=scoring,
    )
    best_exp.fit(x)
    return {
        "leaderboard": board,
        "best_name": best_name,
        "best_score": best_score,
        "best_experiment": best_exp,
    }


def backtest(model, X, lags=12, horizon=1, preprocessing=None, cv=None, scoring="neg_mean_squared_error"):
    """Evaluate a forecasting model on expanding time-series folds."""

    x = _as_1d_series(X)
    exp = TimeSeriesExperiment(
        model=model,
        lags=lags,
        horizon=horizon,
        preprocessing=preprocessing,
        search=None,
        cv=cv,
        scoring=scoring,
    )
    rows = []
    scores = []
    true = []
    pred = []

    for i, (tr, te) in enumerate(exp._split(x)):
        pipe = exp._make_pipe()
        hist = x[tr]
        y = x[te]
        yh = pipe.fit(hist).forecast(steps=y.size, history=hist)
        score = _score(y, yh, scoring)
        rows.append({
            "fold": i,
            "score": float(score),
            "n_train": int(tr.size),
            "n_test": int(te.size),
            "true": y.copy(),
            "pred": yh.copy(),
        })
        scores.append(score)
        true.append(y)
        pred.append(yh)

    scores = np.asarray(scores, dtype=float)
    return {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "true": np.concatenate(true),
        "pred": np.concatenate(pred),
        "folds": rows,
        "metrics": ForecastMetrics.report(np.concatenate(true), np.concatenate(pred)),
    }


def rolling_backtest(
    model,
    X,
    lags=12,
    horizon=1,
    preprocessing=None,
    test_size=1,
    step=1,
    window=None,
    scoring="neg_mean_squared_error",
):
    """Evaluate a forecasting model on rolling or expanding windows."""

    x = _as_1d_series(X)
    test_size = int(test_size)
    step = int(step)
    min_train = int(lags) + int(horizon)
    if test_size < 1 or step < 1:
        raise ValueError("test_size and step must be at least 1")
    if window is not None and int(window) < min_train:
        raise ValueError("window must be at least lags + horizon")
    if x.size <= min_train:
        raise ValueError("X is too short for rolling backtesting")

    rows = []
    scores = []
    true = []
    pred = []
    start = min_train
    while start < x.size:
        train_start = 0 if window is None else max(0, start - int(window))
        train = x[train_start:start]
        test = x[start : min(x.size, start + test_size)]
        if train.size < min_train or test.size == 0:
            break
        pipe = TimeSeriesPipeline(
            model=clone(model),
            lags=lags,
            horizon=horizon,
            preprocessing=preprocessing,
        )
        forecast = pipe.fit(train).forecast(steps=test.size, history=train)
        score = _score(test, forecast, scoring)
        rows.append({
            "start": int(start),
            "n_train": int(train.size),
            "n_test": int(test.size),
            "score": float(score),
            "true": test.copy(),
            "pred": np.asarray(forecast).copy(),
        })
        scores.append(score)
        true.append(test)
        pred.append(forecast)
        start += step

    if not rows:
        raise ValueError("no rolling backtest folds were produced")
    true = np.concatenate(true)
    pred = np.concatenate(pred)
    return {
        "scores": np.asarray(scores, dtype=float),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "true": true,
        "pred": pred,
        "folds": rows,
        "metrics": ForecastMetrics.report(true, pred),
    }


__all__ = [
    "TimeSeriesExperiment",
    "backtest",
    "compare_time_series_models",
    "rolling_backtest",
]
