"""Shared candidate generation and result aggregation for model search."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from ..utils.estimator import clone, set_params as _apply_params
from ..utils.random import resolve_rng
from .model_selection import cross_validate


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
    return _apply_params(estimator, **params)


def _sample_param_distributions(param_distributions, n_iter, random_state=None):
    if not isinstance(param_distributions, dict) or not param_distributions:
        raise ValueError("param_distributions must be a non-empty dict")
    rng = resolve_rng(random_state)
    keys = list(param_distributions)
    out = []
    for _ in range(int(n_iter)):
        params = {}
        for key in keys:
            values = list(param_distributions[key])
            if not values:
                raise ValueError(f"Parameter distribution for '{key}' must be non-empty")
            params[key] = values[int(rng.integers(0, len(values)))]
        out.append(params)
    return out


sample_params = _sample_param_distributions


def _mean(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if not np.all(np.isnan(values)) else np.nan


def _summarize_search(
    estimator,
    candidates,
    X,
    y,
    cv=None,
    scoring=None,
    refit=True,
    return_train_score=False,
    groups=None,
    error_score=np.nan,
):
    if error_score != "raise":
        try:
            error_score = float(error_score)
        except (TypeError, ValueError) as exc:
            raise ValueError("error_score must be 'raise' or a numeric value") from exc

    params_list = []
    records = []
    for params in candidates:
        params_list.append(params)
        try:
            model = _set_params(clone(estimator), params)
            scores = cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring=scoring,
                return_train_score=return_train_score,
                groups=groups,
                error_score=error_score,
            )
            errors = scores.get("errors")
            error = None if errors is None else "; ".join(item for item in errors if item)
            records.append({"scores": scores, "error": error})
        except Exception as exc:
            if error_score == "raise":
                raise
            records.append({"scores": None, "error": f"{type(exc).__name__}: {exc}"})

    successful = [record["scores"] for record in records if record["scores"] is not None]
    if not successful:
        raise RuntimeError("All search candidates failed")
    metric_keys = sorted({key for scores in successful for key in scores if key.startswith(("test_", "train_"))})
    test_keys = [key for key in metric_keys if key.startswith("test_")]
    if not test_keys:
        raise RuntimeError("No test scores were produced during search")
    n_folds = next(len(scores[key]) for scores in successful for key in scores if key.startswith("test_"))

    summary: dict[str, dict[str, list[float]]] = {
        key: {"mean": [], "std": []} for key in metric_keys
    }
    mean_fit = []
    mean_score = []
    for record in records:
        scores = record["scores"]
        if scores is None:
            mean_fit.append(np.nan)
            mean_score.append(np.nan)
        else:
            mean_fit.append(_mean(scores["fit_time"]))
            mean_score.append(_mean(scores["score_time"]))
        for key in metric_keys:
            if scores is None:
                values = np.full(n_folds, error_score, dtype=float)
            else:
                values = scores.get(key)
                if values is None:
                    continue
            summary[key]["mean"].append(_mean(values))
            values = np.asarray(values, dtype=float)
            summary[key]["std"].append(float(np.nanstd(values)) if not np.all(np.isnan(values)) else np.nan)

    if isinstance(refit, str):
        refit_name = f"test_{refit}" if not refit.startswith("test_") else refit
        if refit_name not in summary:
            raise ValueError(f"Unknown refit metric: {refit}")
    else:
        refit_name = test_keys[0]

    refit_values = np.asarray(summary[refit_name]["mean"], dtype=float)
    if np.all(np.isnan(refit_values)):
        raise RuntimeError("All search candidates failed")
    best_idx = int(np.nanargmax(refit_values))
    best_params = params_list[best_idx]
    best_score = float(refit_values[best_idx])

    results: dict[str, Any] = {
        "params": params_list,
        "mean_fit_time": np.asarray(mean_fit, dtype=float),
        "mean_score_time": np.asarray(mean_score, dtype=float),
    }
    for key, stats in summary.items():
        results[f"mean_{key}"] = np.asarray(stats["mean"], dtype=float)
        results[f"std_{key}"] = np.asarray(stats["std"], dtype=float)
    errors = [record["error"] for record in records]
    if any(error is not None for error in errors):
        results["errors"] = errors

    best_estimator = None
    if refit:
        best_estimator = _set_params(clone(estimator), best_params)
        best_estimator.fit(X, y)
    return best_estimator, best_params, best_score, best_idx, results
