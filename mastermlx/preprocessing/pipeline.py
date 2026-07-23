from __future__ import annotations

import numpy as np
from typing import Any
from numpy.typing import ArrayLike

from ..base import BaseEstimator
from ..utils.estimator import clone
from ..utils.validation import check_X


class Pipeline(BaseEstimator):
    """Chain transformers and a final estimator."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.steps_: list[tuple[str, Any]] | None = None
        self.n_features_in_: int | None = None

    def _fitted_steps(self):
        if self.steps_ is None:
            raise RuntimeError("Pipeline has not been fit yet")
        return self.steps_

    def _validate_steps(self):
        if not self.steps:
            raise ValueError("steps must be non-empty")
        names = [name for name, _ in self.steps]
        if len(names) != len(set(names)):
            raise ValueError("step names must be unique")
        return self.steps

    def __len__(self):
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def __repr__(self):
        parts = ", ".join(f"({name!r}, {step.__class__.__name__})" for name, step in self.steps)
        return f"Pipeline([{parts}])"

    def get_params(self, deep=True):
        params = {"steps": self.steps}
        for name, step in self.steps:
            params[name] = step
            if deep and hasattr(step, "get_params"):
                for key, value in step.get_params().items():
                    params[f"{name}__{key}"] = value
            elif deep and hasattr(step, "__dict__"):
                for key, value in step.__dict__.items():
                    if key.endswith("_") or key.startswith("_") or callable(value):
                        continue
                    params[f"{name}__{key}"] = value
        return params

    def set_params(self, **params):
        steps = list(self.steps)
        name_to_idx = {name: idx for idx, (name, _) in enumerate(steps)}

        for key, value in params.items():
            if "__" not in key:
                if key == "steps":
                    steps = list(value)
                    name_to_idx = {name: idx for idx, (name, _) in enumerate(steps)}
                elif key in name_to_idx:
                    steps[name_to_idx[key]] = (key, value)
                else:
                    raise ValueError(f"Invalid parameter '{key}' for Pipeline")
                continue

            name, subkey = key.split("__", 1)
            if name not in name_to_idx:
                raise ValueError(f"Unknown step '{name}'")
            step_name, step = steps[name_to_idx[name]]
            if hasattr(step, "set_params"):
                step.set_params(**{subkey: value})
            elif hasattr(step, subkey):
                setattr(step, subkey, value)
            else:
                raise ValueError(f"Invalid parameter '{subkey}' for step '{name}'")
            steps[name_to_idx[name]] = (step_name, step)

        self.steps = steps
        return self

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "Pipeline":
        steps = self._validate_steps()
        Xt = check_X(X)
        self.n_features_in_ = Xt.shape[1]
        self.steps_ = None
        fitted = []

        for name, step in steps[:-1]:
            obj = clone(step)
            if not hasattr(obj, "fit") or not hasattr(obj, "transform"):
                raise TypeError(f"Step '{name}' must define fit and transform")
            Xt = obj.fit_transform(Xt, y)
            fitted.append((name, obj))

        name, step = steps[-1]
        obj = clone(step)
        if not hasattr(obj, "fit"):
            raise TypeError(f"Step '{name}' must define fit")
        obj.fit(Xt, y)
        fitted.append((name, obj))
        self.steps_ = fitted
        return self

    def fit_transform(self, X: ArrayLike, y: ArrayLike | None = None) -> np.ndarray:
        self.fit(X, y)
        last = self._fitted_steps()[-1][1]
        if hasattr(last, "transform"):
            return self.transform(X)
        return self._transform_input(X)

    def _transform_input(self, X):
        steps = self._fitted_steps()
        Xt = self._check_X(X)
        for _, step in steps[:-1]:
            Xt = step.transform(Xt)
        return Xt

    @property
    def named_steps(self):
        if self.steps_ is None:
            return {name: step for name, step in self.steps}
        return {name: step for name, step in self._fitted_steps()}

    def transform(self, X: ArrayLike) -> np.ndarray:
        Xt = self._transform_input(X)
        last = self._fitted_steps()[-1][1]
        if hasattr(last, "transform"):
            return last.transform(Xt)
        return Xt

    def get_feature_names_out(self, input_features=None):
        """Return output names from the final transformer when available."""

        last = self._fitted_steps()[-1][1]
        if not hasattr(last, "get_feature_names_out"):
            if input_features is None:
                if self.n_features_in_ is None:
                    raise RuntimeError("Pipeline has not recorded its input feature count")
                input_features = [f"x{idx}" for idx in range(self.n_features_in_)]
            return np.asarray(input_features, dtype=object)
        return last.get_feature_names_out(input_features)

    def predict(self, X: ArrayLike) -> Any:
        Xt = self._transform_input(X)
        last = self._fitted_steps()[-1][1]
        if not hasattr(last, "predict"):
            raise AttributeError("Final step does not define predict")
        return last.predict(Xt)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        Xt = self._transform_input(X)
        last = self._fitted_steps()[-1][1]
        if not hasattr(last, "predict_proba"):
            raise AttributeError("Final step does not define predict_proba")
        return last.predict_proba(Xt)

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        Xt = self._transform_input(X)
        last = self._fitted_steps()[-1][1]
        if not hasattr(last, "decision_function"):
            raise AttributeError("Final step does not define decision_function")
        return last.decision_function(Xt)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        Xt = self._transform_input(X)
        last = self._fitted_steps()[-1][1]
        if hasattr(last, "score"):
            return last.score(Xt, y)
        pred = last.predict(Xt)
        return np.mean(pred == np.asarray(y))
