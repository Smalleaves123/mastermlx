from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..utils.estimator import clone
from ..utils.validation import check_X


class Pipeline(BaseEstimator):
    """Chain transformers and a final estimator."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.steps_ = None

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
                    setattr(self, key, value)
                continue

            name, subkey = key.split("__", 1)
            if name not in name_to_idx:
                raise ValueError(f"Unknown step '{name}'")
            step_name, step = steps[name_to_idx[name]]
            if hasattr(step, "set_params"):
                step.set_params(**{subkey: value})
            else:
                setattr(step, subkey, value)
            steps[name_to_idx[name]] = (step_name, step)

        self.steps = steps
        return self

    def fit(self, X, y=None):
        steps = self._validate_steps()
        Xt = check_X(X)
        self._set_n_features(Xt)
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

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        last = self.steps_[-1][1]
        if hasattr(last, "transform"):
            return self.transform(X)
        return self._transform_input(X)

    def _transform_input(self, X):
        self._check_fitted("steps_")
        Xt = self._check_X(X)
        for _, step in self.steps_[:-1]:
            Xt = step.transform(Xt)
        return Xt

    @property
    def named_steps(self):
        if self.steps_ is None:
            return {name: step for name, step in self.steps}
        return {name: step for name, step in self.steps_}

    def transform(self, X):
        Xt = self._transform_input(X)
        last = self.steps_[-1][1]
        if hasattr(last, "transform"):
            return last.transform(Xt)
        return Xt

    def predict(self, X):
        Xt = self._transform_input(X)
        last = self.steps_[-1][1]
        if not hasattr(last, "predict"):
            raise AttributeError("Final step does not define predict")
        return last.predict(Xt)

    def predict_proba(self, X):
        Xt = self._transform_input(X)
        last = self.steps_[-1][1]
        if not hasattr(last, "predict_proba"):
            raise AttributeError("Final step does not define predict_proba")
        return last.predict_proba(Xt)

    def decision_function(self, X):
        Xt = self._transform_input(X)
        last = self.steps_[-1][1]
        if not hasattr(last, "decision_function"):
            raise AttributeError("Final step does not define decision_function")
        return last.decision_function(Xt)

    def score(self, X, y):
        Xt = self._transform_input(X)
        last = self.steps_[-1][1]
        if hasattr(last, "score"):
            return last.score(Xt, y)
        pred = last.predict(Xt)
        return np.mean(pred == np.asarray(y))
