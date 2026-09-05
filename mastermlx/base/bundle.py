"""Portable model bundles for validated end-to-end inference."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike

from .estimator import BaseEstimator
from .results import BaseReport, to_json_safe
from ..utils.estimator import clone
from ..version import __version__


class ModelBundle(BaseEstimator):
    """Package a data contract, preprocessor, and estimator as one model.

    Parameters are treated as unfitted templates. :meth:`fit` clones them,
    validates the training boundary, and records a portable inference unit.
    ``preprocessing="auto"`` selects :class:`AutoPreprocessor` lazily so the
    base package keeps its existing import behavior.
    """

    def __init__(self, model, *, preprocessing=None, data_contract=None, metadata=None):
        self.model = model
        self.preprocessing = preprocessing
        self.data_contract = data_contract
        self.metadata = {} if metadata is None else dict(metadata)
        self.model_: Any | None = None
        self.preprocessing_: Any | None = None
        self.data_contract_: Any | None = None
        self.feature_names_in_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.library_version_: str | None = None

    @staticmethod
    def _input_schema(X):
        arr = np.asarray(X)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D table")
        columns = getattr(X, "columns", None)
        names = [f"x{idx}" for idx in range(arr.shape[1])] if columns is None else list(columns)
        if len(names) != arr.shape[1] or len(names) != len(set(names)):
            raise ValueError("X columns must be unique and match the feature count")
        return arr, names

    def _make_preprocessor(self):
        if self.preprocessing is None:
            return None
        if isinstance(self.preprocessing, str) and self.preprocessing == "auto":
            from ..preprocessing import AutoPreprocessor

            return AutoPreprocessor()
        if isinstance(self.preprocessing, str):
            raise ValueError("preprocessing must be None, 'auto', or a transformer")
        return clone(self.preprocessing)

    def _require_fitted(self):
        if self.model_ is None or self.library_version_ is None:
            raise RuntimeError("ModelBundle has not been fit yet")

    @classmethod
    def from_fitted(
        cls,
        model,
        *,
        feature_names,
        data_contract=None,
        metadata=None,
    ) -> "ModelBundle":
        """Create a deployment bundle around an already-fitted estimator."""

        if not callable(getattr(model, "predict", None)):
            raise TypeError("model must define predict()")
        names = [str(name) for name in feature_names]
        if not names or len(names) != len(set(names)):
            raise ValueError("feature_names must be non-empty and unique")
        contract = deepcopy(data_contract)
        if contract is not None:
            if not callable(getattr(contract, "validate", None)):
                raise TypeError("data_contract must define validate()")
            contract.summary()
        bundle = cls(clone(model), metadata=metadata)
        bundle.model_ = deepcopy(model)
        bundle.data_contract_ = contract
        bundle.feature_names_in_ = np.asarray(names, dtype=object)
        bundle.n_features_in_ = len(names)
        bundle.library_version_ = __version__
        return bundle

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "ModelBundle":
        arr, names = self._input_schema(X)
        if y is not None:
            target = np.asarray(y)
            if target.ndim == 0 or target.shape[0] != arr.shape[0]:
                raise ValueError("X and y must contain the same number of samples")

        contract = deepcopy(self.data_contract)
        if contract is not None:
            if not callable(getattr(contract, "fit", None)) or not callable(
                getattr(contract, "check", None)
            ):
                raise TypeError("data_contract must define fit() and check()")
            contract.fit(X, y)
            contract.check(X, raise_on_error=True)

        preprocessor = self._make_preprocessor()
        if preprocessor is None:
            transformed = X
        else:
            if not callable(getattr(preprocessor, "fit_transform", None)):
                raise TypeError("preprocessing must define fit_transform()")
            transformed = preprocessor.fit_transform(X, y)

        model = clone(self.model)
        if not callable(getattr(model, "fit", None)):
            raise TypeError("model must define fit()")
        model.fit(transformed, y)

        self.data_contract_ = contract
        self.preprocessing_ = preprocessor
        self.model_ = model
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = int(arr.shape[1])
        self.library_version_ = __version__
        return self

    def validate(self, X) -> BaseReport:
        """Validate an inference batch without running the model."""

        self._require_fitted()
        arr, names = self._input_schema(X)
        errors = []
        if arr.shape[1] != self.n_features_in_:
            errors.append(
                {
                    "code": "feature_count",
                    "message": "input feature count differs from the fitted bundle",
                    "expected": self.n_features_in_,
                    "observed": int(arr.shape[1]),
                }
            )
        if self.feature_names_in_ is not None and names != self.feature_names_in_.tolist():
            errors.append(
                {
                    "code": "feature_names",
                    "message": "input feature names or order differ from the fitted bundle",
                    "expected": self.feature_names_in_.tolist(),
                    "observed": names,
                }
            )

        contract_report = None
        if self.data_contract_ is not None:
            contract_report = self.data_contract_.validate(X)
            errors.extend(contract_report["errors"])
        return BaseReport(
            {
                "valid": not errors,
                "errors": errors,
                "contract": contract_report,
                "n_samples": int(arr.shape[0]),
                "n_features": int(arr.shape[1]),
            }
        )

    def _transform_input(self, X):
        report = self.validate(X)
        if not report["valid"]:
            first = report["errors"][0]
            raise ValueError(f"model bundle input violation: {first['message']}")
        if self.preprocessing_ is None:
            return X
        return self.preprocessing_.transform(X)

    def predict(self, X):
        transformed = self._transform_input(X)
        model = cast(Any, self.model_)
        if not callable(getattr(model, "predict", None)):
            raise AttributeError("bundled model does not define predict()")
        return model.predict(transformed)

    def predict_proba(self, X):
        transformed = self._transform_input(X)
        model = cast(Any, self.model_)
        if not callable(getattr(model, "predict_proba", None)):
            raise AttributeError("bundled model does not define predict_proba()")
        return model.predict_proba(transformed)

    def decision_function(self, X):
        transformed = self._transform_input(X)
        model = cast(Any, self.model_)
        if not callable(getattr(model, "decision_function", None)):
            raise AttributeError("bundled model does not define decision_function()")
        return model.decision_function(transformed)

    def transform(self, X):
        return self._transform_input(X)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        transformed = self._transform_input(X)
        model = cast(Any, self.model_)
        if callable(getattr(model, "score", None)):
            return float(model.score(transformed, y))
        prediction = model.predict(transformed)
        return float(np.mean(np.asarray(prediction) == np.asarray(y)))

    def summary(self) -> BaseReport:
        self._require_fitted()
        return BaseReport(
            to_json_safe(
                {
                    "format": "mastermlx-model-bundle",
                    "library_version": self.library_version_,
                    "model": type(self.model_).__name__,
                    "preprocessing": (
                        None if self.preprocessing_ is None else type(self.preprocessing_).__name__
                    ),
                    "has_data_contract": self.data_contract_ is not None,
                    "n_features_in": self.n_features_in_,
                    "feature_names_in": self.feature_names_in_,
                    "metadata": self.metadata,
                }
            )
        )

    def export_summary(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.summary().to_json(output)
        return output

    def model_card(self) -> BaseReport:
        """Return deployment and governance metadata for the fitted bundle."""

        self._require_fitted()
        metadata = to_json_safe(self.metadata)
        governance_fields = ("owner", "intended_use", "limitations", "metrics")
        missing = [name for name in governance_fields if not metadata.get(name)]
        contract = None
        if self.data_contract_ is not None:
            contract = self.data_contract_.summary()
        model = cast(Any, self.model_)
        preprocessor = cast(Any, self.preprocessing_)
        model_params = model.get_params(deep=False) if hasattr(model, "get_params") else None
        preprocessing_params = (
            preprocessor.get_params(deep=False)
            if preprocessor is not None and hasattr(preprocessor, "get_params")
            else None
        )
        return BaseReport(
            to_json_safe(
                {
                    "format": "mastermlx-model-card",
                    "library_version": self.library_version_,
                    "model": {
                        "type": type(model).__name__,
                        "parameters": model_params,
                        "task": metadata.get("task"),
                    },
                    "input": {
                        "n_features": self.n_features_in_,
                        "feature_names": self.feature_names_in_,
                        "data_contract": contract,
                    },
                    "preprocessing": {
                        "type": None if preprocessor is None else type(preprocessor).__name__,
                        "parameters": preprocessing_params,
                    },
                    "governance": {
                        "owner": metadata.get("owner"),
                        "intended_use": metadata.get("intended_use"),
                        "out_of_scope": metadata.get("out_of_scope"),
                        "limitations": metadata.get("limitations"),
                        "metrics": metadata.get("metrics"),
                        "training_data": metadata.get("training_data"),
                        "tags": metadata.get("tags", []),
                        "missing_fields": missing,
                    },
                    "metadata": metadata,
                }
            )
        )

    def export_model_card(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.model_card().to_json(output)
        return output


__all__ = ["ModelBundle"]
