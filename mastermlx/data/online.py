"""Incremental and streaming tabular learning workflows."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from ..utils.metrics import accuracy, f1_score, mean_absolute_error, r2_score, root_mean_squared_error
from .contract import DataContract
from .drift import drift_report


def _batch_arrays(X, y=None):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a non-empty 2D array")
    if y is None:
        return X, None
    y = np.asarray(y)
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be a 1D array with one value per row in X")
    return X, y


def _metrics(task, y_true, prediction):
    if task == "classification":
        return {
            "accuracy": float(accuracy(y_true, prediction)),
            "f1_macro": float(f1_score(y_true, prediction, average="macro")),
        }
    return {
        "r2": float(r2_score(y_true, prediction)),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(root_mean_squared_error(y_true, prediction)),
    }


class OnlineTabularExperiment:
    """Run prequential online learning with drift and delayed-label support.

    If the estimator implements ``partial_fit`` and no ``window_size`` is
    configured, batches update the model incrementally.  A configured window
    intentionally refits the estimator on only the most recent samples,
    which gives models without ``partial_fit`` a bounded-memory path.
    """

    def __init__(
        self,
        estimator,
        *,
        task="classification",
        window_size=None,
        data_contract=None,
        drift_thresholds=None,
    ):
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        if not hasattr(estimator, "fit") and not hasattr(estimator, "partial_fit"):
            raise TypeError("estimator must define fit or partial_fit")
        if window_size is not None and int(window_size) < 1:
            raise ValueError("window_size must be positive")
        if data_contract is not None and not isinstance(data_contract, DataContract):
            raise TypeError("data_contract must be a DataContract")

        self.estimator = estimator
        self.task = task
        self.window_size = None if window_size is None else int(window_size)
        self.data_contract = data_contract
        self.drift_thresholds = {
            "psi": 0.2,
            "tvd": 0.2,
            "missing_rate_diff": 0.1,
        }
        if drift_thresholds is not None:
            self.drift_thresholds.update({str(key): float(value) for key, value in drift_thresholds.items()})

        self.reference_X_ = None
        self._history_X = None
        self._history_y = None
        self._initialized = False
        self._n_samples = 0
        self._n_updates = 0
        self._metric_history = []
        self._drift_history = []
        self._pending = {}
        self._pending_counter = 0

    @property
    def mode(self):
        if self.window_size is not None:
            return "sliding_window"
        if hasattr(self.estimator, "partial_fit"):
            return "partial_fit"
        return "refit_all"

    def _validate_batch(self, X, y=None):
        X, y = _batch_arrays(X, y)
        if self.data_contract is not None:
            if self.data_contract.reference_ is None:
                self.data_contract.fit(X, y)
            else:
                self.data_contract.check(X, raise_on_error=True)
        return X, y

    def _drift(self, X):
        if self.reference_X_ is None:
            return None
        report = drift_report(self.reference_X_, X)
        alerts = []
        for column in report["columns"]:
            for metric in ("psi", "tvd"):
                value = column.get(metric)
                if value is not None and np.isfinite(value) and value > self.drift_thresholds[metric]:
                    alerts.append({"column": column["name"], "metric": metric, "value": float(value), "threshold": self.drift_thresholds[metric]})
            value = abs(float(column["missing_rate_diff"]))
            if value > self.drift_thresholds["missing_rate_diff"]:
                alerts.append(
                    {
                        "column": column["name"],
                        "metric": "missing_rate_diff",
                        "value": value,
                        "threshold": self.drift_thresholds["missing_rate_diff"],
                    }
                )
        return {"detected": bool(alerts), "alerts": alerts, "report": report}

    def _remember(self, X, y):
        if self._history_X is None:
            self._history_X = np.array(X, copy=True)
            self._history_y = np.array(y, copy=True)
        else:
            self._history_X = np.concatenate([self._history_X, X], axis=0)
            self._history_y = np.concatenate([self._history_y, y], axis=0)
        if self.window_size is not None:
            self._history_X = self._history_X[-self.window_size :]
            self._history_y = self._history_y[-self.window_size :]

    def _train(self, X, y, classes=None):
        if self.mode == "partial_fit":
            kwargs = {}
            if not self._initialized and classes is not None:
                kwargs["classes"] = classes
            self.estimator.partial_fit(X, y, **kwargs)
        else:
            self._remember(X, y)
            self.estimator.fit(self._history_X, self._history_y)
        self._initialized = True

    def partial_fit(self, X, y, *, classes=None, batch_id=None, timestamp=None):
        """Score a batch before training on it, then update the estimator."""

        X, y = self._validate_batch(X, y)
        prediction = None
        metrics = None
        if self._initialized and hasattr(self.estimator, "predict"):
            prediction = np.asarray(self.estimator.predict(X))
            metrics = _metrics(self.task, y, prediction)
            self._metric_history.append({"batch_id": batch_id, "timestamp": timestamp, **metrics})

        is_reference = self.reference_X_ is None
        if is_reference:
            self.reference_X_ = np.array(X, copy=True)
        drift = None if is_reference else self._drift(X)
        if drift is not None:
            self._drift_history.append({"batch_id": batch_id, "timestamp": timestamp, **drift})

        self._train(X, y, classes=classes)
        self._n_samples += int(X.shape[0])
        self._n_updates += 1
        return {
            "batch_id": batch_id,
            "timestamp": timestamp,
            "n_samples": int(X.shape[0]),
            "metrics": metrics,
            "drift": drift,
            "trained": True,
            "prediction": None if prediction is None else prediction.tolist(),
        }

    def predict(self, X):
        if not self._initialized:
            raise RuntimeError("OnlineTabularExperiment has not been fitted")
        X, _ = self._validate_batch(X)
        return self.estimator.predict(X)

    def predict_unlabeled(self, X, sample_ids=None, *, timestamp=None):
        """Store predictions until labels arrive for delayed evaluation/training."""

        if not self._initialized:
            raise RuntimeError("OnlineTabularExperiment has not been fitted")
        X, _ = self._validate_batch(X)
        prediction = np.asarray(self.estimator.predict(X))
        if sample_ids is None:
            sample_ids = [f"pending-{self._pending_counter + index}" for index in range(X.shape[0])]
        sample_ids = list(sample_ids)
        if len(sample_ids) != X.shape[0] or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample_ids must be unique and match the batch size")
        for sample_id, row, value in zip(sample_ids, X, prediction):
            if sample_id in self._pending:
                raise ValueError(f"sample_id is already pending: {sample_id}")
            self._pending[sample_id] = {
                "X": np.array(row, copy=True),
                "prediction": value,
                "timestamp": timestamp,
            }
        self._pending_counter += X.shape[0]
        return {
            "sample_ids": sample_ids,
            "prediction": prediction.tolist(),
            "timestamp": timestamp,
        }

    def update_labels(self, sample_ids, y, *, classes=None, timestamp=None):
        """Consume delayed labels, score stored predictions, and train on them."""

        sample_ids = list(sample_ids)
        y = np.asarray(y)
        if y.ndim != 1 or len(sample_ids) != y.shape[0]:
            raise ValueError("sample_ids and y must have the same non-zero length")
        if not sample_ids or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample_ids must be non-empty and unique")
        missing = [sample_id for sample_id in sample_ids if sample_id not in self._pending]
        if missing:
            raise KeyError(f"unknown pending sample_ids: {missing}")

        records = [self._pending.pop(sample_id) for sample_id in sample_ids]
        X = np.stack([record["X"] for record in records], axis=0)
        prediction = np.asarray([record["prediction"] for record in records])
        metrics = _metrics(self.task, y, prediction)
        X, y = self._validate_batch(X, y)
        self._train(X, y, classes=classes)
        self._n_samples += int(X.shape[0])
        self._n_updates += 1
        self._metric_history.append({"batch_id": "delayed", "timestamp": timestamp, **metrics})
        return {
            "sample_ids": sample_ids,
            "n_samples": int(X.shape[0]),
            "metrics": metrics,
            "trained": True,
        }

    def report(self):
        """Return the current online-learning and monitoring state."""

        return {
            "mode": self.mode,
            "task": self.task,
            "initialized": self._initialized,
            "n_updates": self._n_updates,
            "n_samples": self._n_samples,
            "window_size": self.window_size,
            "pending_labels": len(self._pending),
            "metrics": deepcopy(self._metric_history),
            "drift": deepcopy(self._drift_history),
            "data_contract": None
            if self.data_contract is None
            else (
                self.data_contract.summary()
                if self.data_contract.reference_ is not None
                else {"fitted": False}
            ),
        }


__all__ = ["OnlineTabularExperiment"]
