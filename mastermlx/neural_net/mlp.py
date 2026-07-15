"""Public MLP estimators backed by the shared MLP runtime."""

from __future__ import annotations

import numpy as np

from ..base import BaseEstimator
from ..data.split import train_test_split
from ..utils.array import one_hot
from ..utils.metrics import accuracy, r2_score
from ..utils.validation import as_2d, check_1d_array, check_2d_array, check_same_rows
from .losses import CrossEntropyLoss, MSELoss
from ._training import run_supervised_training_loop
from .mlp_core import _BaseMLP, _softmax


class MLPClassifier(_BaseMLP, BaseEstimator):
    """Small NumPy MLP for classification."""

    def fit(self, X, y=None, resume=False):
        X = check_2d_array(X)
        y = check_1d_array(y)
        X, y = check_same_rows(X, y)
        cfg = self.training_config_

        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("MLPClassifier requires at least two classes")
        if resume and getattr(self, "classes_", None) is not None and not np.array_equal(self.classes_, classes):
            raise ValueError("resume data classes do not match the checkpoint")
        self.classes_: np.ndarray = classes
        y_idx = np.searchsorted(classes, y)

        if not resume or not self.layers_:
            self._build_layers(X.shape[1], classes.size)
            self._build_optimizer()
            self._resolve_lr_scheduler()
        elif self.optimizer_ is None:
            raise RuntimeError("resume=True requires a loaded training checkpoint")
        self._reset_grad_accumulation()
        for callback in self.callbacks:
            if hasattr(callback, "set_model"):
                callback.set_model(self)
        if cfg.validation_split and cfg.validation_split > 0.0:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y_idx,
                test_size=cfg.validation_split,
                shuffle=cfg.shuffle,
                random_state=cfg.random_state,
            )
        else:
            X_train, y_train, X_val, y_val = X, y_idx, None, None
        batch_size = X_train.shape[0] if cfg.batch_size is None else max(1, min(int(cfg.batch_size), X_train.shape[0]))
        loss_fn = CrossEntropyLoss(from_logits=True)

        def train_step(xb, yb):
            logits = self._forward(xb)
            target = one_hot(yb, classes.size)
            loss = loss_fn(target, logits)
            if cfg.l2:
                loss += self._regularization_penalty()
            grad = loss_fn.grad(target, logits)
            self._backward_weighted(grad, weight=xb.shape[0])

        result = run_supervised_training_loop(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            batch_size=batch_size,
            n_iter=cfg.n_iter,
            shuffle=cfg.shuffle,
            random_state=cfg.random_state,
            tol=cfg.tol,
            patience=cfg.patience,
            verbose=cfg.verbose,
            train_step=train_step,
            flush_step=self._flush_accumulation,
            evaluate_train_loss=lambda Xt, yt: self._evaluate_loss(Xt, yt, loss_fn, classes=classes),
            evaluate_val_loss=lambda Xt, yt: self._evaluate_loss(Xt, yt, loss_fn, classes=classes),
            evaluate_train_metrics=lambda Xt, yt: self._evaluate_metrics(Xt, yt, "classification", classes),
            evaluate_val_metrics=lambda Xt, yt: self._evaluate_metrics(Xt, yt, "classification", classes),
            snapshot_state=self._snapshot_state,
            restore_state=self._restore_state,
            on_epoch_end=self._on_epoch_end,
            callbacks=self.callbacks,
        )
        self.loss_ = result["loss"]
        self.val_loss_ = result["val_loss"]
        self.history_ = result["history"]
        self.best_epoch_ = result["best_epoch"]
        self.best_val_loss_ = result["best_val_loss"]

        return self

    def predict_proba(self, X):
        X = as_2d(X)
        was_training = self._current_mode()
        self._set_mode(False)
        try:
            logits = self._forward(X)
            return _softmax(logits)
        finally:
            self._set_mode(was_training)

    def predict(self, X):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]

    def evaluate(self, X, y, metrics=None):
        return self._eval(
            X,
            y,
            "classification",
            CrossEntropyLoss(from_logits=True),
            classes=self.classes_,
            metrics=metrics,
        )

    def score(self, X, y):
        return accuracy(y, self.predict(X))


class MLPRegressor(_BaseMLP, BaseEstimator):
    """Small NumPy MLP for regression."""

    def fit(self, X, y=None, resume=False):
        X = check_2d_array(X)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2 or y.shape[1] < 1:
            raise ValueError("regression targets must be 1D or 2D with at least one output")
        X, y = check_same_rows(X, y)
        cfg = self.training_config_
        task = "multioutput_regression" if y.shape[1] > 1 else "regression"

        if not resume or not self.layers_:
            self._build_layers(X.shape[1], y.shape[1])
            self._build_optimizer()
            self._resolve_lr_scheduler()
        elif self.optimizer_ is None:
            raise RuntimeError("resume=True requires a loaded training checkpoint")
        elif self.layers_[-1].n_outputs != y.shape[1]:
            raise ValueError("resume target width does not match the checkpoint")
        self.n_outputs_ = y.shape[1]
        self._reset_grad_accumulation()
        for callback in self.callbacks:
            if hasattr(callback, "set_model"):
                callback.set_model(self)
        if cfg.validation_split and cfg.validation_split > 0.0:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=cfg.validation_split,
                shuffle=cfg.shuffle,
                random_state=cfg.random_state,
            )
        else:
            X_train, y_train, X_val, y_val = X, y, None, None
        batch_size = X_train.shape[0] if cfg.batch_size is None else max(1, min(int(cfg.batch_size), X_train.shape[0]))
        loss_fn = MSELoss()

        def train_step(xb, yb):
            pred = self._forward(xb)
            grad = loss_fn.grad(yb, pred)
            self._backward_weighted(grad, weight=xb.shape[0])

        result = run_supervised_training_loop(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            batch_size=batch_size,
            n_iter=cfg.n_iter,
            shuffle=cfg.shuffle,
            random_state=cfg.random_state,
            tol=cfg.tol,
            patience=cfg.patience,
            verbose=cfg.verbose,
            train_step=train_step,
            flush_step=self._flush_accumulation,
            evaluate_train_loss=lambda Xt, yt: self._evaluate_loss(Xt, yt, loss_fn),
            evaluate_val_loss=lambda Xt, yt: self._evaluate_loss(Xt, yt, loss_fn),
            evaluate_train_metrics=lambda Xt, yt: self._evaluate_metrics(Xt, yt, task),
            evaluate_val_metrics=lambda Xt, yt: self._evaluate_metrics(Xt, yt, task),
            snapshot_state=self._snapshot_state,
            restore_state=self._restore_state,
            on_epoch_end=self._on_epoch_end,
            callbacks=self.callbacks,
        )
        self.loss_ = result["loss"]
        self.val_loss_ = result["val_loss"]
        self.history_ = result["history"]
        self.best_epoch_ = result["best_epoch"]
        self.best_val_loss_ = result["best_val_loss"]

        return self

    def predict(self, X):
        X = as_2d(X)
        was_training = self._current_mode()
        self._set_mode(False)
        try:
            output = self._forward(X)
            return output[:, 0] if output.shape[1] == 1 else output
        finally:
            self._set_mode(was_training)

    def evaluate(self, X, y, metrics=None):
        outputs = getattr(self, "n_outputs_", 1)
        task = "multioutput_regression" if outputs > 1 else "regression"
        return self._eval(X, y, task, MSELoss(), metrics=metrics)

    def score(self, X, y):
        return r2_score(y, self.predict(X))


__all__ = ["MLPClassifier", "MLPRegressor"]
