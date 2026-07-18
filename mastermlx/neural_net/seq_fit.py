from __future__ import annotations

import numpy as np
from typing import Any, Protocol

from ..data.split import train_test_split
from ..utils.array import one_hot
from ..utils.validation import check_1d_array, check_same_rows
from ._training import run_supervised_training_loop
from .layers import Dense
from .losses import BinaryCrossEntropyLoss, CrossEntropyLoss, MSELoss


class _SequentialContext(Protocol):
    callbacks: list[Any]
    classes_: np.ndarray | None
    best_epoch_: Any
    best_val_loss_: Any
    history_: Any
    layers: list[Any]
    lr_scheduler: Any
    loss_: Any
    optimizer_: Any
    task: str
    training_config_: Any
    val_loss_: Any

    def _backward_weighted(self, grad: Any, weight: float = ...) -> Any: ...
    def _build_optimizer(self) -> Any: ...
    def _check_input(self, X: Any) -> np.ndarray: ...
    def _evaluate_loss(self, X: Any, y: Any, loss_fn: Any, classes: Any = ...) -> Any: ...
    def _evaluate_metrics(self, X: Any, y: Any) -> Any: ...
    def _flush_accumulation(self) -> Any: ...
    def _forward(self, X: Any) -> Any: ...
    def _on_epoch_end(self, epoch: int, logs: Any) -> Any: ...
    def _reset_grad_accumulation(self) -> Any: ...
    def _resolve_lr_scheduler(self) -> Any: ...
    def _restore_state(self, snapshot: Any) -> Any: ...
    def _snapshot_state(self) -> Any: ...
    def train(self, mode: bool = ...) -> Any: ...


class _SequentialFit:
    classes_: np.ndarray | None

    def fit(self: _SequentialContext, X, y=None, resume=False):
        X = self._check_input(X)
        resume = bool(resume)
        if resume and self.optimizer_ is None:
            raise RuntimeError("resume=True requires a loaded training checkpoint")
        if not resume:
            self._build_optimizer()
            self._resolve_lr_scheduler()
        elif self.lr_scheduler is not None and hasattr(self.lr_scheduler, "optimizer"):
            self.lr_scheduler.optimizer = self.optimizer_
        self._reset_grad_accumulation()
        for callback in self.callbacks:
            if hasattr(callback, "set_model"):
                callback.set_model(self)
        self.train(True)
        cfg = self.training_config_

        if self.task == "classification":
            y = check_1d_array(y)
            X, y = check_same_rows(X, y)
            classes = np.unique(y)
            if classes.size < 2:
                raise ValueError("Sequential classification requires at least two classes")
            self.classes_ = classes
            y_idx = np.searchsorted(classes, y)
            loss_fn: Any = CrossEntropyLoss(from_logits=True)
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

            def train_step(xb, yb):
                logits = self._forward(xb)
                target = one_hot(yb, classes.size)
                loss = loss_fn(target, logits)
                if cfg.l2:
                    loss += 0.5 * float(cfg.l2) * sum(
                        np.sum(layer.W_**2)
                        for layer in self.layers
                        if isinstance(layer, Dense) and layer.W_ is not None
                    )
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
                evaluate_train_metrics=lambda Xt, yt: self._evaluate_metrics(Xt, yt),
                evaluate_val_metrics=lambda Xt, yt: self._evaluate_metrics(Xt, yt),
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

        if self.task == "multilabel":
            y = np.asarray(y, dtype=float)
            if y.ndim == 1:
                y = y[:, None]
            if y.ndim != 2 or y.shape[0] != X.shape[0]:
                raise ValueError("multilabel targets must have shape (n_samples, n_labels)")
            if np.any((y < 0.0) | (y > 1.0)):
                raise ValueError("multilabel targets must be in [0, 1]")
            X, y = check_same_rows(X, y)
            loss_fn = BinaryCrossEntropyLoss(from_logits=True)
            if cfg.validation_split and cfg.validation_split > 0.0:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=cfg.validation_split, shuffle=cfg.shuffle, random_state=cfg.random_state
                )
            else:
                X_train, y_train, X_val, y_val = X, y, None, None
            batch_size = X_train.shape[0] if cfg.batch_size is None else max(1, min(int(cfg.batch_size), X_train.shape[0]))

            def train_step(xb, yb):
                logits = self._forward(xb)
                self._backward_weighted(loss_fn.grad(yb, logits), weight=xb.shape[0])

            result = run_supervised_training_loop(
                X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                batch_size=batch_size, n_iter=cfg.n_iter, shuffle=cfg.shuffle,
                random_state=cfg.random_state, tol=cfg.tol, patience=cfg.patience,
                verbose=cfg.verbose, train_step=train_step, flush_step=self._flush_accumulation,
                evaluate_train_loss=lambda Xt, yt: self._evaluate_loss(Xt, yt, loss_fn),
                evaluate_val_loss=lambda Xt, yt: self._evaluate_loss(Xt, yt, loss_fn),
                evaluate_train_metrics=lambda Xt, yt: self._evaluate_metrics(Xt, yt),
                evaluate_val_metrics=lambda Xt, yt: self._evaluate_metrics(Xt, yt),
                snapshot_state=self._snapshot_state, restore_state=self._restore_state,
                on_epoch_end=self._on_epoch_end, callbacks=self.callbacks,
            )
            self.loss_ = result["loss"]
            self.val_loss_ = result["val_loss"]
            self.history_ = result["history"]
            self.best_epoch_ = result["best_epoch"]
            self.best_val_loss_ = result["best_val_loss"]
            return self

        if self.task in {"regression", "multioutput_regression"}:
            y = np.asarray(y, dtype=float)
            if y.ndim not in {1, 2}:
                raise ValueError("regression targets must be 1D or 2D")
            X, y = check_same_rows(X, y)
            loss_fn = MSELoss()
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

            def train_step(xb, yb):
                if yb.ndim == 1:
                    yb = yb[:, None]
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
                evaluate_train_metrics=lambda Xt, yt: self._evaluate_metrics(Xt, yt),
                evaluate_val_metrics=lambda Xt, yt: self._evaluate_metrics(Xt, yt),
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

        raise ValueError("task must be 'classification' or 'regression'")



__all__ = ["_SequentialFit"]
