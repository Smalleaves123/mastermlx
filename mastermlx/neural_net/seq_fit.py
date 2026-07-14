from __future__ import annotations

import numpy as np

from ..data.split import train_test_split
from ..utils.array import one_hot
from ..utils.validation import check_1d_array, check_same_rows
from ._training import run_supervised_training_loop
from .layers import Dense
from .losses import CrossEntropyLoss, MSELoss


class _SequentialFit:
    def fit(self, X, y=None):
        X = self._check_input(X)
        self._build_optimizer()
        self._resolve_lr_scheduler()
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
            loss_fn = CrossEntropyLoss(from_logits=True)
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
                    loss += 0.5 * cfg.l2 * sum(np.sum(layer.W_ ** 2) for layer in self.layers if isinstance(layer, Dense))
                grad = loss_fn.grad(target, logits)
                self._backward(grad)

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
                evaluate_train_loss=lambda Xt, yt: self._evaluate_loss(Xt, yt, loss_fn, classes=classes),
                evaluate_val_loss=lambda Xt, yt: self._evaluate_loss(Xt, yt, loss_fn, classes=classes),
                snapshot_state=self._snapshot_layers,
                restore_state=self._restore_layers,
                on_epoch_end=self._on_epoch_end,
                callbacks=self.callbacks,
            )
            self.loss_ = result["loss"]
            self.val_loss_ = result["val_loss"]
            self.history_ = result["history"]
            self.best_epoch_ = result["best_epoch"]
            self.best_val_loss_ = result["best_val_loss"]
            return self

        if self.task == "regression":
            y = check_1d_array(y)
            X, y = check_same_rows(X, y)
            y = y.astype(float)
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
                yb = yb[:, None]
                pred = self._forward(xb)
                grad = loss_fn.grad(yb, pred)
                self._backward(grad)

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
                evaluate_train_loss=lambda Xt, yt: self._evaluate_loss(Xt, yt, loss_fn),
                evaluate_val_loss=lambda Xt, yt: self._evaluate_loss(Xt, yt, loss_fn),
                snapshot_state=self._snapshot_layers,
                restore_state=self._restore_layers,
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
