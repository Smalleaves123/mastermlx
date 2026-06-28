from __future__ import annotations

import copy
import numpy as np

from ..base import BaseEstimator
from ..data.split import train_test_split
from ..utils.array import one_hot
from ..utils.grad import clip_grads
from ..utils.metrics import accuracy, r2_score
from ..utils.validation import as_2d, check_1d_array, check_same_rows
from .config import OptCfg, OptimizerConfig, build_opt, resolve_opt_cfg, resolve_train_cfg
from .layers import Dense, Embedding
from .losses import CrossEntropyLoss, MSELoss
from .optimizers import Adam, RMSProp, SGD
from ._training import run_supervised_training_loop


def _softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _one_hot(y, n_classes):
    out = np.zeros((y.shape[0], n_classes), dtype=float)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


class Sequential(BaseEstimator):
    """A minimal sequential neural network container."""

    def __init__(
        self,
        layers=None,
        loss="cross_entropy",
        optimizer="sgd",
        lr=0.01,
        n_iter=1000,
        batch_size=None,
        l2=0.0,
        tol=1e-6,
        random_state=None,
        task="classification",
        training_config=None,
        optimizer_config=None,
        shuffle=True,
        validation_split=0.0,
        patience=None,
        verbose=0,
    ):
        self.layers = list(layers or [])
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.l2 = l2
        self.tol = tol
        self.random_state = random_state
        self.task = task
        if training_config is None:
            self.training_config_ = resolve_train_cfg(
                None,
                n_iter=n_iter,
                batch_size=batch_size,
                lr=lr,
                l2=l2,
                tol=tol,
                random_state=random_state,
                shuffle=shuffle,
                validation_split=validation_split,
                patience=patience,
                verbose=verbose,
            )
        else:
            overrides = {}
            if n_iter != 1000:
                overrides["n_iter"] = n_iter
            if batch_size is not None:
                overrides["batch_size"] = batch_size
            if lr != 0.01:
                overrides["lr"] = lr
            if l2 != 0.0:
                overrides["l2"] = l2
            if tol != 1e-6:
                overrides["tol"] = tol
            if random_state is not None:
                overrides["random_state"] = random_state
            if shuffle is not True:
                overrides["shuffle"] = shuffle
            if validation_split != 0.0:
                overrides["validation_split"] = validation_split
            if patience is not None:
                overrides["patience"] = patience
            if verbose != 0:
                overrides["verbose"] = verbose
            self.training_config_ = resolve_train_cfg(training_config, **overrides)
        self.optimizer_config_ = resolve_opt_cfg(optimizer_config, lr=self.training_config_.lr)
        self.loss_ = []
        self.val_loss_ = []
        self.optimizer_ = None
        self.classes_ = None
        self.best_epoch_ = None
        self.best_val_loss_ = None
        self.history_ = []

    def _regularization_penalty(self):
        l2 = self.training_config_.l2
        if not l2:
            return 0.0
        return 0.5 * l2 * sum(np.sum(layer.W_ ** 2) for layer in self.layers if isinstance(layer, Dense))

    def _current_mode(self):
        for layer in self.layers:
            if hasattr(layer, "training"):
                return bool(getattr(layer, "training"))
        return True

    def _set_mode(self, mode):
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train(mode)

    def _check_input(self, X):
        X = np.asarray(X)
        if X.size == 0:
            raise ValueError("Expected a non-empty array")
        if X.ndim not in {2, 3}:
            raise ValueError(f"Expected 2D or 3D array, got shape {X.shape}")
        return X

    def add(self, layer):
        self.layers.append(layer)
        return self

    def train(self, mode=True):
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def _build_optimizer(self):
        if self.optimizer is None:
            self.optimizer_ = build_opt(resolve_opt_cfg(self.optimizer_config_, name="sgd", lr=self.training_config_.lr))
        elif isinstance(self.optimizer, str):
            cfg = resolve_opt_cfg(self.optimizer_config_, name=self.optimizer, lr=self.training_config_.lr)
            self.optimizer_ = build_opt(cfg)
        elif isinstance(self.optimizer, (dict, OptCfg, OptimizerConfig)):
            self.optimizer_ = build_opt(self.optimizer)
        else:
            self.optimizer_ = self.optimizer

    def _forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def _backward(self, grad):
        out = grad
        for layer in reversed(self.layers):
            out = layer.backward(out)

        clip_norm = self.training_config_.clip_norm
        if clip_norm is not None:
            clip_grads(self.layers, clip_norm)

        dense_idx = 0
        l2 = self.training_config_.l2
        lr = self.training_config_.lr
        for layer in self.layers:
            if hasattr(layer, "step"):
                if isinstance(layer, (Dense, Embedding)):
                    layer.step(optimizer=self.optimizer_, key_prefix=f"dense{dense_idx}", l2=l2, lr=lr)
                    dense_idx += 1
                else:
                    layer.step(lr=lr, optimizer=self.optimizer_, key_prefix=layer.__class__.__name__.lower())

    def _evaluate_loss(self, X, y, loss_fn, classes=None):
        was_training = self._current_mode()
        self._set_mode(False)
        try:
            out = self._forward(X)
            if self.task == "classification":
                target = one_hot(y, classes.size)
            else:
                target = y
            loss = loss_fn(target, out if self.task == "classification" else out.ravel())
            return loss + self._regularization_penalty()
        finally:
            self._set_mode(was_training)

    def _snapshot_layers(self):
        return copy.deepcopy(self.layers)

    def _restore_layers(self, snapshot):
        self.layers = copy.deepcopy(snapshot)

    def fit(self, X, y=None):
        X = self._check_input(X)
        self._build_optimizer()
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
            )
            self.loss_ = result["loss"]
            self.val_loss_ = result["val_loss"]
            self.history_ = result["history"]
            self.best_epoch_ = result["best_epoch"]
            self.best_val_loss_ = result["best_val_loss"]
            return self

        raise ValueError("task must be 'classification' or 'regression'")

    def predict_proba(self, X):
        if self.task != "classification":
            raise RuntimeError("predict_proba is only available for classification")
        X = self._check_input(X)
        was_training = self._current_mode()
        self._set_mode(False)
        try:
            logits = self._forward(X)
            probs = _softmax(logits)
            return probs[0] if probs.shape[0] == 1 else probs
        finally:
            self._set_mode(was_training)

    def predict(self, X):
        X = self._check_input(X)
        was_training = self._current_mode()
        self._set_mode(False)
        try:
            out = self._forward(X)
            if self.task == "classification":
                probs = _softmax(out)
                idx = np.argmax(probs, axis=1)
                pred = self.classes_[idx]
                return pred[0] if pred.shape[0] == 1 else pred
            pred = out.ravel()
            return float(pred[0]) if pred.shape[0] == 1 else pred
        finally:
            self._set_mode(was_training)

    def score(self, X, y):
        if self.task == "classification":
            return accuracy(y, self.predict(X))
        return r2_score(y, self.predict(X))

    def num_parameters(self):
        total = 0
        for layer in self.layers:
            for name in ("W_", "b_", "gamma_", "beta_"):
                value = getattr(layer, name, None)
                if value is not None:
                    total += int(np.size(value))
        return total

    def summary(self):
        return {
            "model": self.__class__.__name__,
            "task": self.task,
            "n_layers": len(self.layers),
            "num_parameters": self.num_parameters(),
            "optimizer": getattr(self.optimizer_, "__class__", type(self.optimizer_)).__name__ if self.optimizer_ is not None else None,
            "training_config": self.training_config_,
        }
