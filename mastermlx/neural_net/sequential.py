from __future__ import annotations

import numpy as np

from ..base import BaseEstimator, Module
from ..utils.metrics import accuracy, r2_score
from .config import resolve_opt_cfg, resolve_train_cfg
from .seq_fit import _SequentialFit
from .seq_runtime import _SequentialRuntime


def _softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _sigmoid(logits):
    logits = np.asarray(logits, dtype=float)
    out = np.empty_like(logits)
    positive = logits >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    out[~positive] = exp_logits / (1.0 + exp_logits)
    return out


def _one_hot(y, n_classes):
    out = np.zeros((y.shape[0], n_classes), dtype=float)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


class Sequential(_SequentialRuntime, _SequentialFit, Module, BaseEstimator):
    """A minimal sequential neural network container."""

    def __init__(
        self,
        layers=None,
        loss="cross_entropy",
        optimizer="sgd",
        lr_scheduler=None,
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
        callbacks=None,
        metrics=None,
        accumulation_steps=1,
    ):
        self.layers = list(layers or [])
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr = lr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.l2 = l2
        self.tol = tol
        self.random_state = random_state
        self.task = task
        self.callbacks = list(callbacks or [])
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
                metrics=tuple(metrics or ()),
                accumulation_steps=accumulation_steps,
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
            if metrics is not None:
                overrides["metrics"] = tuple(metrics)
            if accumulation_steps != 1:
                overrides["accumulation_steps"] = accumulation_steps
            self.training_config_ = resolve_train_cfg(training_config, **overrides)
        self.optimizer_config_ = resolve_opt_cfg(optimizer_config, lr=self.training_config_.lr)
        self.loss_ = []
        self.val_loss_ = []
        self.optimizer_ = None
        self.classes_ = None
        self.best_epoch_ = None
        self.best_val_loss_ = None
        self.history_ = []


    def predict_proba(self, X):
        if self.task not in {"classification", "multilabel"}:
            raise RuntimeError("predict_proba is only available for classification tasks")
        X = self._check_input(X)
        was_training = self._current_mode()
        self._set_mode(False)
        try:
            logits = self._forward(X)
            probs = _softmax(logits) if self.task == "classification" else _sigmoid(logits)
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
            if self.task == "multilabel":
                pred = (_sigmoid(out) >= 0.5).astype(int)
                return pred[0] if pred.shape[0] == 1 else pred
            if self.task == "multioutput_regression":
                return out[0] if out.shape[0] == 1 else out
            pred = out.ravel()
            return float(pred[0]) if pred.shape[0] == 1 else pred
        finally:
            self._set_mode(was_training)

    def score(self, X, y):
        if self.task == "classification":
            return accuracy(y, self.predict(X))
        if self.task == "multilabel":
            pred = np.asarray(self.predict(X))
            target = np.asarray(y)
            return float(np.mean(np.all(pred == target, axis=1)))
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


__all__ = ["Sequential"]
