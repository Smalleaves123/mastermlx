from __future__ import annotations

import copy

import numpy as np

from ..utils.array import one_hot
from ..utils.grad import clip_grads
from .config import OptCfg, OptimizerConfig, build_opt, resolve_opt_cfg
from .layers import Dense, Embedding


class _SequentialRuntime:
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

    def _resolve_lr_scheduler(self):
        scheduler = self.lr_scheduler
        if scheduler is None:
            return None
        if callable(scheduler) and not hasattr(scheduler, "step"):
            scheduler = scheduler(self.optimizer_)
        if hasattr(scheduler, "optimizer"):
            scheduler.optimizer = self.optimizer_  # type: ignore[attr-defined]
            if hasattr(scheduler, "_base_lr"):
                scheduler._base_lr = getattr(self.optimizer_, "lr", scheduler._base_lr)
        if hasattr(scheduler, "_epoch"):
            scheduler._epoch = 0  # type: ignore[attr-defined]
        if hasattr(scheduler, "_wait"):
            scheduler._wait = 0  # type: ignore[attr-defined]
        if hasattr(scheduler, "best"):
            scheduler.best = np.inf if getattr(scheduler, "mode", "min") == "min" else -np.inf  # type: ignore[attr-defined]
        self.lr_scheduler = scheduler
        return scheduler

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

    def _on_epoch_end(self, epoch, logs):
        scheduler = self.lr_scheduler
        if scheduler is None:
            return
        monitor_loss = logs.get("monitor_loss")
        try:
            if monitor_loss is None:
                scheduler.step()
            else:
                scheduler.step(monitor_loss)
        except TypeError:
            scheduler.step()



__all__ = ["_SequentialRuntime"]
