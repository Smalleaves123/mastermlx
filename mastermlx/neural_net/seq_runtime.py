from __future__ import annotations

import copy
from typing import Any

import numpy as np

from ..utils.array import one_hot
from ..utils.grad import accumulate_gradients, clip_grads, load_accumulated_gradients
from .config import OptCfg, OptimizerConfig, build_opt, normalize_metrics, resolve_opt_cfg
from .layers import Dense, Embedding
from .losses import BinaryCrossEntropyLoss, CrossEntropyLoss, MSELoss
from .metric_eval import evaluate_metrics


class _SequentialRuntime:
    callbacks: list[Any]
    classes_: np.ndarray | None
    layers: list[Any]
    lr_scheduler: Any
    optimizer: Any
    optimizer_: Any
    optimizer_config_: Any
    task: str
    training_config_: Any

    def _regularization_penalty(self):
        l2 = float(self.training_config_.l2)
        if not l2:
            return 0.0
        return 0.5 * l2 * sum(
            np.sum(layer.W_**2)
            for layer in self.layers
            if isinstance(layer, Dense) and layer.W_ is not None
        )

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
        return self._backward_weighted(grad, weight=1.0)

    def _backward_weighted(self, grad, weight=1.0):
        out = grad
        for layer in reversed(self.layers):
            out = layer.backward(out)

        steps = max(1, int(self.training_config_.accumulation_steps))
        if not hasattr(self, "_grad_accum_"):
            self._reset_grad_accumulation()
        if steps > 1:
            accumulate_gradients(self.layers, self._grad_accum_, weight=weight)
            self._accum_count_ += 1
            self._accum_weight_ += float(weight)
            if self._accum_count_ < steps:
                return out
            load_accumulated_gradients(self.layers, self._grad_accum_, self._accum_weight_)
            self._apply_gradients()
            self._reset_grad_accumulation()
            return out
        self._apply_gradients()
        return out

    def _reset_grad_accumulation(self):
        self._grad_accum_ = {}
        self._accum_count_ = 0
        self._accum_weight_ = 0.0

    def _flush_accumulation(self):
        if getattr(self, "_accum_count_", 0) < 1:
            return
        load_accumulated_gradients(self.layers, self._grad_accum_, self._accum_weight_)
        self._apply_gradients()
        self._reset_grad_accumulation()

    def _apply_gradients(self):
        clip_norm = self.training_config_.clip_norm
        if clip_norm is not None:
            clip_grads(self.layers, clip_norm)

        dense_idx = 0
        l2 = self.training_config_.l2
        lr = self.training_config_.lr
        begin_step = getattr(self.optimizer_, "begin_step", None)
        end_step = getattr(self.optimizer_, "end_step", None)
        if begin_step is not None:
            begin_step()
        try:
            for layer in self.layers:
                if hasattr(layer, "step"):
                    if isinstance(layer, (Dense, Embedding)):
                        layer.step(optimizer=self.optimizer_, key_prefix=f"dense{dense_idx}", l2=l2, lr=lr)
                        dense_idx += 1
                    else:
                        layer.step(lr=lr, optimizer=self.optimizer_, key_prefix=layer.__class__.__name__.lower())
        finally:
            if end_step is not None:
                end_step()

    def _evaluate_loss(self, X, y, loss_fn, classes=None):
        was_training = self._current_mode()
        self._set_mode(False)
        try:
            out = self._forward(X)
            if self.task == "classification":
                target = one_hot(y, classes.size)
                value = out
            else:
                target = y[:, None] if np.asarray(y).ndim == 1 and out.ndim == 2 and out.shape[1] == 1 else y
                value = out
            loss = loss_fn(target, value)
            return loss + self._regularization_penalty()
        finally:
            self._set_mode(was_training)

    def _evaluate_metrics(self, X, y):
        was_training = self._current_mode()
        self._set_mode(False)
        try:
            output = self._forward(X)
            metric_y = y
            if self.task == "classification" and self.classes_ is not None:
                metric_y = self.classes_[np.asarray(y, dtype=int)]
            return evaluate_metrics(
                self.task,
                self.training_config_.metrics,
                metric_y,
                output,
                classes=self.classes_,
            )
        finally:
            self._set_mode(was_training)

    def _snapshot_layers(self):
        return copy.deepcopy(self.layers)

    def _restore_layers(self, snapshot):
        self.layers = copy.deepcopy(snapshot)

    def _snapshot_state(self):
        return copy.deepcopy(self.layers)

    def _restore_state(self, snapshot):
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

    def evaluate(self, X, y, metrics=None):
        """Return a consistent ``{"loss": ..., "metrics": ...}`` result."""
        X = self._check_input(X)
        was_training = self._current_mode()
        self._set_mode(False)
        try:
            output = self._forward(X)
            if self.task == "classification":
                if self.classes_ is None:
                    raise RuntimeError("model must be fitted before evaluate")
                target_labels = np.asarray(y)
                if target_labels.ndim != 1 or target_labels.shape[0] != X.shape[0]:
                    raise ValueError("classification targets must be 1D with one value per sample")
                if not np.all(np.isin(target_labels, self.classes_)):
                    raise ValueError("y contains labels outside the fitted classes")
                indices = np.searchsorted(self.classes_, target_labels)
                target = one_hot(indices, self.classes_.size)
                loss = CrossEntropyLoss(from_logits=True)(target, output)
                metric_y = target_labels
                classes = self.classes_
            elif self.task == "multilabel":
                target = np.asarray(y, dtype=float)
                if target.ndim == 1:
                    target = target[:, None]
                if target.ndim != 2 or target.shape != output.shape:
                    raise ValueError("multilabel targets must match model output shape")
                loss = BinaryCrossEntropyLoss(from_logits=True)(target, output)
                metric_y = target
                classes = None
            elif self.task in {"regression", "multioutput_regression"}:
                target = np.asarray(y, dtype=float)
                if target.ndim == 1 and output.ndim == 2 and output.shape[1] == 1:
                    target = target[:, None]
                if target.shape != output.shape:
                    raise ValueError("regression targets must match model output shape")
                loss = MSELoss()(target, output)
                metric_y = target
                classes = None
            else:
                raise ValueError(f"Unsupported task: {self.task}")

            specs = self.training_config_.metrics if metrics is None else normalize_metrics(metrics)
            values = evaluate_metrics(self.task, specs, metric_y, output, classes=classes)
            return {"loss": float(loss + self._regularization_penalty()), "metrics": values}
        finally:
            self._set_mode(was_training)



__all__ = ["_SequentialRuntime"]
