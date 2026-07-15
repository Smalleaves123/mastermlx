"""Shared MLP configuration, runtime, and training state."""

from __future__ import annotations

import copy
import numpy as np

from ..base import Module
from ..utils.array import one_hot
from ..utils.grad import accumulate_gradients, clip_grads, load_accumulated_gradients
from ..utils.validation import check_1d_array, check_2d_array
from .config import OptCfg, OptimizerConfig, build_opt, normalize_metrics, resolve_opt_cfg, resolve_train_cfg
from .layers import Dense, ReLU, Sigmoid, Tanh
from .metric_eval import evaluate_metrics


def _softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _make_activation(name):
    name = name.lower()
    if name == "relu":
        return ReLU(), "he"
    if name == "tanh":
        return Tanh(), "xavier"
    if name == "sigmoid":
        return Sigmoid(), "xavier"
    raise ValueError("activation must be one of: relu, tanh, sigmoid")


class _BaseMLP(Module):
    def __init__(
        self,
        hidden_layer_sizes=(32,),
        activation="relu",
        lr=0.1,
        lr_scheduler=None,
        n_iter=1000,
        batch_size=None,
        l2=0.0,
        tol=1e-6,
        random_state=None,
        optimizer=None,
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
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.l2 = l2
        self.tol = tol
        self.random_state = random_state
        self.optimizer = optimizer
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
                metrics=normalize_metrics(metrics),
                accumulation_steps=accumulation_steps,
            )
        else:
            overrides = {}
            if n_iter != 1000:
                overrides["n_iter"] = n_iter
            if batch_size is not None:
                overrides["batch_size"] = batch_size
            if lr != 0.1:
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
                overrides["metrics"] = normalize_metrics(metrics)
            if accumulation_steps != 1:
                overrides["accumulation_steps"] = accumulation_steps
            self.training_config_ = resolve_train_cfg(training_config, **overrides)
        self.optimizer_config_ = resolve_opt_cfg(optimizer_config, lr=self.training_config_.lr)
        self.layers_ = []
        self.loss_ = []
        self.val_loss_ = []
        self.optimizer_ = None
        self.best_epoch_ = None
        self.best_val_loss_ = None
        self.history_ = []

    def _regularization_penalty(self):
        l2 = self.training_config_.l2
        if not l2:
            return 0.0
        total = 0.0
        for layer in self.layers_:
            if isinstance(layer, Dense) and layer.W_ is not None:
                total += float(np.sum(layer.W_ ** 2))
        return 0.5 * l2 * total

    def _build_layers(self, n_features, n_outputs):
        self.layers_ = []
        dims = [n_features, *self.hidden_layer_sizes, n_outputs]
        for i in range(len(dims) - 2):
            act, init = _make_activation(self.activation)
            self.layers_.append(Dense(dims[i], dims[i + 1], random_state=None if self.random_state is None else self.random_state + i, init=init))
            self.layers_.append(act)
        self.layers_.append(Dense(dims[-2], dims[-1], random_state=None if self.random_state is None else self.random_state + len(dims), init="xavier"))

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

    def num_parameters(self):
        total = 0
        for layer in self.layers_:
            for name in ("W_", "b_"):
                value = getattr(layer, name, None)
                if value is not None:
                    total += int(np.size(value))
        return total

    def summary(self):
        return {
            "model": self.__class__.__name__,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation": self.activation,
            "num_parameters": self.num_parameters(),
            "training_config": self.training_config_,
        }

    def _forward(self, X):
        out = X
        for layer in self.layers_:
            out = layer.forward(out)
        return out

    def _backward(self, grad):
        return self._backward_weighted(grad, weight=1.0)

    def _backward_weighted(self, grad, weight=1.0):
        out = grad
        for layer in reversed(self.layers_):
            out = layer.backward(out)
        steps = max(1, int(self.training_config_.accumulation_steps))
        if not hasattr(self, "_grad_accum_"):
            self._reset_grad_accumulation()
        if steps > 1:
            accumulate_gradients(self.layers_, self._grad_accum_, weight=weight)
            self._accum_count_ += 1
            self._accum_weight_ += float(weight)
            if self._accum_count_ < steps:
                return out
            load_accumulated_gradients(self.layers_, self._grad_accum_, self._accum_weight_)
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
        load_accumulated_gradients(self.layers_, self._grad_accum_, self._accum_weight_)
        self._apply_gradients()
        self._reset_grad_accumulation()

    def _apply_gradients(self):
        clip_norm = self.training_config_.clip_norm
        if clip_norm is not None:
            clip_grads(self.layers_, clip_norm)
        dense_idx = 0
        l2 = self.training_config_.l2
        lr = self.training_config_.lr
        begin_step = getattr(self.optimizer_, "begin_step", None)
        end_step = getattr(self.optimizer_, "end_step", None)
        if begin_step is not None:
            begin_step()
        try:
            for layer in self.layers_:
                if isinstance(layer, Dense):
                    layer.step(lr=lr, l2=l2, optimizer=self.optimizer_, key_prefix=f"dense{dense_idx}")
                    dense_idx += 1
        finally:
            if end_step is not None:
                end_step()

    def _current_mode(self):
        for layer in self.layers_:
            if hasattr(layer, "training"):
                return bool(getattr(layer, "training"))
        return True

    def _set_mode(self, mode):
        for layer in self.layers_:
            if hasattr(layer, "train"):
                layer.train(mode)

    def _evaluate_loss(self, X, y, loss_fn, classes=None):
        was_training = self._current_mode()
        self._set_mode(False)
        try:
            out = self._forward(X)
            if classes is not None:
                target = one_hot(y, classes.size)
                loss = loss_fn(target, out)
            else:
                target = y[:, None] if np.asarray(y).ndim == 1 and out.ndim == 2 and out.shape[1] == 1 else y
                loss = loss_fn(target, out)
            return loss + self._regularization_penalty()
        finally:
            self._set_mode(was_training)

    def _evaluate_metrics(self, X, y, task, classes=None):
        was_training = self._current_mode()
        self._set_mode(False)
        try:
            out = self._forward(X)
            metric_y = y if task != "classification" else classes[np.asarray(y, dtype=int)]
            return evaluate_metrics(task, self.training_config_.metrics, metric_y, out, classes=classes)
        finally:
            self._set_mode(was_training)

    def _eval(self, X, y, task, loss_fn, classes=None, metrics=None):
        X = check_2d_array(X)
        was_training = self._current_mode()
        self._set_mode(False)
        try:
            output = self._forward(X)
            if task == "classification":
                if classes is None:
                    raise RuntimeError("model must be fitted before evaluate")
                target_labels = check_1d_array(y)
                if target_labels.shape[0] != X.shape[0] or not np.all(np.isin(target_labels, classes)):
                    raise ValueError("classification targets do not match the fitted model")
                indices = np.searchsorted(classes, target_labels)
                target = one_hot(indices, classes.size)
                metric_y = target_labels
            else:
                target = np.asarray(y, dtype=float)
                if target.ndim == 1 and output.ndim == 2 and output.shape[1] == 1:
                    target = target[:, None]
                if target.shape != output.shape:
                    raise ValueError("regression targets must match model output shape")
                metric_y = target
            value = loss_fn(target, output)
            specs = self.training_config_.metrics if metrics is None else normalize_metrics(metrics)
            values = evaluate_metrics(task, specs, metric_y, output, classes=classes)
            return {"loss": float(value + self._regularization_penalty()), "metrics": values}
        finally:
            self._set_mode(was_training)

    def _snapshot_layers(self):
        return copy.deepcopy(self.layers_)

    def _restore_layers(self, snapshot):
        self.layers_ = copy.deepcopy(snapshot)

    def _snapshot_state(self):
        return copy.deepcopy(self.layers_)

    def _restore_state(self, snapshot):
        self.layers_ = copy.deepcopy(snapshot)

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


__all__ = ["_BaseMLP", "_softmax"]
