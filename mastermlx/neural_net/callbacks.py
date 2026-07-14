from __future__ import annotations

import numpy as np


class Callback:
    """Base hook for supervised training events."""

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class History(Callback):
    """List-like record of epoch metrics."""

    def __init__(self):
        self.records = []

    def on_train_begin(self, logs=None):
        self.records = []

    def on_epoch_end(self, epoch, logs=None):
        row = {"epoch": int(epoch)}
        row.update(dict(logs or {}))
        self.records.append(row)

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, item):
        return self.records[item]

    def get(self, name, default=None):
        values = [row.get(name) for row in self.records]
        return values if values else default

    def keys(self):
        names = set()
        for row in self.records:
            names.update(row)
        return sorted(names)

    def to_dict(self):
        return {name: self.get(name) for name in self.keys()}


class EarlyStop(Callback):
    """Stop training when a monitored metric stops improving."""

    def __init__(self, monitor="val_loss", patience=5, min_delta=0.0, mode="min"):
        self.monitor = monitor
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.best = None
        self.wait = 0
        self.stop_training = False

    def on_train_begin(self, logs=None):
        self.best = None
        self.wait = 0
        self.stop_training = False

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get(self.monitor)
        if value is None or not np.isfinite(value):
            return
        value = float(value)
        if self.best is None:
            improved = True
        elif self.mode == "min":
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta
        if improved:
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True


__all__ = ["Callback", "EarlyStop", "History"]
