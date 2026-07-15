from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.0
    nesterov: bool = False
    rho: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass(frozen=True)
class TrainingConfig:
    n_iter: int = 1000
    batch_size: int | None = None
    lr: float = 0.01
    l2: float = 0.0
    tol: float = 1e-6
    random_state: int | None = None
    shuffle: bool = True
    validation_split: float = 0.0
    patience: int | None = None
    verbose: int = 0
    clip_norm: float | None = None
    accumulation_steps: int = 1
    metrics: tuple[str, ...] = ()

    def __post_init__(self):
        if int(self.accumulation_steps) < 1:
            raise ValueError("accumulation_steps must be at least 1")
        names = (self.metrics,) if isinstance(self.metrics, str) else tuple(self.metrics or ())
        object.__setattr__(self, "accumulation_steps", int(self.accumulation_steps))
        object.__setattr__(self, "metrics", names)


def _coerce_dict(value, cls):
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        return cls(**value)
    raise TypeError(f"Expected {cls.__name__} or dict, got {type(value).__name__}")


def resolve_optimizer_config(config=None, **overrides):
    base = _coerce_dict(config, OptimizerConfig)
    defaults = OptimizerConfig()
    if config is None:
        return replace(base, **{k: v for k, v in overrides.items() if v is not None})
    applied = {
        k: v
        for k, v in overrides.items()
        if v is not None and getattr(defaults, k, object()) != v
    }
    return replace(base, **applied)


def resolve_training_config(config=None, **overrides):
    base = _coerce_dict(config, TrainingConfig)
    defaults = TrainingConfig()
    if config is None:
        return replace(base, **{k: v for k, v in overrides.items() if v is not None})
    applied = {
        k: v
        for k, v in overrides.items()
        if v is not None and getattr(defaults, k, object()) != v
    }
    return replace(base, **applied)


def build_optimizer(config):
    from .optimizers import Adam, RMSProp, SGD

    cfg = _coerce_dict(config, OptimizerConfig)
    name = cfg.name.lower()
    if name == "sgd":
        return SGD(lr=cfg.lr, momentum=cfg.momentum, nesterov=cfg.nesterov)
    if name == "adam":
        return Adam(lr=cfg.lr, beta1=cfg.beta1, beta2=cfg.beta2, eps=cfg.eps)
    if name == "rmsprop":
        return RMSProp(lr=cfg.lr, rho=cfg.rho, eps=cfg.eps)
    raise ValueError("optimizer config name must be one of: sgd, adam, rmsprop")


OptCfg = OptimizerConfig
TrainCfg = TrainingConfig
resolve_opt_cfg = resolve_optimizer_config
resolve_train_cfg = resolve_training_config
build_opt = build_optimizer
