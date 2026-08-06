from __future__ import annotations

from copy import deepcopy
import inspect
from typing import Any, TypeVar


EstimatorT = TypeVar("EstimatorT")


def _is_estimator_like(value):
    return callable(getattr(value, "fit", None)) and (
        callable(getattr(value, "predict", None))
        or callable(getattr(value, "score", None))
    )


def _clone_value(value):
    if hasattr(value, "get_params") or _is_estimator_like(value):
        return clone(value)
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    return deepcopy(value)


def clone(obj: EstimatorT) -> EstimatorT:
    """Construct an unfitted estimator with the same public parameters.

    Objects implementing ``get_params`` are reconstructed through their
    constructor so learned attributes are never copied into cross-validation
    folds. Plain Python estimator-like objects retain the historical deepcopy
    fallback for compatibility.
    """

    if not hasattr(obj, "get_params") and not _is_estimator_like(obj):
        return deepcopy(obj)
    params = obj.get_params(deep=False) if hasattr(obj, "get_params") else get_params(obj, deep=False)
    cloned_params = {name: _clone_value(value) for name, value in params.items()}
    try:
        cloned = type(obj)(**cloned_params)
    except TypeError as exc:
        raise TypeError(
            f"{type(obj).__name__} cannot be cloned from its public parameters"
        ) from exc
    return cloned


def get_params(obj: Any, deep: bool = True) -> dict[str, Any]:
    params: dict[str, Any] = {}
    try:
        signature = inspect.signature(obj.__init__)
    except (AttributeError, TypeError, ValueError):
        signature = None
    if signature is not None:
        names = [
            name
            for name, parameter in signature.parameters.items()
            if name != "self"
            and parameter.kind
            in {parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY}
        ]
    else:
        names = [name for name in vars(obj) if not name.startswith("_") and not name.endswith("_")]
    for name in names:
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        params[name] = value
        if deep and hasattr(value, "get_params"):
            nested = value.get_params(deep=True)
            for key, nested_value in nested.items():
                params[f"{name}__{key}"] = nested_value
    return params


def set_params(obj: EstimatorT, **params: Any) -> EstimatorT:
    if not params:
        return obj

    valid_params = get_params(obj, deep=False)
    for key, value in params.items():
        if "__" not in key:
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter '{key}' for estimator {type(obj).__name__}"
                )
            setattr(obj, key, value)
            continue
        name, subkey = key.split("__", 1)
        if name not in valid_params:
            raise ValueError(f"Unknown parameter '{name}'")
        nested = getattr(obj, name)
        if not hasattr(nested, "set_params"):
            raise ValueError(f"Parameter '{name}' does not support nested parameters")
        nested.set_params(**{subkey: value})
    return obj
