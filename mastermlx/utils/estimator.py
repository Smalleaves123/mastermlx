from __future__ import annotations

from copy import deepcopy


def clone(obj):
    return deepcopy(obj)


def get_params(obj, deep=True):
    params = {}
    for name, value in vars(obj).items():
        if name.endswith("_") or name.startswith("_"):
            continue
        if callable(value):
            continue
        params[name] = value
        if deep and hasattr(value, "get_params"):
            nested = value.get_params()
            for key, nested_value in nested.items():
                params[f"{name}__{key}"] = nested_value
    return params


def set_params(obj, **params):
    if not params:
        return obj

    for key, value in params.items():
        if "__" not in key:
            setattr(obj, key, value)
            continue
        name, subkey = key.split("__", 1)
        if not hasattr(obj, name):
            raise ValueError(f"Unknown parameter '{name}'")
        nested = getattr(obj, name)
        if not hasattr(nested, "set_params"):
            raise ValueError(f"Parameter '{name}' does not support nested parameters")
        nested.set_params(**{subkey: value})
    return obj
