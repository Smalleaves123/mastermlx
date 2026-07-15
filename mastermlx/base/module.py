"""Small module and parameter abstractions for NumPy neural networks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..version import __version__
from .checkpoint import load_object, save_object


def _name(name):
    return name[:-1] if name.endswith("_") else name


def _is_param(name, value):
    if value is None or name.startswith("d"):
        return False
    if not _looks_param_name(name):
        return False
    return isinstance(value, np.ndarray) or np.isscalar(value)


def _looks_param_name(name):
    key = _name(name)
    return (
        key in {"W", "U", "b", "u", "c", "gamma", "beta", "weight", "bias", "scale", "shift"}
        or key.startswith(("W_", "U_", "b_"))
    )


def _is_buffer(name, value):
    return value is not None and _name(name).startswith(("running_mean", "running_var"))


class Parameter:
    """A lightweight view over a module parameter.

    Existing layers keep using NumPy arrays internally.  ``Parameter`` exposes
    those arrays through a stable interface without forcing a tensor wrapper
    into every numerical operation.
    """

    def __init__(self, data=None, owner=None, name=None):
        self._data = None if data is None else np.asarray(data)
        self._owner = owner
        self._name = name

    @property
    def data(self):
        if self._owner is None:
            return self._data
        return getattr(self._owner, self._name)

    @data.setter
    def data(self, value):
        if self._owner is None:
            self._data = np.asarray(value)
        else:
            setattr(self._owner, self._name, value)

    @property
    def grad(self):
        if self._owner is None:
            return None
        return getattr(self._owner, "d" + self._name, None)

    @property
    def shape(self):
        return np.asarray(self.data).shape

    @property
    def size(self):
        return int(np.asarray(self.data).size)

    def __array__(self, dtype=None):
        return np.asarray(self.data, dtype=dtype)

    def __repr__(self):
        return f"Parameter(shape={self.shape}, dtype={np.asarray(self.data).dtype})"


class Module:
    """PyTorch-inspired state and mode interface for NumPy modules."""

    training = True

    def _own_parameters(self):
        return [(name, value) for name, value in vars(self).items() if _is_param(name, value)]

    def _own_buffers(self):
        return [(name, value) for name, value in vars(self).items() if _is_buffer(name, value)]

    def _children(self):
        for attr, value in vars(self).items():
            if isinstance(value, Module):
                yield attr, value
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    if isinstance(child, Module):
                        yield f"{attr}.{index}", child
            elif isinstance(value, dict):
                for key, child in value.items():
                    if isinstance(child, Module):
                        yield f"{attr}.{key}", child

    def named_parameters(self, prefix=""):
        for name, _ in self._own_parameters():
            yield f"{prefix}{_name(name)}", Parameter(owner=self, name=name)
        for child_name, child in self._children():
            yield from child.named_parameters(f"{prefix}{child_name}.")

    def parameters(self):
        return [parameter for _, parameter in self.named_parameters()]

    def named_buffers(self, prefix=""):
        for name, _ in self._own_buffers():
            yield f"{prefix}{_name(name)}", np.asarray(getattr(self, name))
        for child_name, child in self._children():
            yield from child.named_buffers(f"{prefix}{child_name}.")

    def _state_refs(self, prefix="", include_empty=False):
        refs = {}
        for name, value in vars(self).items():
            if _looks_param_name(name) and (include_empty or _is_param(name, value)):
                refs[f"{prefix}{_name(name)}"] = (self, name)
        for name, _ in self._own_buffers():
            refs[f"{prefix}{_name(name)}"] = (self, name)
        for child_name, child in self._children():
            refs.update(child._state_refs(f"{prefix}{child_name}.", include_empty=include_empty))
        return refs

    def state_dict(self):
        return {
            key: np.array(getattr(owner, name), copy=True)
            for key, (owner, name) in self._state_refs().items()
        }

    def load_state_dict(self, state, strict=True):
        if not hasattr(state, "items"):
            raise TypeError("state must be a mapping of parameter names to arrays")
        state = dict(state)
        refs = self._state_refs(include_empty=True)
        missing = sorted(set(refs) - set(state))
        unexpected = sorted(set(state) - set(refs))
        if strict and (missing or unexpected):
            raise ValueError(f"state mismatch: missing={missing}, unexpected={unexpected}")
        loaded = []
        for key, value in state.items():
            if key not in refs:
                continue
            owner, name = refs[key]
            current = getattr(owner, name)
            array = np.asarray(value)
            if current is None:
                setattr(owner, name, np.array(array, copy=True))
            elif np.isscalar(current):
                if array.size != 1:
                    raise ValueError(f"state[{key!r}] must be scalar")
                setattr(owner, name, array.reshape(-1)[0].item())
            else:
                if array.shape != current.shape:
                    raise ValueError(f"state[{key!r}] has shape {array.shape}, expected {current.shape}")
                setattr(owner, name, np.asarray(array, dtype=current.dtype).copy())
            loaded.append(key)
        return {"missing": missing, "unexpected": unexpected, "loaded": loaded}

    def train(self, mode=True):
        self.training = bool(mode)
        for _, child in self._children():
            child.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def save(self, path):
        """Save the state dictionary as a portable ``.npz`` file."""

        path = Path(path)
        target = path if path.suffix == ".npz" else Path(f"{path}.npz")
        np.savez(target, **self.state_dict())
        return target

    def load(self, path, strict=True):
        """Load a state dictionary from a ``.npz`` file."""

        with np.load(Path(path), allow_pickle=False) as archive:
            state = {key: archive[key] for key in archive.files}
        return self.load_state_dict(state, strict=strict)

    def save_checkpoint(self, path):
        """Save a versioned checkpoint without executable pickle payloads."""
        return save_object(self, path, __version__)

    @classmethod
    def load_checkpoint(cls, path):
        """Load a versioned checkpoint created by :meth:`save_checkpoint`."""
        obj = load_object(path)
        if not isinstance(obj, cls):
            raise TypeError(f"checkpoint must contain a {cls.__name__}")
        for callback in getattr(obj, "callbacks", ()):
            if hasattr(callback, "set_model"):
                callback.set_model(obj)
        return obj

    def extra_repr(self):
        return ""

    def summary(self):
        return {
            "module": self.__class__.__name__,
            "training": bool(getattr(self, "training", True)),
            "parameters": int(sum(parameter.size for parameter in self.parameters())),
        }


__all__ = ["Module", "Parameter"]
