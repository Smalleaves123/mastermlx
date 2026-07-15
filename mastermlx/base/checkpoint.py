"""Safe, versioned serialization for fitted mastermlx objects."""

from __future__ import annotations

import base64
import importlib
import io
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np


SCHEMA_VERSION = 1
FORMAT = "mastermlx-checkpoint"
STATE_SCHEMA = "object_graph"
STATE_SCHEMA_VERSION = 1


def _class_path(value):
    cls = value.__class__
    module = cls.__module__
    qualname = cls.__qualname__
    if not (module == "mastermlx" or module.startswith("mastermlx.")) or "<locals>" in qualname:
        raise TypeError(f"unsupported checkpoint class: {module}.{qualname}")
    return module, qualname


def _resolve_class(module, qualname):
    if not (module == "mastermlx" or module.startswith("mastermlx.")) or "<locals>" in qualname:
        raise ValueError("checkpoint contains a disallowed class")
    value = importlib.import_module(module)
    for name in qualname.split("."):
        value = getattr(value, name)
    return value


class _Encoder:
    def __init__(self):
        self.arrays = {}
        self.seen = {}
        self.next_ref = 0

    def _ref(self, value):
        key = id(value)
        if key in self.seen:
            return {"kind": "ref", "ref": self.seen[key]}
        ref = str(self.next_ref)
        self.next_ref += 1
        self.seen[key] = ref
        return ref

    def encode(self, value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, np.generic):
            return {"kind": "scalar", "dtype": value.dtype.str, "value": value.item()}
        if isinstance(value, Path):
            return {"kind": "path", "value": str(value)}
        if isinstance(value, bytes):
            return {"kind": "bytes", "value": base64.b64encode(value).decode("ascii")}
        if isinstance(value, np.random.Generator):
            ref = self._ref(value)
            if isinstance(ref, dict):
                return ref
            return {"kind": "rng", "ref": ref, "state": self.encode(value.bit_generator.state)}
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("object arrays are not supported in checkpoints")
            ref = self._ref(value)
            if isinstance(ref, dict):
                return ref
            self.arrays[ref] = np.array(value, copy=True)
            return {"kind": "array", "ref": ref}
        if isinstance(value, dict):
            ref = self._ref(value)
            if isinstance(ref, dict):
                return ref
            return {
                "kind": "dict",
                "ref": ref,
                "items": [[self.encode(key), self.encode(item)] for key, item in value.items()],
            }
        if isinstance(value, list):
            ref = self._ref(value)
            if isinstance(ref, dict):
                return ref
            return {"kind": "list", "ref": ref, "items": [self.encode(item) for item in value]}
        if isinstance(value, tuple):
            return {"kind": "tuple", "items": [self.encode(item) for item in value]}
        if isinstance(value, set):
            return {"kind": "set", "items": [self.encode(item) for item in value]}
        if callable(value):
            raise TypeError("callables cannot be serialized in a safe checkpoint")
        if not hasattr(value, "__dict__"):
            raise TypeError(f"unsupported checkpoint value: {type(value).__name__}")

        module, qualname = _class_path(value)
        ref = self._ref(value)
        if isinstance(ref, dict):
            return ref
        return {
            "kind": "object",
            "ref": ref,
            "module": module,
            "qualname": qualname,
            "attrs": {name: self.encode(item) for name, item in vars(value).items()},
        }


class _Decoder:
    def __init__(self, arrays):
        self.arrays = arrays
        self.refs = {}

    def decode(self, node):
        if node is None or isinstance(node, (bool, int, float, str)):
            return node
        kind = node.get("kind")
        if kind == "ref":
            return self.refs[node["ref"]]
        if kind == "scalar":
            return np.asarray(node["value"], dtype=np.dtype(node["dtype"])).reshape(()).item()
        if kind == "path":
            return Path(node["value"])
        if kind == "bytes":
            return base64.b64decode(node["value"])
        if kind == "array":
            array_value = np.array(self.arrays[node["ref"]], copy=True)
            self.refs[node["ref"]] = array_value
            return array_value
        if kind == "rng":
            rng_value = np.random.default_rng()
            self.refs[node["ref"]] = rng_value
            rng_value.bit_generator.state = self.decode(node["state"])
            return rng_value
        if kind == "dict":
            dict_value: dict[Any, Any] = {}
            self.refs[node["ref"]] = dict_value
            for key, item in node["items"]:
                dict_value[self.decode(key)] = self.decode(item)
            return dict_value
        if kind == "list":
            list_value: list[Any] = []
            self.refs[node["ref"]] = list_value
            list_value.extend(self.decode(item) for item in node["items"])
            return list_value
        if kind == "tuple":
            return tuple(self.decode(item) for item in node["items"])
        if kind == "set":
            return set(self.decode(item) for item in node["items"])
        if kind == "object":
            cls = _resolve_class(node["module"], node["qualname"])
            value = object.__new__(cls)
            self.refs[node["ref"]] = value
            for name, item in node["attrs"].items():
                object.__setattr__(value, name, self.decode(item))
            return value
        raise ValueError(f"unknown checkpoint node kind: {kind!r}")


def save_object(value, path, library_version):
    """Write a trusted object graph without executable pickle payloads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoder = _Encoder()
    root = encoder.encode(value)
    manifest = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "state_schema": {"name": STATE_SCHEMA, "version": STATE_SCHEMA_VERSION},
        "library_version": str(library_version),
        "root": root,
        "arrays": sorted(encoder.arrays),
    }
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, separators=(",", ":")))
            for ref, array in encoder.arrays.items():
                buffer = io.BytesIO()
                np.save(buffer, array, allow_pickle=False)
                archive.writestr(f"arrays/{ref}.npy", buffer.getvalue())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def load_object(path):
    """Load a versioned checkpoint and reject unsupported formats."""
    path = Path(path)
    if not zipfile.is_zipfile(path):
        raise ValueError("unsupported checkpoint format; expected a versioned mastermlx archive")
    with zipfile.ZipFile(path) as archive:
        try:
            manifest = json.loads(archive.read("manifest.json"))
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError("checkpoint manifest is missing or invalid") from exc
        if manifest.get("format") != FORMAT:
            raise ValueError("unsupported checkpoint format")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported checkpoint schema: {manifest.get('schema_version')!r}"
            )
        state_schema = manifest.get("state_schema")
        if state_schema != {"name": STATE_SCHEMA, "version": STATE_SCHEMA_VERSION}:
            raise ValueError(f"unsupported checkpoint state schema: {state_schema!r}")
        arrays = {}
        for ref in manifest.get("arrays", ()):
            try:
                arrays[ref] = np.load(
                    io.BytesIO(archive.read(f"arrays/{ref}.npy")), allow_pickle=False
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(f"invalid checkpoint array: {ref}") from exc
    return _Decoder(arrays).decode(manifest["root"])
