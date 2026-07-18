from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_signal(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        if suffix == ".npy":
            return np.asarray(np.load(path, allow_pickle=False), dtype=float)
        with np.load(path, allow_pickle=False) as data:
            if "signal" in data:
                return np.asarray(data["signal"], dtype=float)
            return np.asarray(data[data.files[0]], dtype=float)
    if suffix in {".csv", ".txt", ".tsv"}:
        delimiter = "," if suffix == ".csv" else "\t"
        return np.loadtxt(path, delimiter=delimiter, dtype=float)
    raise ValueError("Unsupported signal format. Use .npy, .npz, .csv, .txt, or .tsv")


def save_signal(path, signal, sample_rate=None):
    path = Path(path)
    signal = np.asarray(signal, dtype=float)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, signal)
        return path
    if suffix == ".npz":
        payload = {"signal": signal}
        if sample_rate is not None:
            payload["sample_rate"] = np.asarray(sample_rate)
        np.savez(path, **payload)
        return path
    if suffix in {".csv", ".txt", ".tsv"}:
        delimiter = "," if suffix == ".csv" else "\t"
        np.savetxt(path, signal, delimiter=delimiter)
        return path
    raise ValueError("Unsupported signal format. Use .npy, .npz, .csv, .txt, or .tsv")


def save_signal_bundle(path, signal, sample_rate=None, metadata=None):
    path = Path(path)
    signal = np.asarray(signal, dtype=float)
    payload = {"signal": signal}
    if sample_rate is not None:
        payload["sample_rate"] = np.asarray(sample_rate)
    if metadata is not None:
        payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez(path, **payload)
    return path


def load_signal_bundle(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        bundle: dict[str, Any] = {
            "signal": np.asarray(data["signal"], dtype=float),
            "sample_rate": None,
            "metadata": None,
        }
        if "sample_rate" in data.files:
            bundle["sample_rate"] = float(np.asarray(data["sample_rate"]).reshape(-1)[0])
        if "metadata_json" in data.files:
            bundle["metadata"] = json.loads(str(np.asarray(data["metadata_json"]).item()))
        return bundle


__all__ = ["load_signal", "load_signal_bundle", "save_signal", "save_signal_bundle"]
