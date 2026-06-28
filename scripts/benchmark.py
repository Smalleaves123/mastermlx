"""Shared helpers for benchmark scripts.

This file is intentionally small for now. It can later grow into a common
benchmark runner used by task-specific scripts.
"""

from __future__ import annotations

from pathlib import Path


def print_title(title: str):
    line = "=" * len(title)
    print(f"{title}\n{line}")


def ensure_output_dir(path="outputs"):
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
