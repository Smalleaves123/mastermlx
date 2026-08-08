"""Shared predicates for selecting optional compiled implementations."""

from __future__ import annotations

from ..config import get_backend


def use_cpp_backend(backend=None):
    """Return whether C++ kernels may be selected for this request."""

    selected = get_backend() if backend is None else str(backend).lower()
    return selected == "auto"


def use_cython_backend(backend=None):
    """Return whether Cython kernels may be selected for this request."""

    selected = get_backend() if backend is None else str(backend).lower()
    return selected in {"auto", "cython"}


__all__ = ["use_cpp_backend", "use_cython_backend"]
