"""Utilities for implementing the public API deprecation policy."""

from __future__ import annotations

import functools
import warnings


def deprecated(replacement, *, since=None):
    """Mark a callable as deprecated while preserving its signature metadata."""

    replacement = str(replacement)

    def decorator(function):
        detail = f" since {since}" if since is not None else ""
        message = (
            f"{function.__module__}.{function.__name__} is deprecated{detail}; "
            f"use {replacement}"
        )

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return function(*args, **kwargs)

        return wrapper

    return decorator


__all__ = ["deprecated"]
