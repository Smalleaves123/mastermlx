"""Numerical optimization utilities."""

from .core import Result, finite_diff, gd, minimize, quad_gd

__all__ = ["Result", "finite_diff", "gd", "minimize", "quad_gd"]
