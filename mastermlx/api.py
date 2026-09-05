"""Machine-readable public API stability information.

This module deliberately depends only on the static lazy-export registry, so
querying API status does not import NumPy or any domain implementation.
"""

from __future__ import annotations

from ._lazy_exports import EXPORTS, _PACKAGE_EXPORTS, _PUBLIC_SUBMODULES


STABILITY_LEVELS = ("stable", "beta", "experimental", "internal")

_STABLE_PACKAGES = {
    "mastermlx.base",
    "mastermlx.clustering",
    "mastermlx.data",
    "mastermlx.linear_models",
    "mastermlx.math_tools",
    "mastermlx.neighbors",
    "mastermlx.preprocessing",
    "mastermlx.trees",
}
_BETA_PACKAGES = {
    "mastermlx.anomaly",
    "mastermlx.decomposition",
    "mastermlx.ensemble",
    "mastermlx.manifold",
    "mastermlx.probabilistic",
    "mastermlx.selection",
    "mastermlx.semi_supervised",
    "mastermlx.signal",
    "mastermlx.svm",
    "mastermlx.tabular",
}

PACKAGE_STABILITY = {
    package: (
        "stable"
        if package in _STABLE_PACKAGES
        else "beta"
        if package in _BETA_PACKAGES
        else "experimental"
    )
    for package in _PACKAGE_EXPORTS
}

STABILITY_OVERRIDES = {
    "mastermlx.__version__": "stable",
    "mastermlx.get_backend": "stable",
    "mastermlx.set_backend": "stable",
    "mastermlx.ModelBundle": "beta",
    "mastermlx.permutation_importance": "beta",
    "mastermlx.partial_dependence": "beta",
}


def _resolve_public_name(name: str) -> tuple[str, str]:
    value = str(name).strip()
    if not value:
        raise KeyError("API name must be non-empty")
    if value.startswith("mastermlx."):
        value = value[len("mastermlx.") :]
    if "." in value:
        package_name, symbol = value.rsplit(".", 1)
        if symbol.startswith("_"):
            return f"mastermlx.{package_name}", symbol
        owner = f"mastermlx.{package_name}"
        if symbol not in _PACKAGE_EXPORTS.get(owner, ()):
            raise KeyError(f"unknown public API: {name!r}")
        return owner, symbol
    owner = EXPORTS.get(value)
    if owner is None:
        if value in _PUBLIC_SUBMODULES:
            return f"mastermlx.{value}", value
        raise KeyError(f"unknown public API: {name!r}")
    return owner, value


def get_api_stability(name: str) -> str:
    """Return ``stable``, ``beta``, ``experimental``, or ``internal``."""

    owner, symbol = _resolve_public_name(name)
    value = str(name).strip().removeprefix("mastermlx.")
    qualified = f"{owner}.{symbol}" if "." in value else f"mastermlx.{symbol}"
    override = STABILITY_OVERRIDES.get(
        qualified, STABILITY_OVERRIDES.get(f"mastermlx.{symbol}")
    )
    if override is not None:
        return override
    if symbol.startswith("_"):
        return "internal"
    if owner in PACKAGE_STABILITY:
        return PACKAGE_STABILITY[owner]
    if owner in {"mastermlx.version", "mastermlx.config", "mastermlx.utils"}:
        return "stable"
    return "experimental"


def api_stability_report() -> dict[str, tuple[str, ...]]:
    """Group every top-level public name by its current stability tier."""

    grouped: dict[str, list[str]] = {level: [] for level in STABILITY_LEVELS}
    for name in EXPORTS:
        grouped[get_api_stability(name)].append(name)
    return {level: tuple(sorted(names)) for level, names in grouped.items()}


__all__ = [
    "PACKAGE_STABILITY",
    "STABILITY_LEVELS",
    "STABILITY_OVERRIDES",
    "api_stability_report",
    "get_api_stability",
]
