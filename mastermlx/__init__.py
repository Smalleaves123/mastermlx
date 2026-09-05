"""mastermlx: a NumPy-first machine learning playground.

The top-level facade is intentionally lazy. Importing :mod:`mastermlx` loads
only version/configuration metadata; domain packages are imported when their
public names are first accessed.
"""

from importlib import import_module as _import_module
from typing import TYPE_CHECKING

from ._lazy_exports import (
    _AMBIGUOUS_EXPORTS,
    _PUBLIC_SUBMODULES,
    EXPORTS as _EXPORTS,
    PUBLIC_NAMES as _PUBLIC_NAMES,
)
from .config import get_backend as get_backend, set_backend as set_backend
from .version import __version__ as __version__


if TYPE_CHECKING:
    # Keep static analyzers aware of the broad compatibility facade without
    # paying the runtime import cost. Runtime ownership comes from the static
    # registry, including the explicit LDA ambiguity rule.
    from .anomaly import *  # noqa: F401,F403
    from .bandits import *  # noqa: F401,F403
    from .base import *  # noqa: F401,F403
    from .clustering import *  # noqa: F401,F403
    from .control import *  # noqa: F401,F403
    from .data import *  # noqa: F401,F403
    from .decomposition import *  # noqa: F401,F403
    from .ensemble import *  # noqa: F401,F403
    from .estimation import *  # noqa: F401,F403
    from .graphs import *  # noqa: F401,F403
    from .linear_models import *  # noqa: F401,F403
    from .manifold import *  # noqa: F401,F403
    from .math_tools import *  # noqa: F401,F403
    from .neighbors import *  # noqa: F401,F403
    from .neural_net import *  # noqa: F401,F403
    from .nlp import *  # noqa: F401,F403
    from .optimize import *  # noqa: F401,F403
    from .planning import *  # noqa: F401,F403
    from .preprocessing import *  # noqa: F401,F403
    from .probabilistic import *  # type: ignore[assignment]  # noqa: F401,F403
    from .rl import *  # noqa: F401,F403
    from .robotics import *  # noqa: F401,F403
    from .selection import *  # noqa: F401,F403
    from .semi_supervised import *  # noqa: F401,F403
    from .signal import *  # noqa: F401,F403
    from .svm import *  # noqa: F401,F403
    from .tabular import *  # noqa: F401,F403
    from .trees import *  # noqa: F401,F403
    from .variational import *  # noqa: F401,F403
    from .vision import *  # noqa: F401,F403
    from .viz import *  # noqa: F401,F403


__all__ = list(_PUBLIC_NAMES)
_SHADOWED_SUBMODULES = set(_PUBLIC_SUBMODULES).intersection(_EXPORTS)


def __getattr__(name):
    choices = _AMBIGUOUS_EXPORTS.get(name)
    if choices is not None:
        raise AttributeError(
            f"mastermlx.{name} is ambiguous; import one of: {', '.join(choices)}"
        )

    module_name = _EXPORTS.get(name)
    if module_name is not None:
        module = _import_module(module_name)
        value = getattr(module, name)
        globals()[name] = value
        # Importing ``mastermlx.utils`` temporarily binds that submodule on
        # the package. ``utils`` is also a historical signal export, so keep
        # the documented top-level binding lazy and order-independent.
        for shadowed_name in _SHADOWED_SUBMODULES.difference({name}):
            globals().pop(shadowed_name, None)
        return value

    if name in _PUBLIC_SUBMODULES:
        module = _import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    raise AttributeError(f"module 'mastermlx' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()).union(__all__).union(_PUBLIC_SUBMODULES))
