"""Generate or validate the static registry used by ``mastermlx`` lazy imports."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PACKAGE_MODULES = [
    "mastermlx.base",
    "mastermlx.anomaly",
    "mastermlx.bandits",
    "mastermlx.data",
    "mastermlx.decomposition",
    "mastermlx.ensemble",
    "mastermlx.control",
    "mastermlx.linear_models",
    "mastermlx.manifold",
    "mastermlx.estimation",
    "mastermlx.neighbors",
    "mastermlx.neural_net",
    "mastermlx.nlp",
    "mastermlx.probabilistic",
    "mastermlx.rl",
    "mastermlx.selection",
    "mastermlx.semi_supervised",
    "mastermlx.signal",
    "mastermlx.tabular",
    "mastermlx.robotics",
    "mastermlx.svm",
    "mastermlx.trees",
    "mastermlx.variational",
    "mastermlx.viz",
    "mastermlx.vision",
    "mastermlx.preprocessing",
    "mastermlx.math_tools",
    "mastermlx.graphs",
    "mastermlx.optimize",
    "mastermlx.planning",
    "mastermlx.clustering",
]

PUBLIC_EXPORT_ORDER = [
    "mastermlx.base",
    "mastermlx.anomaly",
    "mastermlx.bandits",
    "mastermlx.data",
    "mastermlx.decomposition",
    "mastermlx.ensemble",
    "mastermlx.control",
    "mastermlx.estimation",
    "mastermlx.linear_models",
    "mastermlx.math_tools",
    "mastermlx.graphs",
    "mastermlx.optimize",
    "mastermlx.planning",
    "mastermlx.manifold",
    "mastermlx.neighbors",
    "mastermlx.neural_net",
    "mastermlx.nlp",
    "mastermlx.probabilistic",
    "mastermlx.rl",
    "mastermlx.preprocessing",
    "mastermlx.selection",
    "mastermlx.semi_supervised",
    "mastermlx.signal",
    "mastermlx.tabular",
    "mastermlx.robotics",
    "mastermlx.svm",
    "mastermlx.trees",
    "mastermlx.variational",
    "mastermlx.viz",
    "mastermlx.vision",
    "mastermlx.clustering",
]

SPECIAL_EXPORTS = {
    "__version__": "mastermlx.version",
    "get_backend": "mastermlx.config",
    "set_backend": "mastermlx.config",
    "create_rng": "mastermlx.utils",
    "set_seed": "mastermlx.utils",
    "log_sum_exp": "mastermlx.utils",
}

PUBLIC_SUBMODULES = [
    "accel",
    "anomaly",
    "bandits",
    "base",
    "clustering",
    "control",
    "data",
    "decomposition",
    "ensemble",
    "estimation",
    "graphs",
    "linear_models",
    "manifold",
    "math_tools",
    "neighbors",
    "neural_net",
    "nlp",
    "optimize",
    "planning",
    "preprocessing",
    "probabilistic",
    "rl",
    "robotics",
    "selection",
    "semi_supervised",
    "signal",
    "sim",
    "svm",
    "tabular",
    "trees",
    "utils",
    "variational",
    "vision",
    "viz",
]

REGISTRY_PATH = ROOT / "mastermlx" / "_lazy_exports.py"


def _package_exports():
    exports = {}
    for module_name in PACKAGE_MODULES:
        module = importlib.import_module(module_name)
        exports[module_name] = list(module.__all__)
    return dict(sorted(exports.items()))


def render_registry():
    exports = _package_exports()
    sections = [
        '''"""Static registry for the lazy top-level mastermlx facade.

This module intentionally contains names only. Importing it must not import any
NumPy-backed domain package. Package export changes are checked against this
registry by tests/test_api_compat.py.
"""''',
        f"_PACKAGE_EXPORTS = {json.dumps(exports, indent=4)}",
        f"_BINDING_ORDER = {json.dumps(PACKAGE_MODULES, indent=4)}",
        f"_PUBLIC_EXPORT_ORDER = {json.dumps(PUBLIC_EXPORT_ORDER, indent=4)}",
        f"_SPECIAL_EXPORTS = {json.dumps(SPECIAL_EXPORTS, indent=4)}",
        f"_PUBLIC_SUBMODULES = {json.dumps(PUBLIC_SUBMODULES, indent=4)}",
        '''_AMBIGUOUS_EXPORTS = {
    "LDA": ("mastermlx.nlp.NLP_LDA", "mastermlx.probabilistic.DiscriminantLDA"),
}''',
        '''def _build_export_map():
    exports = {}
    for module_name in _BINDING_ORDER:
        for name in _PACKAGE_EXPORTS[module_name]:
            exports[name] = module_name
    exports.update(_SPECIAL_EXPORTS)
    for name in _AMBIGUOUS_EXPORTS:
        exports.pop(name, None)
    return exports''',
        '''def _build_public_names():
    names = list(_SPECIAL_EXPORTS)
    for module_name in _PUBLIC_EXPORT_ORDER:
        for name in _PACKAGE_EXPORTS[module_name]:
            if module_name == "mastermlx.variational" and name in {"BayesGMM", "VGMM"}:
                continue
            if name not in names:
                names.append(name)
    for name in _AMBIGUOUS_EXPORTS:
        if name in names:
            names.remove(name)
    return names''',
        "EXPORTS = _build_export_map()\nPUBLIC_NAMES = _build_public_names()",
    ]
    return "\n\n".join(sections) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail if the registry is stale")
    mode.add_argument("--stdout", action="store_true", help="print the generated registry")
    args = parser.parse_args()

    generated = render_registry()
    if args.stdout:
        sys.stdout.write(generated)
        return 0
    if args.check:
        current = REGISTRY_PATH.read_text() if REGISTRY_PATH.exists() else ""
        if current != generated:
            print(
                "lazy export registry is stale; run: "
                "python scripts/generate_lazy_exports.py",
                file=sys.stderr,
            )
            return 1
        print("lazy export registry is up to date")
        return 0

    REGISTRY_PATH.write_text(generated)
    print(f"wrote {REGISTRY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
