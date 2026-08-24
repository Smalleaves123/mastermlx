from __future__ import annotations

import json
import subprocess
import sys


def _run_probe(source):
    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_top_level_import_defers_domain_packages():
    result = _run_probe(
        """
import json
import sys
import mastermlx

print(json.dumps({
    "modules": sorted(
        name for name in sys.modules
        if name == "mastermlx" or name.startswith("mastermlx.")
    ),
    "numpy_loaded": "numpy" in sys.modules,
}))
"""
    )

    assert result == {
        "modules": [
            "mastermlx",
            "mastermlx._lazy_exports",
            "mastermlx.config",
            "mastermlx.version",
        ],
        "numpy_loaded": False,
    }


def test_top_level_symbol_loads_its_owner_on_first_access():
    result = _run_probe(
        """
import json
import sys
import mastermlx

before = "mastermlx.clustering" in sys.modules
module = mastermlx.KMeans.__module__
after = "mastermlx.clustering" in sys.modules
unrelated = "mastermlx.signal" in sys.modules
print(json.dumps({
    "before": before,
    "module": module,
    "after": after,
    "unrelated": unrelated,
}))
"""
    )

    assert result == {
        "before": False,
        "module": "mastermlx.clustering.kmeans",
        "after": True,
        "unrelated": False,
    }


def test_public_submodule_attribute_is_lazy():
    result = _run_probe(
        """
import json
import sys
import mastermlx

before = "mastermlx.nlp" in sys.modules
alias = mastermlx.nlp.NLP_LDA.__name__
after = "mastermlx.nlp" in sys.modules
print(json.dumps({"before": before, "alias": alias, "after": after}))
"""
    )

    assert result == {"before": False, "alias": "LDA", "after": True}
