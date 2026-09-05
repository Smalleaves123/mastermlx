import json
import subprocess
import sys
import zipfile

import numpy as np
import pytest

import mastermlx
from mastermlx.api import STABILITY_LEVELS, api_stability_report, get_api_stability
from mastermlx.data import DataContract
from mastermlx.linear_models import LogisticRegression


class Frame:
    def __init__(self, values, columns=("age", "city")):
        self.values = np.asarray(values, dtype=object)
        self.columns = list(columns)

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)


def test_every_top_level_api_has_a_stability_tier():
    report = api_stability_report()
    classified = {name for names in report.values() for name in names}

    assert tuple(report) == STABILITY_LEVELS
    assert classified == set(mastermlx.__all__)
    assert get_api_stability("KMeans") == "stable"
    assert get_api_stability("mastermlx.ModelBundle") == "beta"
    assert get_api_stability("robotics.RobotModel") == "experimental"
    assert get_api_stability("__version__") == "stable"
    assert get_api_stability("robotics._private") == "internal"
    with pytest.raises(KeyError, match="unknown public API"):
        get_api_stability("DoesNotExist")


def test_api_stability_queries_keep_domain_packages_lazy():
    code = """
import json, sys
import mastermlx.api as api
api.get_api_stability('KMeans')
print(json.dumps({
    'mastermlx': sorted(name for name in sys.modules if name.startswith('mastermlx.')),
    'numpy_loaded': 'numpy' in sys.modules,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    loaded = json.loads(result.stdout)

    assert "mastermlx.clustering" not in loaded["mastermlx"]
    assert "mastermlx.api" in loaded["mastermlx"]
    assert not loaded["numpy_loaded"]


def test_model_bundle_validates_preprocesses_and_round_trips(tmp_path):
    X = Frame([[20, "bj"], [30, "sh"], [40, "bj"], [50, "sh"]])
    y = np.array([0, 0, 1, 1])
    contract = DataContract(
        rules={
            "age": {"kind": "numeric", "min": 0, "max": 120},
            "city": {"kind": "categorical", "allowed_values": ["bj", "sh"]},
        }
    )
    bundle = mastermlx.ModelBundle(
        LogisticRegression(n_iter=400, random_state=0),
        preprocessing="auto",
        data_contract=contract,
        metadata={"owner": "quality-team", "task": "classification"},
    ).fit(X, y)

    expected = bundle.predict(X)
    assert expected.shape == y.shape
    assert bundle.predict_proba(X).shape == (4, 2)
    assert bundle.validate(X)["valid"]
    assert bundle.summary()["has_data_contract"]
    assert bundle.summary()["feature_names_in"] == ["age", "city"]

    with pytest.raises(ValueError, match="model bundle input violation"):
        bundle.predict(Frame([[140, "bj"]]))
    with pytest.raises(ValueError, match="model bundle input violation"):
        bundle.predict(Frame([[30, "new"]]))

    path = tmp_path / "classifier.mlx"
    bundle.save(path)
    restored = mastermlx.ModelBundle.load(path)

    assert np.array_equal(restored.predict(X), expected)
    assert restored.summary()["metadata"]["owner"] == "quality-team"
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["format"] == "mastermlx-checkpoint"
        assert not any(name.endswith((".pkl", ".pickle")) for name in archive.namelist())


def test_data_contract_keeps_schema_without_training_rows():
    X = Frame([[20, "bj"], [30, "sh"]])
    contract = DataContract().fit(X)

    assert contract.reference_ is None
    assert contract.reference_schema_["shape"] == (2, 2)
    assert contract.reference_schema_["columns"] == ["age", "city"]
    assert contract.validate(Frame([[40, "bj"]]))["valid"]
