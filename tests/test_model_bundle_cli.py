import json
import subprocess
import sys

import numpy as np

from mastermlx import DataContract, LinearRegression, ModelBundle, StandardScaler
from mastermlx.tabular import TabularExperiment


def _saved_bundle(tmp_path):
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = 2.0 * X.ravel() + 1.0
    bundle = ModelBundle(
        LinearRegression(),
        data_contract=DataContract(
            rules={"x0": {"kind": "numeric", "min": 0.0, "max": 10.0}}
        ),
        metadata={
            "task": "regression",
            "owner": "forecast-team",
            "intended_use": "local demand estimates",
            "limitations": ["validated only on non-negative inputs"],
            "metrics": {"r2": 1.0},
        },
    ).fit(X, y)
    path = tmp_path / "linear.mlx"
    bundle.save(path)
    return bundle, path


def test_model_bundle_model_card_and_export(tmp_path):
    bundle, _ = _saved_bundle(tmp_path)
    card = bundle.model_card()
    output = bundle.export_model_card(tmp_path / "model-card.json")

    assert card["format"] == "mastermlx-model-card"
    assert card["model"]["task"] == "regression"
    assert card["governance"]["missing_fields"] == []
    assert card["input"]["data_contract"]["feature_names"] == ["x0"]
    assert json.loads(output.read_text())["governance"]["owner"] == "forecast-team"


def test_bundle_cli_inspects_and_predicts_json_and_csv(tmp_path):
    _, bundle_path = _saved_bundle(tmp_path)
    csv_input = tmp_path / "input.csv"
    csv_input.write_text("x0\n4\n5\n")
    json_input = tmp_path / "input.json"
    json_input.write_text(json.dumps({"records": [{"x0": 6.0}]}))
    csv_output = tmp_path / "predictions.csv"

    inspect_result = subprocess.run(
        [sys.executable, "-m", "mastermlx.cli", "inspect", str(bundle_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    csv_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mastermlx.cli",
            "predict",
            str(bundle_path),
            str(csv_input),
            "--output",
            str(csv_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    json_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mastermlx.cli",
            "predict",
            str(bundle_path),
            str(json_input),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(inspect_result.stdout)["format"] == "mastermlx-model-card"
    assert csv_result.stdout == ""
    assert csv_output.read_text().splitlines() == ["prediction", "9.0", "11.0"]
    assert np.allclose(json.loads(json_result.stdout)["predictions"], [13.0])


def test_bundle_cli_validate_returns_nonzero_for_contract_violation(tmp_path):
    _, bundle_path = _saved_bundle(tmp_path)
    bad_input = tmp_path / "bad.json"
    bad_input.write_text(json.dumps({"rows": [[20.0]], "columns": ["x0"]}))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mastermlx.cli",
            "validate",
            str(bundle_path),
            str(bad_input),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert not json.loads(result.stdout)["valid"]


def test_tabular_experiment_exports_fitted_pipeline_without_retraining(tmp_path):
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = 3.0 * X.ravel() - 2.0
    experiment = TabularExperiment(
        LinearRegression(),
        preprocessing=StandardScaler(),
        search=None,
        task="regression",
        data_contract=DataContract(
            rules={"x0": {"kind": "numeric", "min": 0.0, "max": 20.0}}
        ),
    ).fit(X, y)

    bundle = experiment.to_bundle(metadata={"owner": "deployment-team"})
    expected = experiment.predict([[4.0], [7.0]])
    path = tmp_path / "selected-model.mlx"
    bundle.save(path)
    restored = ModelBundle.load(path)

    assert np.allclose(bundle.predict([[4.0], [7.0]]), expected)
    assert np.allclose(restored.predict([[4.0], [7.0]]), expected)
    assert bundle.model_card()["model"]["task"] == "regression"
    assert bundle.model_card()["governance"]["owner"] == "deployment-team"
