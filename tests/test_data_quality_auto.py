import numpy as np
import pytest

from mastermlx.data import DataContract, compare_schema, drift_report, quality_report
from mastermlx.data.drift import drift_report as direct_drift_report
from mastermlx.data.schema import compare_schema as direct_compare_schema
from mastermlx.preprocessing import AutoPreprocessor, OneHotEncoder, SimpleImputer
from mastermlx import LogisticRegression
from mastermlx.tabular import TabularExperiment


class Frame:
    columns = ["age", "city"]

    def __init__(self, values):
        self.values = np.asarray(values, dtype=object)

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)


def test_quality_report_covers_missing_duplicates_and_outliers():
    X = np.array([[1.0, "a"], [1.0, "a"], [1.0, "b"], [100.0, "b"], [np.nan, None]], dtype=object)
    report = quality_report(X, low_freq=0.3)

    assert report["n_rows"] == 5
    assert report["duplicate_rows"] == 1
    assert report["numeric_columns"] == ["x0"]
    assert report["categorical_columns"] == ["x1"]
    assert report["columns"][0]["outlier_count"] == 1
    assert report["columns"][1]["missing_count"] == 1
    assert report["columns"][1]["low_freq_values"] == []


def test_schema_and_drift_report_named_columns():
    train = Frame([[1.0, "a"], [2.0, "b"], [3.0, "a"]])
    test = Frame([[5.0, "new"], [6.0, "new"]])
    schema = compare_schema(train, test)
    drift = drift_report(train, test)

    assert compare_schema is direct_compare_schema
    assert drift_report is direct_drift_report
    assert schema["order_match"]
    assert schema["missing_columns"] == []
    assert drift["columns"][1]["unseen_rate"] == 1.0
    assert drift["columns"][1]["tvd"] > 0.0


def test_data_contract_checks_schema_missing_ranges_and_categories():
    train = Frame([[20, "bj"], [30, "sh"], [40, "bj"]])
    contract = DataContract(
        rules={
            "age": {"kind": "numeric", "min": 0, "max": 120, "max_missing_rate": 0.1},
            "city": {"kind": "categorical", "allowed_values": ["bj", "sh"]},
        }
    ).fit(train)

    valid = contract.validate(Frame([[25, "bj"], [35, "sh"]]))
    invalid = contract.validate(Frame([[140, "new"], [None, "bj"]]))

    assert valid["valid"]
    assert not invalid["valid"]
    assert {item["code"] for item in invalid["errors"]} == {"range_max", "allowed_values", "missing_rate"}
    assert contract.summary()["required_columns"] == ["age", "city"]
    with pytest.raises(ValueError, match="data contract violation"):
        contract.check(Frame([[140, "bj"], [30, "bj"]]), raise_on_error=True)


def test_auto_preprocessor_detects_types_names_outputs_and_unknowns():
    train = Frame([[20, "bj"], [30, "sh"], [40, "bj"], [None, "gz"]])
    test = Frame([[25, "new"], [None, "bj"]])
    pre = AutoPreprocessor().fit(train)

    Xt = pre.transform(test)
    names = pre.get_feature_names_out().tolist()

    assert pre.numeric_cols_.tolist() == ["age"]
    assert pre.categorical_cols_.tolist() == ["city"]
    assert Xt.shape == (2, 4)
    assert names == ["num__age", "cat__city_bj", "cat__city_gz", "cat__city_sh"]
    assert np.isfinite(Xt).all()

    wrong = Frame([["bj", 25]])
    wrong.columns = ["city", "age"]
    with pytest.raises(ValueError, match="columns"):
        pre.transform(wrong)


def test_categorical_imputer_and_unknown_encoder():
    imp = SimpleImputer(strategy="most_frequent").fit(np.array([["a"], [None]], dtype=object))
    assert imp.transform(np.array([[None]], dtype=object)).tolist() == [["a"]]

    enc = OneHotEncoder(handle_unknown="ignore").fit(np.array([["a"], ["b"]], dtype=object))
    assert enc.transform(np.array([["new"]], dtype=object)).tolist() == [[0.0, 0.0]]


def test_tabular_experiment_accepts_auto_preprocessing():
    X = np.array([[0.0, "a"], [0.2, "a"], [0.8, "b"], [1.0, "b"]], dtype=object)
    y = np.array([0, 0, 1, 1])
    experiment = TabularExperiment(
        LogisticRegression(n_iter=300, random_state=0),
        preprocessing="auto",
        search=None,
    ).fit(X, y)

    assert experiment.predict(X).shape == y.shape


def test_tabular_experiment_enforces_contract_and_reports_it():
    X = np.array([[0.0], [0.2], [0.8], [1.0]])
    y = np.array([0, 0, 1, 1])
    contract = DataContract(rules={"x0": {"kind": "numeric", "min": 0.0, "max": 1.0}})
    experiment = TabularExperiment(
        LogisticRegression(n_iter=300, random_state=0),
        search=None,
        data_contract=contract,
    ).fit(X, y)

    report = experiment.report()
    assert report["contract"]["valid"]
    assert report["summary"]["has_data_contract"]
    with pytest.raises(ValueError, match="data contract violation"):
        experiment.predict(np.array([[1.5]]))
