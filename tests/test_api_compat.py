import inspect
import importlib

import numpy as np
import pytest

import mastermlx
from mastermlx.base import BaseEstimator, BaseExperiment, BaseReport, BaseResult, Module
from mastermlx.clustering import KMeans
from mastermlx.neural_net import MLPClassifier, MLPRegressor, Sequential
from mastermlx.nlp import LDA as NLP_LDA
from mastermlx.preprocessing import Pipeline, PolynomialFeatures, StandardScaler
from mastermlx.probabilistic import DiscriminantLDA


PUBLIC_PACKAGES = (
    "base", "anomaly", "bandits", "data", "decomposition", "ensemble",
    "control", "estimation", "linear_models", "math_tools", "graphs",
    "optimize", "planning", "manifold", "neighbors", "neural_net", "nlp",
    "probabilistic", "rl", "preprocessing", "selection", "semi_supervised",
    "signal", "tabular", "robotics", "svm", "trees", "variational", "viz",
    "vision", "clustering",
)


def test_core_public_api_remains_available():
    for name in ("fit", "score", "save", "load", "get_params", "set_params"):
        assert hasattr(BaseEstimator, name)
    for name in ("state_dict", "load_state_dict", "save", "load", "save_checkpoint", "load_checkpoint"):
        assert hasattr(Module, name)
    for cls in (Sequential, MLPClassifier, MLPRegressor):
        assert hasattr(cls, "fit")
        assert hasattr(cls, "evaluate")
        assert "X" in inspect.signature(cls.fit).parameters
    assert hasattr(mastermlx, "get_backend")
    assert hasattr(mastermlx, "set_backend")
    assert hasattr(BaseResult({"ok": True}), "as_dict")
    assert issubclass(BaseReport, BaseResult)
    assert hasattr(BaseExperiment(), "export_report")
    assert mastermlx.BaseResult is BaseResult


def test_checkpoint_signatures_are_stable():
    assert "path" in inspect.signature(Module.save_checkpoint).parameters
    assert "path" in inspect.signature(Module.load_checkpoint).parameters
    assert "strict" in inspect.signature(Module.load).parameters


def test_ambiguous_lda_models_have_explicit_namespaced_aliases():
    assert NLP_LDA is not DiscriminantLDA
    assert mastermlx.nlp.NLP_LDA is NLP_LDA
    assert mastermlx.probabilistic.DiscriminantLDA is DiscriminantLDA
    assert "LDA" not in mastermlx.__all__
    with pytest.raises(AttributeError, match="ambiguous"):
        getattr(mastermlx, "LDA")


def test_top_level_exports_do_not_silently_overwrite_subpackage_objects():
    owners = {}
    for package_name in PUBLIC_PACKAGES:
        package = importlib.import_module(f"mastermlx.{package_name}")
        for name in package.__all__:
            if package_name == "variational" and name in {"BayesGMM", "VGMM"}:
                continue
            owners.setdefault(name, []).append((package_name, getattr(package, name)))

    conflicts = {
        name
        for name, values in owners.items()
        if len({id(value) for _, value in values}) > 1
    }
    assert conflicts == {"LDA"}
    for name, values in owners.items():
        if name == "LDA":
            continue
        assert name in mastermlx.__all__
        assert getattr(mastermlx, name) is values[-1][1]


def test_public_base_interfaces_expose_type_annotations():
    for method in (BaseEstimator.fit, BaseEstimator.get_params, BaseEstimator.set_params):
        assert method.__annotations__
    for method in (StandardScaler.fit, StandardScaler.transform, KMeans.fit):
        assert method.__annotations__


def test_set_params_rejects_unknown_estimator_parameter():
    model = KMeans(n_clusters=2)

    with pytest.raises(ValueError, match="does_not_exist"):
        model.set_params(does_not_exist=1)

    model.set_params(n_clusters=3)
    assert model.n_clusters == 3


def test_pipeline_rejects_unknown_parameter_and_nested_step_parameter():
    pipe = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
        ]
    )

    with pytest.raises(ValueError, match="unknown_step"):
        pipe.set_params(unknown_step__degree=3)
    with pytest.raises(ValueError, match="does_not_exist"):
        pipe.set_params(does_not_exist=1)
    with pytest.raises(ValueError, match="does_not_exist"):
        pipe.set_params(poly__does_not_exist=3)

    pipe.set_params(poly__degree=3)
    assert pipe.steps[0][1].degree == 3


def test_representative_estimator_transformer_contracts():
    X = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])

    model = KMeans(n_clusters=2, random_state=0, n_init=1)
    assert model.fit(X) is model
    assert model.predict(X).shape == (3,)
    assert model.get_params()["n_clusters"] == 2

    scaler = StandardScaler()
    assert scaler.fit(X) is scaler
    transformed = scaler.fit_transform(X)
    assert transformed.shape == X.shape
    assert np.allclose(scaler.inverse_transform(transformed), X)
