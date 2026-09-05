import inspect

import numpy as np
import pytest

from mastermlx.clustering import KMeans, MiniBatchKMeans
from mastermlx.linear_models import (
    LinearRegression,
    LogisticRegression,
    SGDClassifier,
    SGDRegressor,
)
from mastermlx.neighbors import KNNClassifier, KNNRegressor
from mastermlx.preprocessing import (
    Binarizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)
from mastermlx.trees import DecisionTreeClassifier, DecisionTreeRegressor
from mastermlx.utils import clone
from mastermlx.utils.validation import check_X, to_dense


X = np.array(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [2.0, 0.0],
        [2.0, 1.0],
    ]
)
Y_CLASS = np.array([0, 0, 0, 1, 1, 1])
Y_REG = X @ np.array([2.0, -0.5]) + 1.0


@pytest.mark.parametrize(
    ("factory", "target"),
    [
        (lambda: LinearRegression(), Y_REG),
        (lambda: LogisticRegression(n_iter=100, random_state=0), Y_CLASS),
        (lambda: SGDClassifier(max_iter=20, random_state=0), Y_CLASS),
        (lambda: SGDRegressor(max_iter=20, random_state=0), Y_REG),
        (lambda: KNNClassifier(k=1), Y_CLASS),
        (lambda: KNNRegressor(k=2), Y_REG),
        (lambda: DecisionTreeClassifier(max_depth=3), Y_CLASS),
        (lambda: DecisionTreeRegressor(max_depth=3), Y_REG),
    ],
    ids=[
        "linear-regression",
        "logistic-regression",
        "sgd-classifier",
        "sgd-regressor",
        "knn-classifier",
        "knn-regressor",
        "decision-tree-classifier",
        "decision-tree-regressor",
    ],
)
def test_stable_estimators_share_fitted_shape_and_clone_contracts(factory, target):
    estimator = factory()
    cloned = clone(estimator)

    assert type(cloned) is type(estimator)
    assert cloned.get_params() == estimator.get_params()
    assert estimator.fit(X, target) is estimator
    assert estimator.n_features_in_ == X.shape[1]
    assert estimator.predict(X[:1]).shape == (1,)
    assert np.isfinite(estimator.score(X, target))
    with pytest.raises(ValueError, match="different number of features"):
        estimator.predict(np.ones((1, X.shape[1] + 1)))


@pytest.mark.parametrize(
    "model",
    [
        KMeans(n_clusters=2, n_init=1, random_state=0),
        MiniBatchKMeans(n_clusters=2, n_init=1, max_iter=5, random_state=0),
    ],
    ids=lambda model: type(model).__name__,
)
def test_stable_clusterer_shares_fitted_feature_contract(model):
    assert model.fit(X) is model
    assert model.n_features_in_ == X.shape[1]
    assert model.predict(X[:1]).shape == (1,)
    assert np.isfinite(model.inertia_)
    with pytest.raises(ValueError, match="different number of features"):
        model.predict(np.ones((1, X.shape[1] + 1)))


@pytest.mark.parametrize(
    "transformer",
    [
        StandardScaler(),
        MinMaxScaler(),
        MaxAbsScaler(),
        RobustScaler(),
        Normalizer(),
        Binarizer(),
    ],
    ids=lambda transformer: type(transformer).__name__,
)
def test_stable_transformers_share_feature_count_contract(transformer):
    assert transformer.fit(X) is transformer
    assert transformer.n_features_in_ == X.shape[1]
    assert transformer.transform(X[:1]).shape == (1, X.shape[1])
    with pytest.raises(ValueError, match="different number of features"):
        transformer.transform(np.ones((1, X.shape[1] + 1)))


def test_linear_regression_multioutput_contract_uses_vector_intercept():
    targets = np.column_stack([Y_REG, -2.0 * Y_REG + 3.0])
    model = LinearRegression().fit(X, targets)

    assert model.predict(X[:1]).shape == (1, 2)
    assert isinstance(model.intercept_, np.ndarray)
    assert model.intercept_.shape == (2,)
    assert np.isclose(model.score(X, targets), 1.0)


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_logistic_regression_warm_start_preserves_binary_contract(fit_intercept):
    model = LogisticRegression(
        n_iter=20,
        random_state=0,
        warm_start=True,
        fit_intercept=fit_intercept,
    ).fit(X, Y_CLASS)
    previous = np.asarray(model.coef_).copy()

    assert model.fit(X, Y_CLASS) is model
    assert model.coef_.shape == previous.shape
    assert np.isfinite(model.loss_).all()
    assert model.predict_proba(X[:1]).shape == (1, 2)


def test_sgd_incremental_signatures_and_multiclass_contracts():
    classifier_signature = inspect.signature(SGDClassifier.partial_fit)
    regressor_signature = inspect.signature(SGDRegressor.partial_fit)
    assert "classes" in classifier_signature.parameters
    assert "classes" not in regressor_signature.parameters

    classifier = SGDClassifier(random_state=0)
    classifier.partial_fit(X[:3], np.array([0, 1, 2]), classes=np.array([0, 1, 2]))
    classifier.partial_fit(X[3:], np.array([0, 1, 2]))
    assert classifier.predict(X).shape == (X.shape[0],)
    assert classifier.coef_.shape == (X.shape[1], 3)

    regressor = SGDRegressor(random_state=0)
    regressor.partial_fit(X[:3], Y_REG[:3])
    regressor.partial_fit(X[3:], Y_REG[3:])
    assert regressor.predict(X).shape == Y_REG.shape


def test_sparse_validation_preserves_sparse_inputs_and_dense_conversion():
    sparse = pytest.importorskip("scipy.sparse")
    matrix = sparse.csr_matrix(X)

    checked = check_X(matrix, ensure_all_finite=True)

    assert checked is matrix
    assert sparse.issparse(checked)
    assert np.array_equal(to_dense(checked), X)
