import numpy as np
import pytest
from scipy import sparse

from mastermlx import LinearRegression, LogisticRegression, RidgeRegression, SGDRegressor
from mastermlx.ensemble import MultiOutputClassifier, MultiOutputRegressor
from mastermlx.preprocessing import Pipeline, SimpleImputer
from mastermlx.utils import check_X, check_sample_weight


def test_sample_weight_matches_integer_replication_for_linear_regression():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.column_stack([2.0 * X[:, 0] + 1.0, -X[:, 0] + 4.0])
    weights = np.array([1, 3, 2, 1])
    weighted = LinearRegression().fit(sparse.csr_matrix(X), y, sample_weight=weights)

    repeated = np.repeat(np.arange(X.shape[0]), weights)
    reference = LinearRegression().fit(X[repeated], y[repeated])

    assert weighted.coef_.shape == (2, 1)
    assert weighted.predict(sparse.csr_matrix(X)).shape == y.shape
    assert np.allclose(weighted.coef_, reference.coef_)
    assert np.allclose(weighted.intercept_, reference.intercept_)


def test_ridge_matches_weighted_multioutput_contract():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.column_stack([X[:, 0], 2.0 * X[:, 0]])
    model = RidgeRegression(alpha=0.1).fit(
        sparse.csr_matrix(X), y, sample_weight=[1.0, 2.0, 1.0, 2.0]
    )

    assert model.coef_.shape == (2, 1)
    assert model.predict(sparse.csr_matrix(X)).shape == y.shape
    assert model.score(X, y) > 0.9


def test_logistic_supports_multiclass_string_labels_and_weighted_sparse_input():
    X = np.array(
        [[-2.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [0.0, 1.0], [2.0, 0.0], [1.0, 0.0]]
    )
    y = np.array(["left", "left", "middle", "middle", "right", "right"])
    model = LogisticRegression(lr=0.2, n_iter=500, random_state=0).fit(
        sparse.csr_matrix(X), y, sample_weight=[1, 2, 1, 2, 1, 2]
    )

    assert model.predict(sparse.csr_matrix(X)).shape == (X.shape[0],)
    assert model.predict_proba(X).shape == (X.shape[0], 3)
    assert model.score(X, y, sample_weight=np.ones(X.shape[0])) > 0.8


def test_multioutput_wrappers_forward_weights_and_preserve_shapes():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y_reg = np.column_stack([X[:, 0] + X[:, 1], 2.0 * X[:, 0] - X[:, 1]])
    reg = MultiOutputRegressor(LinearRegression()).fit(
        sparse.csr_matrix(X), y_reg, sample_weight=np.arange(1, 7)
    )
    assert reg.predict(sparse.csr_matrix(X)).shape == y_reg.shape
    assert reg.score(X, y_reg, sample_weight=np.arange(1, 7)) > 0.99

    y_cls = np.column_stack([(X[:, 0] > 2).astype(int), (X[:, 1] > 3).astype(int)])
    clf = MultiOutputClassifier(LogisticRegression(n_iter=300, random_state=0)).fit(
        sparse.csr_matrix(X), y_cls, sample_weight=np.arange(1, 7)
    )
    assert clf.predict(sparse.csr_matrix(X)).shape == y_cls.shape


def test_missing_values_require_explicit_imputation_for_linear_models():
    X = np.array([[0.0, 1.0], [1.0, np.nan], [2.0, 3.0]])
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="finite"):
        LinearRegression().fit(X, y)

    pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("model", LinearRegression())]
    ).fit(X, y)
    assert np.isfinite(pipeline.predict(X)).all()


def test_sgd_partial_fit_weights_and_warm_start():
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0
    incremental = SGDRegressor(max_iter=1, eta0=0.001, shuffle=False, random_state=0)
    incremental.partial_fit(sparse.csr_matrix(X[:6]), y[:6], sample_weight=np.ones(6))
    incremental.partial_fit(X[6:], y[6:], sample_weight=np.ones(6))
    assert np.isfinite(incremental.predict(X)).all()

    cold = SGDRegressor(max_iter=1, eta0=0.001, shuffle=False, random_state=0)
    cold.fit(X, y)
    cold_coef = cold.coef_.copy()
    cold.fit(X, y)
    assert np.allclose(cold.coef_, cold_coef)

    warm = SGDRegressor(max_iter=1, eta0=0.001, shuffle=False, random_state=0, warm_start=True)
    warm.fit(X, y)
    warm_coef = warm.coef_.copy()
    warm.fit(X, y)
    assert not np.allclose(warm.coef_, warm_coef)


def test_validation_rejects_invalid_sample_weights_and_keeps_sparse_inputs():
    X = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    assert sparse.issparse(check_X(X))
    with pytest.raises(ValueError, match="one value per sample"):
        check_sample_weight([1.0], 2)
    with pytest.raises(ValueError, match="non-negative"):
        check_sample_weight([1.0, -1.0], 2)
    with pytest.raises(ValueError, match="positive"):
        check_sample_weight([0.0, 0.0], 2)
