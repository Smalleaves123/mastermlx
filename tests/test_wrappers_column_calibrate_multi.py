import numpy as np
import pytest

from mastermlx.preprocessing import ColumnTransformer, StandardScaler, MinMaxScaler
from mastermlx.ensemble import CalibratedClassifierCV, MultiOutputClassifier, MultiOutputRegressor
from mastermlx import LogisticRegression, LinearRegression


# ---------------------------------------------------------------------------
# ColumnTransformer
# ---------------------------------------------------------------------------

def test_column_transformer_basic():
    X = np.random.randn(100, 5)
    ct = ColumnTransformer([
        ("scale", StandardScaler(), [0, 1]),
        ("norm", MinMaxScaler(), [2, 3]),
    ]).fit(X)
    Xt = ct.transform(X)
    assert Xt.shape == (100, 4)
    assert np.allclose(Xt[:, :2].mean(axis=0), 0, atol=0.1)


def test_column_transformer_passthrough():
    X = np.random.randn(50, 4)
    ct = ColumnTransformer([
        ("scale", StandardScaler(), [0]),
    ], remainder="passthrough").fit(X)
    Xt = ct.transform(X)
    assert Xt.shape == (50, 4)  # 1 scaled + 3 passthrough


def test_column_transformer_slice():
    X = np.random.randn(50, 6)
    ct = ColumnTransformer([
        ("first_half", StandardScaler(), slice(0, 3)),
    ], remainder="drop").fit(X)
    assert ct.transform(X).shape == (50, 3)


def test_column_transformer_clones_steps_and_names_outputs():
    X = np.random.randn(20, 3)
    scaler = StandardScaler()
    ct = ColumnTransformer([("scale", scaler, [0, 1])], remainder="passthrough").fit(X)

    assert scaler.mean_ is None
    assert ct.transformers_[0][1] is not scaler
    assert ct.get_feature_names_out().tolist() == ["scale__x0", "scale__x1", "x2"]


def test_column_transformer_accepts_named_columns():
    class Frame:
        columns = ["age", "score", "keep"]

        def __init__(self, data):
            self.data = np.asarray(data)

        def __array__(self, dtype=None):
            return np.asarray(self.data, dtype=dtype)

    X = Frame(np.random.randn(10, 3))
    ct = ColumnTransformer([("scale", StandardScaler(), ["age", "score"])], remainder="passthrough").fit(X)

    assert ct.get_feature_names_out().tolist() == ["scale__age", "scale__score", "keep"]


def test_column_transformer_empty_raises():
    with pytest.raises(ValueError, match="remainder"):
        ColumnTransformer([("x", StandardScaler(), [0])], remainder="bad").fit(np.random.randn(10, 3))
    with pytest.raises(ValueError, match="overlap"):
        ColumnTransformer([("a", StandardScaler(), [0, 1]),
                           ("b", MinMaxScaler(), [1, 2])]).fit(np.random.randn(10, 3))
    with pytest.raises(ValueError, match="out of range"):
        ColumnTransformer([("a", StandardScaler(), [99])]).fit(np.random.randn(10, 3))


# ---------------------------------------------------------------------------
# CalibratedClassifierCV
# ---------------------------------------------------------------------------

def test_calibrated_binary():
    X = np.random.randn(200, 5)
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, 0)
    cal = CalibratedClassifierCV(LogisticRegression(lr=0.1, n_iter=100)).fit(X, y)
    proba = cal.predict_proba(X)
    assert proba.shape == (200, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_calibrated_improves():
    X = np.random.randn(200, 5)
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, 0)
    cal = CalibratedClassifierCV(LogisticRegression(lr=0.1, n_iter=100)).fit(X, y)
    assert cal.score(X, y) > 0.80


def test_calibrated_raises_on_multiclass():
    X = np.random.randn(100, 3)
    y = np.array([0, 1, 2] * 34)[:100]
    with pytest.raises(ValueError, match="binary"):
        CalibratedClassifierCV(LogisticRegression()).fit(X, y)


def test_calibrated_raises_on_empty():
    with pytest.raises(ValueError):
        CalibratedClassifierCV(LogisticRegression()).fit(np.random.randn(3, 2), np.array([]))


# ---------------------------------------------------------------------------
# MultiOutput
# ---------------------------------------------------------------------------

def test_multi_output_classifier():
    X = np.random.randn(100, 5)
    y1 = np.where(X[:, 0] > 0, 1, 0)
    y2 = np.where(X[:, 1] > 0, 1, 0)
    y = np.column_stack([y1, y2])
    moc = MultiOutputClassifier(LogisticRegression(lr=0.1, n_iter=100)).fit(X, y)
    pred = moc.predict(X)
    assert pred.shape == (100, 2)
    assert np.mean(pred[:, 0] == y1) > 0.70
    assert np.mean(pred[:, 1] == y2) > 0.70


def test_multi_output_regressor():
    X = np.random.randn(100, 3)
    y1 = X[:, 0] * 2 + X[:, 1] - 1
    y2 = X[:, 1] * 3 + 0.5
    y = np.column_stack([y1, y2])
    mor = MultiOutputRegressor(LinearRegression()).fit(X, y)
    pred = mor.predict(X)
    assert pred.shape == (100, 2)
    assert mor.score(X, y) > 0.80


def test_multi_output_single_col():
    X = np.random.randn(50, 3)
    y = np.where(X[:, 0] > 0, 1, 0)
    moc = MultiOutputClassifier(LogisticRegression()).fit(X, y)
    assert moc.predict(X).shape == (50, 1)


def test_multi_output_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        MultiOutputClassifier(LogisticRegression()).fit(np.random.randn(10, 2), np.empty((0, 2)))
    with pytest.raises(ValueError, match="same number of rows"):
        MultiOutputRegressor(LinearRegression()).fit(np.random.randn(10, 2), np.random.randn(5, 2))
