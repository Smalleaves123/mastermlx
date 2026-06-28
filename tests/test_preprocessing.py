import numpy as np

from mastermlx.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    RobustScaler,
    SimpleImputer,
    StandardScaler,
)


def test_standard_scaler_works_on_constant_column():
    X = np.array([[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]])
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)

    assert np.allclose(Z.mean(axis=0), np.array([0.0, 0.0]), atol=1e-8)
    assert np.allclose(Z[:, 1], 0.0, atol=1e-8)


def test_standard_scaler_inverse_transform_round_trip():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)

    assert np.allclose(sc.inverse_transform(Z), X)


def test_minmax_scaler_maps_range():
    X = np.array([[0.0, 2.0], [5.0, 2.0], [10.0, 2.0]])
    sc = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(X)
    Z = sc.transform(X)

    assert np.allclose(Z[:, 0], np.array([-1.0, 0.0, 1.0]))
    assert np.allclose(sc.inverse_transform(Z), X)


def test_robust_scaler_uses_median_and_iqr():
    X = np.array([[1.0], [2.0], [3.0], [100.0]])
    sc = RobustScaler().fit(X)
    Z = sc.transform(X)

    assert np.isclose(np.median(Z[:, 0]), 0.0)


def test_normalizer_scales_rows_to_unit_norm():
    X = np.array([[3.0, 4.0], [0.0, 0.0]])
    Z = Normalizer().fit(X).transform(X)

    assert np.allclose(Z[0], np.array([0.6, 0.8]))
    assert np.allclose(Z[1], np.array([0.0, 0.0]))


def test_label_encoder_round_trip():
    y = np.array(["cat", "dog", "cat", "bird"], dtype=object)
    enc = LabelEncoder().fit(y)
    z = enc.transform(y)

    assert np.array_equal(enc.inverse_transform(z), y)


def test_one_hot_encoder_expands_categories():
    X = np.array([["red", "S"], ["blue", "M"], ["red", "M"]], dtype=object)
    enc = OneHotEncoder().fit(X)
    Z = enc.transform(X)

    assert Z.shape == (3, 4)
    assert np.allclose(Z[0], np.array([0.0, 1.0, 0.0, 1.0]))


def test_ordinal_encoder_round_trip():
    X = np.array([["red", "S"], ["blue", "M"], ["red", "M"]], dtype=object)
    enc = OrdinalEncoder().fit(X)
    Z = enc.transform(X)

    assert Z.shape == (3, 2)
    assert np.array_equal(enc.inverse_transform(Z), X)


def test_simple_imputer_fills_missing_values():
    X = np.array([[1.0, np.nan], [3.0, 5.0], [np.nan, 7.0]])
    imp = SimpleImputer(strategy="mean").fit(X)
    Z = imp.transform(X)

    assert np.allclose(Z, np.array([[1.0, 6.0], [3.0, 5.0], [2.0, 7.0]]))


def test_simple_imputer_constant_strategy():
    X = np.array([[np.nan, 1.0], [2.0, np.nan]])
    imp = SimpleImputer(strategy="constant", fill_value=-1.0).fit(X)
    Z = imp.transform(X)

    assert np.allclose(Z, np.array([[-1.0, 1.0], [2.0, -1.0]]))


def test_polynomial_features_builds_quadratic_terms():
    X = np.array([[2.0, 3.0], [1.0, 4.0]])
    poly = PolynomialFeatures(degree=2).fit(X)
    Z = poly.transform(X)

    assert Z.shape == (2, 6)
    assert np.allclose(Z[0], np.array([1.0, 2.0, 3.0, 4.0, 6.0, 9.0]))


def test_polynomial_features_interaction_only():
    X = np.array([[2.0, 3.0, 4.0]])
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True).fit(X)
    Z = poly.transform(X)

    assert np.allclose(Z, np.array([[2.0, 3.0, 4.0, 6.0, 8.0, 12.0]]))


def test_kbins_discretizer_returns_ordinal_bins():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    disc = KBinsDiscretizer(n_bins=2, strategy="uniform", encode="ordinal").fit(X)
    Z = disc.transform(X)

    assert Z.shape == (5, 1)
    assert np.array_equal(np.unique(Z.ravel()), np.array([0.0, 1.0]))


def test_kbins_discretizer_supports_onehot_encoding():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    disc = KBinsDiscretizer(n_bins=2, strategy="quantile", encode="onehot").fit(X)
    Z = disc.transform(X)

    assert Z.shape == (4, 2)
    assert np.allclose(Z.sum(axis=1), 1.0)
