import numpy as np

from mastermlx.linear_models import LinearRegression
from mastermlx.selection import RFE, SelectKBest, VarianceThreshold, f_classif
from mastermlx.semi_supervised import LabelPropagation, LabelSpreading


def test_variance_threshold_drops_constant_columns():
    X = np.array(
        [
            [1.0, 0.0, 3.0],
            [1.0, 1.0, 3.0],
            [1.0, 2.0, 3.0],
        ]
    )

    sel = VarianceThreshold(threshold=0.0).fit(X)
    out = sel.transform(X)

    assert out.shape == (3, 1)
    assert np.array_equal(sel.get_support(), np.array([False, True, False]))


def test_select_k_best_keeps_discriminative_feature():
    X = np.array(
        [
            [0.0, 5.0],
            [0.2, 5.0],
            [0.1, 4.9],
            [2.8, 5.0],
            [3.0, 5.1],
            [2.9, 4.8],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    sel = SelectKBest(score_func=f_classif, k=1).fit(X, y)
    out = sel.transform(X)

    assert out.shape == (6, 1)
    assert np.array_equal(sel.get_support(), np.array([True, False]))
    assert sel.scores_[0] > sel.scores_[1]
    assert np.isfinite(sel.scores_).all()


def test_rfe_selects_linear_signal_feature():
    x1 = np.array([0.0, 0.2, 0.4, 2.8, 3.0, 3.2])
    x2 = np.array([3.0, 2.8, 3.1, 0.2, 0.0, 0.1])
    X = np.column_stack([x1, x2])
    y = 2.0 * x1 + 0.1

    rfe = RFE(LinearRegression(), n_features_to_select=1, step=1).fit(X, y)
    out = rfe.transform(X)

    assert out.shape == (6, 1)
    assert np.array_equal(rfe.get_support(), np.array([True, False]))
    assert rfe.ranking_[0] == 1
    assert rfe.ranking_[1] > 1
    assert rfe.estimator_.coef_.shape == (1,)


def test_label_propagation_spreads_labels():
    X = np.array([[0.0], [0.2], [0.4], [4.6], [4.8], [5.0]])
    y = np.array([0, -1, -1, 1, -1, -1])

    lp = LabelPropagation(kernel="rbf", gamma=2.0, max_iter=200).fit(X, y)
    ls = LabelSpreading(kernel="rbf", gamma=2.0, alpha=0.2, max_iter=200).fit(X, y)

    assert np.array_equal(lp.predict(), np.array([0, 0, 0, 1, 1, 1]))
    assert np.array_equal(ls.predict(), np.array([0, 0, 0, 1, 1, 1]))
    assert lp.predict_proba().shape == (6, 2)
    assert ls.predict_proba().shape == (6, 2)
    assert np.allclose(lp.predict_proba().sum(axis=1), 1.0)
    assert np.allclose(ls.predict_proba().sum(axis=1), 1.0)

