import numpy as np

from mastermlx.math_tools import (
    canberra_distance, make_blobs, make_classification,
    make_moons, make_regression,
)
from mastermlx.utils import bray_curtis_distance, wasserstein_distance
from mastermlx.utils import avg_precision_score, fbeta_score, jaccard_score
from mastermlx.data import LeaveOneOut, RepeatedKFold
from mastermlx.selection import SelectFromModel
from mastermlx.neural_net import CosineLR, ReduceLROnPlateau, StepLR, SGD


# --- Datasets ---
def test_make_blobs():
    X, y = make_blobs(100, 3, centers=2, random_state=0)
    assert X.shape == (100, 3)
    assert len(np.unique(y)) == 2


def test_make_moons():
    X, y = make_moons(100, random_state=0)
    assert X.shape == (100, 2)
    assert len(np.unique(y)) == 2


def test_make_classification():
    X, y = make_classification(100, 10, n_informative=3, n_redundant=2, random_state=0)
    assert X.shape == (100, 10)
    assert len(np.unique(y)) == 2


def test_make_regression():
    X, y = make_regression(100, 8, n_informative=3, random_state=0)
    assert X.shape == (100, 8)
    assert y.shape == (100,)
    assert y.std() > 0


# --- Distances ---
def test_canberra():
    assert np.isclose(canberra_distance(np.array([1.0, 2.0]), np.array([2.0, 4.0])),
                      1.0 / 3.0 + 2.0 / 6.0)


def test_bray_curtis():
    d = bray_curtis_distance(np.array([1, 2, 3]), np.array([3, 2, 1]))
    assert 0 <= d <= 1


def test_wasserstein():
    d = wasserstein_distance(np.array([1, 2, 3]), np.array([4, 5, 6]))
    assert d > 0


# --- Metrics ---
def test_fbeta():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    assert np.isclose(fbeta_score(y_true, y_pred, beta=0.5), 0.7143, atol=0.01)


def test_avg_precision():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.8])
    assert 0.5 < avg_precision_score(y_true, y_score) <= 1.0


def test_jaccard():
    assert np.isclose(jaccard_score(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])), 0.666, atol=0.01)


# --- CV ---
def test_repeated_kfold():
    X = np.random.randn(50, 3)
    rkf = RepeatedKFold(n_splits=3, n_repeats=2, random_state=0)
    splits = list(rkf.split(X))
    assert len(splits) == 6


def test_leave_one_out():
    X = np.random.randn(10, 2)
    loo = LeaveOneOut()
    splits = list(loo.split(X))
    assert len(splits) == 10


# --- SelectFromModel ---
def test_select_from_model():
    from mastermlx import LogisticRegression
    X, y = make_classification(100, 10, n_informative=3, n_redundant=0, random_state=0)
    lr = LogisticRegression(lr=0.1, n_iter=50).fit(X, y)
    sel = SelectFromModel(lr, threshold="mean").fit(X, y)
    assert sel.support_.sum() <= 10
    assert sel.transform(X).shape[1] == sel.support_.sum()


# --- LR Schedulers ---
def test_step_lr():
    opt = SGD(lr=0.1)
    s = StepLR(opt, step_size=5, gamma=0.5)
    for _ in range(5):
        s.step()
    assert np.isclose(opt.lr, 0.05)


def test_cosine_lr():
    opt = SGD(lr=0.1)
    s = CosineLR(opt, T_max=10)
    s.step()
    assert opt.lr < 0.1


def test_reduce_on_plateau():
    opt = SGD(lr=0.1)
    s = ReduceLROnPlateau(opt, patience=3)
    for _ in range(4):
        s.step(1.0)  # not improving
    assert opt.lr < 0.1
