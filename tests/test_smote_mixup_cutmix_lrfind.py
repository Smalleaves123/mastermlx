import numpy as np

from mastermlx.math_tools import cutmix, mixup, smote


# ---------------------------------------------------------------------------
# SMOTE
# ---------------------------------------------------------------------------


def test_smote_balances_dataset():
    # Class 0 has 6, class 1 has 3
    X = np.vstack([np.random.randn(6, 2) * 0.5 + [0, 0],
                   np.random.randn(3, 2) * 0.5 + [5, 5]])
    y = np.array([0]*6 + [1]*3)
    Xr, yr = smote(X, y, k=2, random_state=42)
    assert Xr.shape[0] == 12  # both classes reach 6
    assert np.sum(yr == 0) == 6
    assert np.sum(yr == 1) == 6


def test_smote_synthetic_in_bounds():
    np.random.seed(1)
    X = np.random.randn(100, 3)
    y = np.array([0] * 80 + [1] * 20)
    Xr, yr = smote(X, y, k=3, random_state=0)
    # Synthetic samples should be between class 1 neighbors
    synth = Xr[yr == 1][20:]  # the new ones
    orig = X[y == 1]
    for col in range(3):
        lo, hi = orig[:, col].min(), orig[:, col].max()
        if hi - lo > 0.1:
            assert np.all(synth[:, col] >= lo - 1.0)
            assert np.all(synth[:, col] <= hi + 1.0)


# ---------------------------------------------------------------------------
# MixUp
# ---------------------------------------------------------------------------


def test_mixup_shape():
    X = np.random.randn(50, 5)
    y = np.random.randn(50)
    Xm, y_perm, w = mixup(X, y, alpha=0.2, random_state=0)
    assert Xm.shape == X.shape
    assert y_perm.shape == y.shape
    assert w.shape == (50, 2)
    assert np.allclose(np.sum(w, axis=1), 1.0)


def test_mixup_convex():
    X = np.ones((20, 3))
    y = np.zeros(20)
    Xm, _, _ = mixup(X, y, alpha=1.0, random_state=0)
    # All values should still be 1.0 (convex combination of 1 and 1)
    assert np.allclose(Xm, 1.0)


# ---------------------------------------------------------------------------
# CutMix
# ---------------------------------------------------------------------------


def test_cutmix_shape():
    X = np.random.randn(30, 8)
    y = np.arange(30)
    Xm, y_perm, lam = cutmix(X, y, alpha=1.0, random_state=0)
    assert Xm.shape == X.shape
    assert y_perm.shape == y.shape
    assert lam.shape == (30,)
    assert np.all(lam >= 0.5)
    assert np.all(lam <= 1.0)


def test_cutmix_differs():
    X = np.random.randn(100, 10)
    y = np.zeros(100)
    Xm, _, _ = cutmix(X, y, alpha=1.0, random_state=0)
    # With alpha=1, mixing is common
    assert not np.allclose(Xm, X)
