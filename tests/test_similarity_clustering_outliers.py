import numpy as np

from mastermlx.math_tools import (
    adj_mi,
    adj_rand,
    calinski_harabasz,
    cosine_sim,
    davies_bouldin,
    dot_sim,
    grubbs,
    iqr_outliers,
    kendall_tau,
    mod_zscore,
    mod_zscore_outliers,
    pairwise_cosine,
    pearson_r,
    silhouette,
    spearman_r,
    v_measure,
    zscore,
    zscore_outliers,
)


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def test_cosine_sim_identity():
    a = np.array([1.0, 2.0, 3.0])
    assert np.isclose(cosine_sim(a, a), 1.0)


def test_cosine_sim_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert np.isclose(cosine_sim(a, b), 0.0)


def test_dot_sim():
    assert np.isclose(dot_sim(np.array([2.0, 3.0]), np.array([1.0, 4.0])), 14.0)


def test_pearson_r_perfect():
    x = np.array([1.0, 2.0, 3.0])
    assert np.isclose(pearson_r(x, x), 1.0)
    assert np.isclose(pearson_r(x, -x), -1.0)


def test_spearman_r_monotonic():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 4.0, 9.0, 16.0])
    assert np.isclose(spearman_r(x, y), 1.0)


def test_kendall_tau():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    assert np.isclose(kendall_tau(x, y), 1.0)
    assert np.isclose(kendall_tau(x, -y), -1.0)


def test_pairwise_cosine():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    C = pairwise_cosine(X)
    assert C.shape == (3, 3)
    assert np.isclose(C[0, 0], 1.0)
    assert np.isclose(C[0, 1], 0.0)


# ---------------------------------------------------------------------------
# Clustering evaluation
# ---------------------------------------------------------------------------


def test_silhouette_perfect_clusters():
    X = np.array([[0.0], [0.2], [5.0], [5.2]])
    labels = np.array([0, 0, 1, 1])
    s = silhouette(X, labels)
    assert 0.7 < s <= 1.0


def test_davies_bouldin_lower_is_better():
    X = np.array([[0.0], [0.2], [5.0], [5.2]])
    labels = np.array([0, 0, 1, 1])
    db1 = davies_bouldin(X, labels)
    db2 = davies_bouldin(X, np.array([0, 1, 0, 1]))
    assert db1 < db2


def test_calinski_harabasz():
    X = np.array([[0.0], [0.2], [5.0], [5.2]])
    labels = np.array([0, 0, 1, 1])
    ch1 = calinski_harabasz(X, labels)
    ch2 = calinski_harabasz(X, np.array([0, 1, 0, 1]))
    assert ch1 > ch2


def test_adj_rand_identical_equals_one():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert np.isclose(adj_rand(a, a), 1.0)


def test_adj_rand_random_near_zero():
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 1, 0, 1])
    assert adj_rand(a, b) < 0.2


def test_adj_mi_and_v_measure():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([0, 0, 0, 1, 1, 1])
    ami = adj_mi(a, b)
    h, c, v = v_measure(a, b)
    assert 0.0 <= ami <= 1.0
    assert 0.0 <= h <= 1.0
    assert 0.0 <= c <= 1.0
    assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


def test_zscore():
    x = np.array([0.0] * 20 + [100.0])
    z = zscore(x)
    assert z[-1] == np.max(z)
    assert z[-1] > 3.0


def test_mod_zscore():
    x = np.array([1.0, 2.0, 2.0, 3.0, 100.0])
    z = mod_zscore(x)
    assert z[-1] > 3.0


def test_zscore_outliers():
    x = np.array([0.0] * 20 + [100.0])
    idx = zscore_outliers(x, thresh=2.0)
    assert len(idx) == 1
    assert idx[0] == 20


def test_mod_zscore_outliers():
    x = np.array([1.0, 2.0, 2.0, 3.0, 100.0])
    idx = mod_zscore_outliers(x, thresh=3.0)
    assert len(idx) == 1


def test_iqr_outliers():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 50.0])
    idx = iqr_outliers(x, factor=1.5)
    assert len(idx) == 1
    assert idx[0] == 6


def test_grubbs_detects_outlier():
    x = np.array([1.0, 2.0, 3.0, 4.0, 50.0])
    idx, g, crit, is_outlier = grubbs(x, alpha=0.05)
    assert idx == 4
    assert g > crit
    assert is_outlier
