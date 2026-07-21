import numpy as np

from mastermlx.math_tools import (
    Bernoulli,
    Chi2,
    Exponential,
    LogNormal,
    Normal,
    Poisson,
    StudentT,
    Uniform,
    chi2_contingency,
    f_oneway,
    ks_test,
    kruskal,
    mann_whitney,
    wilcoxon,
)


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


def test_normal_pdf_peaks_at_loc():
    n = Normal(0, 1)
    assert n.pdf(0) > n.pdf(2)
    assert np.isclose(n.cdf(0), 0.5, atol=0.01)
    assert n.sample(10).shape == (10,)


def test_uniform_range():
    u = Uniform(0, 5)
    assert np.isclose(u.pdf(1), 0.2)
    assert u.cdf(3) == 0.6
    assert np.all((u.sample(100) >= 0) & (u.sample(100) <= 5))


def test_exponential_decay():
    e = Exponential(rate=2.0)
    assert np.isclose(e.pdf(0), 2.0)
    assert e.cdf(0) == 0.0
    assert e.cdf(10) > 0.99


def test_lognormal():
    ln = LogNormal(0, 0.5)
    x = np.array([1.0, 2.0])
    assert np.all(np.isfinite(ln.pdf(x)))
    assert np.all(ln.cdf(x) >= 0)


def test_chi2():
    c = Chi2(df=3)
    assert np.isclose(c.cdf(0), 0.0, atol=0.01)
    assert c.sample(5).shape == (5,)


def test_student_t():
    t = StudentT(df=10)
    # pdf should be symmetric
    assert np.isclose(t.pdf(-1), t.pdf(1))
    assert np.isclose(t.cdf(-5) + (1 - t.cdf(5)), 0.0, atol=0.01)


def test_poisson():
    p = Poisson(lam=3)
    assert np.isclose(p.pmf(0), np.exp(-3))
    assert p.sample(5).shape == (5,)


def test_bernoulli():
    b = Bernoulli(p=0.7)
    assert np.isclose(b.pmf(1), 0.7)
    assert np.isclose(b.pmf(0), 0.3)
    assert np.isclose(b.cdf(0.5), 0.3)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def test_mann_whitney_different_groups():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([10, 11, 12, 13, 14])
    u, p = mann_whitney(x, y)
    assert u == 0.0
    assert p < 0.05


def test_mann_whitney_same_group():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    y = rng.normal(size=50)
    _, p = mann_whitney(x, y)
    assert p > 0.01  # unlikely to be significant


def test_wilcoxon_shift():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([6, 7, 8, 9, 10])  # y is consistently higher
    _, p = wilcoxon(x, y)
    assert p < 0.05


def test_wilcoxon_one_sample():
    _, p = wilcoxon(np.array([1.5, 2, 3, 4, 5]))
    assert 0 <= p <= 1


def test_kruskal_different_groups():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([10, 11, 12, 13, 14])
    c = np.array([20, 21, 22, 23, 24])
    h, p = kruskal(a, b, c)
    assert h > 5
    assert p < 0.05


def test_chi2_contingency_independence():
    # Coin flips by two people: independent
    obs = np.array([[25, 25], [25, 25]])
    chi2, dof, p = chi2_contingency(obs)
    assert chi2 < 1.0
    assert p > 0.3  # clearly independent


def test_chi2_contingency_dependence():
    obs = np.array([[50, 0], [0, 50]])
    chi2, dof, p = chi2_contingency(obs)
    assert chi2 > 50
    # clearly dependent, p-value should be small
    assert p < 0.05


def test_f_oneway():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([10, 11, 12, 13, 14])
    f, p = f_oneway(a, b)
    assert f > 50
    assert p < 0.01


def test_ks_test_normal():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 200)
    d, p = ks_test(x, "norm", 0, 1)
    assert 0 <= d <= 1
    assert p > 0.001  # should not reject normal fit at any reasonable alpha
