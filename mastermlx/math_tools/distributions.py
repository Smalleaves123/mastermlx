from __future__ import annotations

import numpy as np
from math import gamma as _gamma, log as _log, sqrt as _sqrt, pi as _pi, exp as _exp, erf as _erf


# ---------------------------------------------------------------------------
# Helper CDFs
# ---------------------------------------------------------------------------

def _norm_cdf(x):
    """Standard normal CDF via the error function."""
    x = np.asarray(x, dtype=float)
    return np.array([0.5 * (1.0 + _erf(float(v) / _sqrt(2))) for v in x.ravel()]).reshape(x.shape)


# ---------------------------------------------------------------------------
# Continuous distributions (pdf, cdf, sample)
# ---------------------------------------------------------------------------


class Normal:
    """Normal distribution with given mean and std."""

    def __init__(self, loc=0.0, scale=1.0):
        self.loc = float(loc)
        self.scale = float(scale)
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        z = (x - self.loc) / self.scale
        return np.exp(-0.5 * z ** 2) / (self.scale * _sqrt(2.0 * _pi))

    def log_pdf(self, x):
        x = np.asarray(x, dtype=float)
        z = (x - self.loc) / self.scale
        return -0.5 * z ** 2 - _log(self.scale) - 0.5 * _log(2.0 * _pi)

    def cdf(self, x):
        return _norm_cdf((np.asarray(x, dtype=float) - self.loc) / self.scale)

    def sample(self, n=1, random_state=None):
        rng = np.random.default_rng(random_state)
        s = rng.normal(self.loc, self.scale, size=int(n))
        return float(s[0]) if int(n) == 1 else s


class Uniform:
    """Uniform distribution on [a, b]."""

    def __init__(self, a=0.0, b=1.0):
        self.a = float(a)
        self.b = float(b)
        if self.b <= self.a:
            raise ValueError("b must be greater than a")

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        h = 1.0 / (self.b - self.a)
        return np.where((x >= self.a) & (x <= self.b), h, 0.0)

    def log_pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where((x >= self.a) & (x <= self.b), -_log(self.b - self.a), -np.inf)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.clip((x - self.a) / (self.b - self.a), 0.0, 1.0)

    def sample(self, n=1, random_state=None):
        rng = np.random.default_rng(random_state)
        s = rng.uniform(self.a, self.b, size=int(n))
        return float(s[0]) if int(n) == 1 else s


class Exponential:
    """Exponential distribution with given rate (or scale=1/rate)."""

    def __init__(self, rate=1.0):
        self.rate = float(rate)
        if self.rate <= 0:
            raise ValueError("rate must be positive")

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= 0, self.rate * np.exp(-self.rate * x), 0.0)

    def log_pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= 0, _log(self.rate) - self.rate * x, -np.inf)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= 0, 1.0 - np.exp(-self.rate * x), 0.0)

    def sample(self, n=1, random_state=None):
        rng = np.random.default_rng(random_state)
        s = rng.exponential(1.0 / self.rate, size=int(n))
        return float(s[0]) if int(n) == 1 else s


class LogNormal:
    """Log-normal distribution: exp(Normal(mu, sigma))."""

    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = float(mu)
        self.sigma = float(sigma)
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        p = np.where(x > 0, np.exp(-0.5 * ((np.log(x) - self.mu) / self.sigma) ** 2) /
                     (x * self.sigma * _sqrt(2.0 * _pi)), 0.0)
        return p

    def log_pdf(self, x):
        x = np.asarray(x, dtype=float)
        log_x = np.log(np.maximum(x, 1e-12))
        return np.where(x > 0, -0.5 * ((log_x - self.mu) / self.sigma) ** 2
                        - log_x - _log(self.sigma) - 0.5 * _log(2.0 * _pi), -np.inf)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x > 0, _norm_cdf((np.log(np.maximum(x, 1e-12)) - self.mu) / self.sigma), 0.0)

    def sample(self, n=1, random_state=None):
        rng = np.random.default_rng(random_state)
        s = rng.lognormal(self.mu, self.sigma, size=int(n))
        return float(s[0]) if int(n) == 1 else s


class Chi2:
    """Chi-square distribution with k degrees of freedom."""

    def __init__(self, df=1):
        self.df = float(df)
        if self.df <= 0:
            raise ValueError("df must be positive")

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        k2 = self.df / 2.0
        return np.where(x > 0, x ** (k2 - 1.0) * np.exp(-x / 2.0) /
                        (2.0 ** k2 * _gamma(k2)), 0.0)

    def log_pdf(self, x):
        x = np.asarray(x, dtype=float)
        k2 = self.df / 2.0
        return np.where(x > 0, (k2 - 1.0) * np.log(np.maximum(x, 1e-12))
                        - x / 2.0 - k2 * _log(2.0) - _log(_gamma(k2)), -np.inf)

    def cdf(self, x):
        from .stats import _chi2_cdf
        return _chi2_cdf(x, self.df)

    def sample(self, n=1, random_state=None):
        rng = np.random.default_rng(random_state)
        s = rng.chisquare(self.df, size=int(n))
        return float(s[0]) if int(n) == 1 else s


class StudentT:
    """Student's t-distribution with df degrees of freedom."""

    def __init__(self, df=1):
        self.df = float(df)
        if self.df <= 0:
            raise ValueError("df must be positive")

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        nu = self.df
        return _gamma((nu + 1.0) / 2.0) / (_sqrt(nu * _pi) * _gamma(nu / 2.0)) * \
               (1.0 + x ** 2 / nu) ** (-(nu + 1.0) / 2.0)

    def log_pdf(self, x):
        x = np.asarray(x, dtype=float)
        nu = self.df
        return _log(_gamma((nu + 1.0) / 2.0)) - _log(_gamma(nu / 2.0)) \
               - 0.5 * _log(nu * _pi) \
               - (nu + 1.0) / 2.0 * np.log(1.0 + x ** 2 / nu)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        nu = self.df
        # Use regularized incomplete beta
        z = nu / (nu + x ** 2)
        return 0.5 * (1.0 + np.sign(x) * (1.0 - _betainc_reg(nu / 2.0, 0.5, z)))

    def sample(self, n=1, random_state=None):
        rng = np.random.default_rng(random_state)
        # t = Z / sqrt(V / nu)
        z = rng.normal(size=int(n))
        v = rng.chisquare(self.df, size=int(n))
        s = z / np.sqrt(v / self.df)
        return float(s[0]) if int(n) == 1 else s


# ---------------------------------------------------------------------------
# Discrete distributions
# ---------------------------------------------------------------------------


class Poisson:
    """Poisson distribution with rate lambda."""

    def __init__(self, lam=1.0):
        self.lam = float(lam)
        if self.lam <= 0:
            raise ValueError("lambda must be positive")

    def pmf(self, k):
        k = np.asarray(k, dtype=int)
        return np.exp(k * _log(self.lam) - self.lam - np.log(
            np.array([_gamma(ki + 1) for ki in k.ravel()], dtype=float)
        ).reshape(k.shape))

    def log_pmf(self, k):
        k = np.asarray(k, dtype=int)
        return k * _log(self.lam) - self.lam - np.log(
            np.array([_gamma(ki + 1) for ki in k.ravel()], dtype=float)
        ).reshape(k.shape)

    def cdf(self, k):
        k = np.asarray(k, dtype=int)
        result = np.zeros_like(k, dtype=float)
        for i, ki in enumerate(k.ravel()):
            s = 0.0
            term = np.exp(-self.lam)
            for j in range(int(ki) + 1):
                s += term
                term *= self.lam / (j + 1)
            result.ravel()[i] = min(s, 1.0)
        return result

    def sample(self, n=1, random_state=None):
        rng = np.random.default_rng(random_state)
        s = rng.poisson(self.lam, size=int(n))
        return int(s[0]) if int(n) == 1 else s


class Bernoulli:
    """Bernoulli distribution with probability p."""

    def __init__(self, p=0.5):
        self.p = float(p)
        if not 0 <= self.p <= 1:
            raise ValueError("p must be in [0, 1]")

    def pmf(self, k):
        k = np.asarray(k, dtype=float)
        return np.where(k == 1, self.p, np.where(k == 0, 1.0 - self.p, 0.0))

    def log_pmf(self, k):
        k = np.asarray(k, dtype=float)
        return np.where(k == 1, _log(self.p), np.where(k == 0, _log(1.0 - self.p), -np.inf))

    def cdf(self, k):
        k = np.asarray(k, dtype=float)
        return np.where(k < 0, 0.0, np.where(k >= 1, 1.0, 1.0 - self.p))

    def sample(self, n=1, random_state=None):
        rng = np.random.default_rng(random_state)
        s = rng.binomial(1, self.p, size=int(n))
        return int(s[0]) if int(n) == 1 else s


# ---------------------------------------------------------------------------
# Helper: regularized incomplete beta (used by t-distribution CDF)
# ---------------------------------------------------------------------------

def _betainc_reg(a, b, x):
    """Regularized incomplete beta via power series."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    a = float(a)
    b = float(b)
    x = float(x)
    lbeta = _log(_gamma(a)) + _log(_gamma(b)) - _log(_gamma(a + b))
    front = _exp(a * _log(x) + b * _log(1.0 - x) - lbeta) / a
    result = 1.0
    term = 1.0
    for k in range(1, 500):
        term *= (a + b + k - 1.0) * x / (a + k)
        result += term
        if abs(term) < 1e-16 * result:
            break
    return float(min(front * result, 1.0))
