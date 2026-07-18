from __future__ import annotations

import numpy as np
from math import gamma as _gamma, log as _log, sqrt as _sqrt, exp as _exp


# ---------------------------------------------------------------------------
# Distribution CDFs for p-value computation (no scipy)
# ---------------------------------------------------------------------------

def _norm_cdf(x):
    """Standard normal CDF via the error function."""
    from math import erf as _erf_func
    x = np.asarray(x, dtype=float)
    return np.array([0.5 * (1.0 + _erf_func(float(v) / _sqrt(2))) for v in x.ravel()]).reshape(x.shape)


def _chi2_cdf(x, df):
    """Chi-square CDF via series or continued fraction."""
    x = max(float(x), 0.0)
    df = float(df)
    if df <= 0 or x <= 0:
        return 0.0
    # For df=1, use normal CDF directly
    if abs(df - 1.0) < 1e-10:
        return float(_norm_cdf(_sqrt(x)))
    # For large x, use normal approximation
    if x > 20 * df and df > 2:
        z = (x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))
        z /= _sqrt(2.0 / (9.0 * df))
        return float(_norm_cdf(z))
    s = x / 2.0
    a = df / 2.0
    log_term = a * _log(max(s, 1e-15)) - s - _log(_gamma(a))
    if log_term < -700:
        # Series underflows → use normal approx at lower threshold
        return float(_norm_cdf((x - df) / _sqrt(2.0 * df)))
    term = _exp(log_term)
    total = term
    for k in range(1, 500):
        term *= s / (a + k)
        total += term
        if abs(term) < 1e-16 * abs(total):
            break
    return float(min(total, 1.0))


def _f_cdf(x, df1, df2):
    """F-distribution CDF via regularized incomplete beta."""
    x = float(x)
    df1 = float(df1)
    df2 = float(df2)
    if x <= 0 or df1 <= 0 or df2 <= 0:
        return 0.0
    if x > 1e6:  # very large F, essentially 1.0
        return 1.0
    z = df1 * x / (df1 * x + df2)
    return _betainc(df1 / 2.0, df2 / 2.0, z)


def _betainc(a, b, x):
    """Regularized incomplete beta via power series (I_x(a,b))."""
    a = float(a)
    b = float(b)
    x = float(x)
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = _log(_gamma(a)) + _log(_gamma(b)) - _log(_gamma(a + b))
    # Use the power series: I_x(a,b) = x^a (1-x)^b / (a*B(a,b)) * sum
    front = _exp(a * _log(x) + b * _log(1.0 - x) - lbeta) / a
    result = 1.0
    term = 1.0
    for k in range(1, 500):
        term *= (a + b + k - 1.0) * x / (a + k)
        result += term
        if abs(term) < 1e-16 * result:
            break
    return min(float(front * result), 1.0)


def _rank(x):
    """Rank 1D data with tie averaging."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    ranks = np.empty(order.size, dtype=float)
    ranks[order] = np.arange(1.0, x.size + 1.0)
    uniq, inv = np.unique(x, return_inverse=True)
    for g in range(uniq.size):
        mask = inv == g
        ranks[mask] = np.mean(ranks[mask])
    return ranks


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def mann_whitney(x, y, alternative="two-sided"):
    """Mann-Whitney U rank-sum test for two independent samples."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        raise ValueError("x and y must be non-empty")
    alt = alternative
    if alt not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    combined = np.concatenate([x, y])
    ranks = _rank(combined)
    r1 = float(np.sum(ranks[:x.size]))
    n1, n2 = x.size, y.size
    u1 = r1 - n1 * (n1 + 1.0) / 2.0
    u2 = n1 * n2 - u1

    if alt == "less":
        u = u1
    elif alt == "greater":
        u = u2
    else:
        u = min(u1, u2)

    # Normal approximation with tie correction
    mu = n1 * n2 / 2.0
    t = np.bincount(combined.astype(int)) if np.all(combined == combined.astype(int)) else np.array([])
    tie_corr = np.sum(t * (t - 1.0) * (t + 1.0)) if t.size > 0 else 0.0
    sigma = _sqrt(n1 * n2 / 12.0 * ((n1 + n2 + 1.0) - tie_corr / ((n1 + n2) * (n1 + n2 - 1.0))))
    sigma = max(sigma, 1e-12)
    z = (u - mu) / sigma
    p = 2.0 * _norm_cdf(-abs(z)) if alt == "two-sided" else _norm_cdf(-z)
    return float(u), float(p)


def wilcoxon(x, y=None, alternative="two-sided"):
    """Wilcoxon signed-rank test for paired samples or one-sample."""
    x = np.asarray(x, dtype=float).ravel()
    if y is not None:
        y = np.asarray(y, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty")
    if y is not None and x.size != y.size:
        raise ValueError("x and y must have the same length")
    alt = alternative
    if alt not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    if y is not None:
        d = x - y
    else:
        d = x
    d = d[d != 0.0]
    n = d.size
    if n == 0:
        return 0.0, 1.0

    abs_d = np.abs(d)
    ranks = _rank(abs_d)
    w_pos = float(np.sum(ranks[d > 0]))
    w_neg = float(np.sum(ranks[d < 0]))
    w = w_pos if alt != "greater" else w_neg

    mu = n * (n + 1.0) / 4.0
    sigma = _sqrt(n * (n + 1.0) * (2.0 * n + 1.0) / 24.0)
    sigma = max(sigma, 1e-12)
    z = (w - mu) / sigma
    p = 2.0 * _norm_cdf(-abs(z)) if alt == "two-sided" else _norm_cdf(-z)
    return float(w), float(p)


def kruskal(*groups):
    """Kruskal-Wallis H test for independent samples."""
    group_arrays = [np.asarray(g, dtype=float).ravel() for g in groups]
    group_arrays = [g for g in group_arrays if g.size > 0]
    k = len(group_arrays)
    if k < 2:
        raise ValueError("Need at least two non-empty groups")

    all_data = np.concatenate(group_arrays)
    ranks = _rank(all_data)
    n = all_data.size
    r_sums = []
    start = 0
    for g in group_arrays:
        r_sums.append(float(np.sum(ranks[start:start + g.size])))
        start += g.size

    h = 12.0 / (n * (n + 1.0)) * sum(r**2 / g.size for r, g in zip(r_sums, group_arrays)) - 3.0 * (n + 1.0)

    # Tie correction
    uniq, cnt = np.unique(all_data, return_counts=True)
    tie_corr = 1.0 - np.sum(cnt ** 3 - cnt) / (n ** 3 - n)
    h = h / max(tie_corr, 1e-12)
    p = 1.0 - _chi2_cdf(h, k - 1)
    p = max(p, 0.0)
    return float(h), float(p)


def chi2_contingency(observed):
    """Chi-squared test of independence for a contingency table."""
    obs = np.asarray(observed, dtype=float)
    if obs.ndim != 2 or obs.shape[0] < 2 or obs.shape[1] < 2:
        raise ValueError("observed must be a 2D array with at least 2 rows and 2 columns")
    n = np.sum(obs)
    row_sum = obs.sum(axis=1, keepdims=True)
    col_sum = obs.sum(axis=0, keepdims=True)
    exp = row_sum @ col_sum / n
    chi2 = float(np.sum((obs - exp) ** 2 / np.maximum(exp, 1e-12)))
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    p = 1.0 - _chi2_cdf(chi2, dof)
    p = max(p, 0.0)
    return float(chi2), int(dof), float(p)


def f_oneway(*groups):
    """One-way ANOVA F-test."""
    group_arrays = [np.asarray(g, dtype=float).ravel() for g in groups]
    group_arrays = [g for g in group_arrays if g.size > 0]
    k = len(group_arrays)
    if k < 2:
        raise ValueError("Need at least two non-empty groups")

    all_data = np.concatenate(group_arrays)
    grand_mean = np.mean(all_data)
    ssb = 0.0
    ssw = 0.0
    for g in group_arrays:
        ssb += g.size * (np.mean(g) - grand_mean) ** 2
        ssw += np.sum((g - np.mean(g)) ** 2)

    df_b = k - 1
    df_w = all_data.size - k
    msb = ssb / df_b if df_b > 0 else 0.0
    msw = ssw / df_w if df_w > 0 else 1e-12
    f_stat = msb / max(msw, 1e-12)
    p = 1.0 - _f_cdf(f_stat, df_b, df_w)
    p = max(p, 0.0)
    return float(f_stat), float(p)


def ks_test(x, cdf="norm", *args):
    """One-sample Kolmogorov-Smirnov test."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty")
    x_sorted = np.sort(x)
    n = x.size

    if cdf == "norm":
        loc = args[0] if len(args) > 0 else np.mean(x)
        scale = args[1] if len(args) > 1 else np.std(x)
        scale = max(scale, 1e-12)
        cdf_vals = _norm_cdf((x_sorted - loc) / scale)
    elif cdf == "expon":
        scale = args[0] if len(args) > 0 else np.mean(x)
        scale = max(scale, 1e-12)
        cdf_vals = 1.0 - np.exp(-x_sorted / scale)
    elif cdf == "uniform":
        lo = args[0] if len(args) > 0 else 0.0
        hi = args[1] if len(args) > 1 else 1.0
        cdf_vals = np.clip((x_sorted - lo) / (hi - lo), 0.0, 1.0)
    else:
        raise ValueError("cdf must be 'norm', 'expon', or 'uniform'")

    ecdf = (np.arange(1, n + 1) - 0.5) / n
    d = float(np.max(np.abs(ecdf - cdf_vals)))
    # Kolmogorov approximation for p-value
    z = d * _sqrt(n)
    p = 2.0 * np.sum([(-1) ** (k - 1) * _exp(-2.0 * k ** 2 * z ** 2) for k in range(1, 100)])
    p = float(np.clip(p, 0.0, 1.0))
    return float(d), float(p)
