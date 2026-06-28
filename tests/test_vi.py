import numpy as np

from mastermlx.probabilistic import digamma, has_converged, log_gamma, log_sum_exp, normalize_log_probs


def test_log_sum_exp_matches_manual_computation():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = log_sum_exp(x, axis=1)
    expected = np.log(np.sum(np.exp(x), axis=1))
    assert np.allclose(out, expected)


def test_normalize_log_probs_returns_row_normalized_probs():
    log_probs = np.log(np.array([[0.2, 0.8], [0.5, 0.5]]))
    probs, log_norm = normalize_log_probs(log_probs, axis=1)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert log_norm.shape == (2,)


def test_special_function_helpers_return_finite_values():
    x = np.array([0.5, 1.0, 2.0])
    assert np.all(np.isfinite(digamma(x)))
    assert np.all(np.isfinite(log_gamma(x)))


def test_has_converged_checks_absolute_tolerance():
    assert has_converged(1.00001, 1.0, 1e-4)
    assert not has_converged(1.01, 1.0, 1e-4)
