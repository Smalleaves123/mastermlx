import numpy as np

from mastermlx import finite_diff, gd, minimize, quad_gd


def test_finite_diff_matches_quadratic_gradient():
    def fun(x):
        return float(np.sum(x * x))

    grad = finite_diff(fun, np.array([1.0, -2.0]))

    assert np.allclose(grad, np.array([2.0, -4.0]), atol=1e-5)


def test_gd_minimizes_quadratic():
    def fun(x):
        return float(np.sum((x - 3.0) ** 2))

    def jac(x):
        return 2.0 * (x - 3.0)

    out = gd(fun, np.array([0.0, 0.0]), jac=jac, lr=0.1, max_iter=500, tol=1e-8)

    assert out.success
    assert np.allclose(out.x, np.array([3.0, 3.0]), atol=1e-5)
    assert out.fun < 1e-8
    assert out.history[0] > out.history[-1]


def test_gd_respects_box_bounds_and_minimize_alias():
    def fun(x):
        return float(np.sum((x - 4.0) ** 2))

    out = minimize(fun, np.array([0.0]), method="gd", lr=0.2, bounds=(0.0, 2.0), max_iter=200)

    assert out.success
    assert np.allclose(out.x, np.array([2.0]), atol=1e-5)


def test_quad_gd_minimizes_quadratic_with_fallback_or_cpp_backend():
    H = np.diag([2.0, 4.0])
    b = np.array([-4.0, 8.0])
    out = quad_gd(H, b, np.zeros(2), lr=0.1, max_iter=500, tol=1e-8)

    assert out.success
    assert np.allclose(out.x, np.array([2.0, -2.0]), atol=1e-5)
    assert out.fun < -7.9


def test_quad_gd_updates_cross_terms_simultaneously():
    H = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([-1.0, 2.0])
    out = quad_gd(H, b, np.zeros(2), lr=0.1, max_iter=1, tol=0.0)

    assert np.allclose(out.x, np.array([0.1, -0.2]))


def test_quad_gd_rejects_invalid_numeric_inputs():
    with np.testing.assert_raises(ValueError):
        quad_gd(np.eye(2), np.array([0.0, np.nan]), np.zeros(2))
    with np.testing.assert_raises(ValueError):
        quad_gd(np.eye(2), np.zeros(2), np.zeros(2), lr=0.0)
