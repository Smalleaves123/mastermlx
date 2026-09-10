import numpy as np
import pytest

from mastermlx import get_backend, set_backend
from mastermlx.estimation import ExtendedKalmanFilter, KalmanFilter, ParticleFilter, systematic_resample


def test_kalman_filter_single_step():
    kf = KalmanFilter(
        x0=np.array([0.0]),
        P0=np.array([[1.0]]),
        F=np.array([[1.0]]),
        H=np.array([[1.0]]),
        Q=np.array([[0.0]]),
        R=np.array([[1.0]]),
    )

    x, P = kf.step(np.array([1.0]))

    assert np.allclose(x, np.array([0.5]))
    assert np.allclose(P, np.array([[0.5]]))


def test_kalman_filter_predict_with_control():
    kf = KalmanFilter(
        x0=np.array([0.0]),
        P0=np.array([[1.0]]),
        F=np.array([[1.0]]),
        H=np.array([[1.0]]),
        Q=np.array([[0.0]]),
        R=np.array([[1.0]]),
        B=np.array([[2.0]]),
    )

    x, _ = kf.predict(u=np.array([3.0]))
    assert np.allclose(x, np.array([6.0]))


def test_kalman_filter_multi_dimensional_update():
    kf = KalmanFilter(
        x0=np.array([0.0, 1.0]),
        P0=np.eye(2),
        F=np.array([[1.0, 0.1], [0.0, 1.0]]),
        H=np.eye(2),
        Q=np.zeros((2, 2)),
        R=0.5 * np.eye(2),
    )

    x_pred, P_pred = kf.predict()
    x, P = kf.update(np.array([1.0, 1.0]))

    assert x_pred.shape == (2,)
    assert P_pred.shape == (2, 2)
    assert x.shape == (2,)
    assert P.shape == (2, 2)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(P))


@pytest.mark.parametrize("backend", ["numpy", "auto"])
def test_kalman_covariance_remains_symmetric_positive_semidefinite(backend):
    old_backend = get_backend()
    try:
        set_backend(backend)
        rotation, _ = np.linalg.qr(
            np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0], [2.0, 1.0, -0.5]])
        )
        covariance = rotation @ np.diag([1e8, 1.0, 1e-8]) @ rotation.T
        kf = KalmanFilter(
            x0=np.zeros(3),
            P0=covariance,
            F=np.eye(3),
            H=np.array([[1.0, -2.0, 0.5]]),
            Q=1e-12 * np.eye(3),
            R=np.array([[1e-10]]),
        )
        for _ in range(100):
            _, covariance = kf.step([0.0])
    finally:
        set_backend(old_backend)

    assert np.array_equal(covariance, covariance.T)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-12


def test_kalman_filter_validates_constructor_and_runtime_dimensions():
    with pytest.raises(ValueError, match="P0 must have shape"):
        KalmanFilter([0.0, 0.0], [[1.0]], np.eye(2), [[1.0, 0.0]], np.eye(2), [[1.0]])

    kf = KalmanFilter([0.0, 0.0], np.eye(2), np.eye(2), [[1.0, 0.0]], np.eye(2), [[1.0]])
    with pytest.raises(ValueError, match="z must have size 1"):
        kf.update([0.0, 1.0])
    with pytest.raises(ValueError, match="F must have shape"):
        kf.predict(F=np.eye(3))


def test_extended_kalman_filter_matches_linear_case():
    A = np.array([[1.0]])
    H = np.array([[1.0]])

    def f(x, u):
        return A @ np.asarray(x, dtype=float).reshape(-1, 1) + np.array([[0.0]])

    def h(x, u):
        return H @ np.asarray(x, dtype=float).reshape(-1, 1)

    def F_jac(x, u):
        return A

    def H_jac(x, u):
        return H

    ekf = ExtendedKalmanFilter(
        x0=np.array([0.0]),
        P0=np.array([[1.0]]),
        f=f,
        h=h,
        F_jac=F_jac,
        H_jac=H_jac,
        Q=np.array([[0.0]]),
        R=np.array([[1.0]]),
    )

    x, P = ekf.step(np.array([1.0]))
    assert np.allclose(x, np.array([0.5]))
    assert np.allclose(P, np.array([[0.5]]))


@pytest.mark.parametrize("backend", ["numpy", "auto"])
def test_extended_kalman_predict_does_not_reapply_jacobian_to_nonlinear_state(backend):
    old_backend = get_backend()
    try:
        set_backend(backend)
        ekf = ExtendedKalmanFilter(
            x0=[2.0],
            P0=[[1.0]],
            f=lambda x, u: np.array([x[0] ** 2]),
            h=lambda x, u: x,
            F_jac=lambda x, u: np.array([[2.0 * x[0]]]),
            H_jac=lambda x, u: np.array([[1.0]]),
            Q=[[0.5]],
            R=[[1.0]],
        )
        state, covariance = ekf.predict()
    finally:
        set_backend(old_backend)

    assert np.allclose(state, [4.0])
    assert np.allclose(covariance, [[16.5]])


def test_extended_kalman_filter_validates_callback_dimensions():
    ekf = ExtendedKalmanFilter(
        x0=[0.0, 0.0],
        P0=np.eye(2),
        f=lambda x, u: [x[0]],
        h=lambda x, u: [x[0]],
        F_jac=lambda x, u: np.eye(2),
        H_jac=lambda x, u: [[1.0, 0.0]],
        Q=np.eye(2),
        R=[[1.0]],
    )

    with pytest.raises(ValueError, match="predicted state must have size 2"):
        ekf.predict()

    ekf.f_ = lambda x, u: x
    ekf.H_jac_ = lambda x, u: [[1.0]]
    with pytest.raises(ValueError, match="H_jac must have shape"):
        ekf.update([0.0])


@pytest.mark.parametrize("backend", ["numpy", "auto"])
def test_extended_kalman_covariance_remains_symmetric_positive_semidefinite(backend):
    old_backend = get_backend()
    try:
        set_backend(backend)
        measurement = np.array([[1.0, -2.0, 0.5]])
        rotation, _ = np.linalg.qr(
            np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0], [2.0, 1.0, -0.5]])
        )
        covariance = rotation @ np.diag([1e8, 1.0, 1e-8]) @ rotation.T
        ekf = ExtendedKalmanFilter(
            x0=np.zeros(3),
            P0=covariance,
            f=lambda x, u: x,
            h=lambda x, u: measurement @ x,
            F_jac=lambda x, u: np.eye(3),
            H_jac=lambda x, u: measurement,
            Q=1e-12 * np.eye(3),
            R=[[1e-10]],
        )
        for _ in range(100):
            _, covariance = ekf.step([0.0])
    finally:
        set_backend(old_backend)

    assert np.array_equal(covariance, covariance.T)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-12


def test_particle_filter_resample_collapses_to_single_particle():
    particles = np.array([[0.0], [1.0], [2.0], [3.0]])
    pf = ParticleFilter(
        particles=particles,
        transition=lambda particle, control: np.asarray(particle, dtype=float),
        likelihood=lambda particle, measurement: 1.0,
        rng=np.random.default_rng(0),
    )

    pf.weights_ = np.array([0.0, 0.0, 1.0, 0.0])
    resampled = pf.resample()

    assert np.allclose(resampled, np.array([[2.0], [2.0], [2.0], [2.0]]))
    assert np.allclose(pf.weights_, np.full(4, 0.25))


def test_particle_filter_update_falls_back_to_uniform_weights():
    particles = np.array([[0.0], [1.0], [2.0], [3.0]])
    pf = ParticleFilter(
        particles=particles,
        transition=lambda particle, control: np.asarray(particle, dtype=float),
        likelihood=lambda particle, measurement: 0.0,
        rng=np.random.default_rng(0),
    )

    weights = pf.update(np.array([0.0]))

    assert np.allclose(weights, np.full(4, 0.25))
    assert np.allclose(pf.weights_, np.full(4, 0.25))


@pytest.mark.parametrize("backend", ["numpy", "auto"])
def test_particle_filter_update_accumulates_sequential_evidence(backend):
    old_backend = get_backend()
    try:
        set_backend(backend)
        pf = ParticleFilter(
            particles=[[0.0], [1.0]],
            weights=[0.8, 0.2],
            likelihood=lambda particle, measurement: (
                2.0 if particle[0] == measurement else 1.0
            ),
        )
        first = pf.update(0.0).copy()
        second = pf.update(0.0)
    finally:
        set_backend(old_backend)

    assert np.allclose(first, [8.0 / 9.0, 1.0 / 9.0])
    assert np.allclose(second, [16.0 / 17.0, 1.0 / 17.0])


@pytest.mark.parametrize(
    "particles, match",
    [
        (np.empty((0, 2)), "non-zero particle"),
        (np.empty((2, 0)), "non-zero particle"),
        ([[0.0], [np.inf]], "finite values"),
    ],
)
def test_particle_filter_rejects_invalid_particle_arrays(particles, match):
    with pytest.raises(ValueError, match=match):
        ParticleFilter(particles)


@pytest.mark.parametrize("weights", [[0.5, -0.1], [0.5, np.nan], [0.5, np.inf]])
def test_particle_filter_rejects_invalid_initial_weights(weights):
    with pytest.raises(ValueError, match="finite and non-negative"):
        ParticleFilter([[0.0], [1.0]], weights=weights)


def test_particle_filter_predict_validates_callback_state_dimension():
    pf = ParticleFilter([[0.0], [1.0]], transition=lambda particle, control: [0.0, 1.0])

    with pytest.raises(ValueError, match="preserve the particle state dimension"):
        pf.predict()


@pytest.mark.parametrize("value", [np.nan, np.inf, [0.5]])
def test_particle_filter_update_validates_likelihood_output(value):
    pf = ParticleFilter([[0.0], [1.0]], likelihood=lambda particle, measurement: value)

    match = "one scalar" if isinstance(value, list) else "must be finite"
    with pytest.raises(ValueError, match=match):
        pf.update(0.0)


def test_systematic_resample_valid_indices():
    idx = systematic_resample(np.array([0.2, 0.3, 0.5]), rng=np.random.default_rng(1))
    assert idx.shape == (3,)
    assert np.all(idx >= 0)
    assert np.all(idx < 3)


@pytest.mark.parametrize("backend", ["numpy", "auto"])
@pytest.mark.parametrize(
    "weights",
    [[0.2, 0.3, 0.5], np.array([0.2, 0.3, 0.5], dtype=np.float32)],
)
def test_systematic_resample_accepts_array_like_weights_across_backends(backend, weights):
    old_backend = get_backend()
    try:
        set_backend(backend)
        actual = systematic_resample(weights, rng=np.random.default_rng(7))
        set_backend("numpy")
        expected = systematic_resample(weights, rng=np.random.default_rng(7))
    finally:
        set_backend(old_backend)

    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("backend", ["numpy", "auto"])
@pytest.mark.parametrize(
    "weights",
    [[0.5, -0.1, 0.6], [0.5, np.nan, 0.5], [0.5, np.inf]],
)
def test_systematic_resample_rejects_invalid_weights_across_backends(backend, weights):
    old_backend = get_backend()
    try:
        set_backend(backend)
        with pytest.raises(ValueError, match="finite and non-negative"):
            systematic_resample(weights, rng=np.random.default_rng(7))
    finally:
        set_backend(old_backend)
