import numpy as np

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


def test_systematic_resample_valid_indices():
    idx = systematic_resample(np.array([0.2, 0.3, 0.5]), rng=np.random.default_rng(1))
    assert idx.shape == (3,)
    assert np.all(idx >= 0)
    assert np.all(idx < 3)
