import numpy as np
import pytest

from mastermlx import get_backend, set_backend
from mastermlx.control import (
    DiscreteLQR,
    LinearMPC,
    PIDController,
    finite_horizon_lqr,
    iLQR,
    rollout_dynamics,
    rollout_linear_dynamics,
)
from mastermlx.control.mpc import _load_cpp_control


def test_pid_controller_basic_response():
    pid = PIDController(kp=2.0, ki=0.5, kd=0.0, setpoint=1.0)

    u1 = pid.update(np.array(0.0), dt=1.0)
    u2 = pid.update(np.array(0.0), dt=1.0)

    assert np.allclose(u1, 2.5)
    assert np.allclose(u2, 3.0)


def test_pid_integral_limits_and_derivative_on_measurement():
    pid = PIDController(kp=0.0, ki=1.0, integral_limits=(-0.5, 0.5))
    assert np.allclose(pid.update(0.0, dt=1.0, setpoint=1.0), 0.5)
    assert np.allclose(pid.update(0.0, dt=1.0, setpoint=1.0), 0.5)

    derivative_pid = PIDController(kp=0.0, kd=1.0, derivative_on_measurement=True)
    derivative_pid.update(0.0, dt=1.0, setpoint=0.0)
    assert np.allclose(derivative_pid.update(0.0, dt=1.0, setpoint=1.0), 0.0)


def test_pid_validation_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="kp"):
        PIDController(kp=np.inf)
    with pytest.raises(ValueError, match="lower bound"):
        PIDController(output_limits=(1.0, 0.0))
    with pytest.raises(ValueError, match="dt"):
        PIDController().update(0.0, dt=0.0)


def test_discrete_lqr_gain_matches_scalar_solution():
    lqr = DiscreteLQR(
        A=np.array([[1.0]]),
        B=np.array([[1.0]]),
        Q=np.array([[1.0]]),
        R=np.array([[1.0]]),
    ).fit()

    assert np.allclose(lqr.K_, np.array([[0.61803399]]), atol=1e-6)
    assert np.allclose(lqr.control(np.array([1.0])), np.array([-0.61803399]), atol=1e-6)


def test_finite_horizon_lqr_and_mpc_one_step():
    A = np.array([[1.0]])
    B = np.array([[1.0]])
    Q = np.array([[1.0]])
    R = np.array([[1.0]])

    gains, _, _ = finite_horizon_lqr(A, B, Q, R, horizon=1, Qf=Q)
    assert np.allclose(gains[0], np.array([[0.5]]))

    mpc = LinearMPC(A, B, Q, R, horizon=1, Qf=Q)
    assert np.allclose(mpc.control(np.array([1.0])), np.array([-0.5]))


def test_finite_horizon_lqr_multi_step_shapes():
    A = np.array([[1.0]])
    B = np.array([[1.0]])
    Q = np.array([[1.0]])
    R = np.array([[1.0]])

    gains, costs, ref = finite_horizon_lqr(A, B, Q, R, horizon=4, Qf=Q)

    assert len(gains) == 4
    assert len(costs) == 5
    assert ref is None
    assert all(g.shape == (1, 1) for g in gains)
    assert all(p.shape == (1, 1) for p in costs)


def test_ilqr_reduces_quadratic_cost():
    def dynamics(x, u):
        return np.asarray(x, dtype=float) + np.asarray(u, dtype=float)

    x0 = np.array([2.0])
    U0 = np.zeros((5, 1), dtype=float)
    Q = np.array([[1.0]])
    R = np.array([[0.1]])
    Qf = np.array([[1.0]])

    U_opt, X_opt, cost_opt = iLQR(dynamics, x0, U0, Q, R, Qf, max_iter=20, tol=1e-9)
    X_base = rollout_dynamics(dynamics, x0, U0)
    base_cost = 0.0
    for t in range(U0.shape[0]):
        base_cost += float(X_base[t] @ Q @ X_base[t] + U0[t] @ R @ U0[t])
    base_cost += float(X_base[-1] @ Qf @ X_base[-1])

    assert U_opt.shape == U0.shape
    assert X_opt.shape == (U0.shape[0] + 1, 1)
    assert cost_opt < base_cost
    assert abs(X_opt[-1, 0]) < abs(X_base[-1, 0])


def test_ilqr_accepts_broadcast_references_and_validates_costs():
    def dynamics(x, u):
        return np.asarray(x, dtype=float) + np.asarray(u, dtype=float)

    args = (dynamics, np.array([1.0]), np.zeros((3, 1)), np.eye(1), np.eye(1), np.eye(1))
    U_opt, X_opt, cost = iLQR(*args, x_ref=np.zeros(1), u_ref=np.zeros(1), max_iter=3)
    assert U_opt.shape == (3, 1)
    assert X_opt.shape == (4, 1)
    assert np.isfinite(cost)
    with pytest.raises(ValueError, match="Q must have shape"):
        iLQR(dynamics, np.array([1.0]), np.zeros((3, 1)), np.eye(2), np.eye(1), np.eye(1))
    with pytest.raises(ValueError, match="line_search"):
        iLQR(*args, line_search=())


def test_linear_mpc_box_solver_respects_bounds_and_supports_references():
    A = np.array([[1.0]])
    B = np.array([[1.0]])
    Q = np.array([[1.0]])
    R = np.array([[0.1]])
    mpc = LinearMPC(A, B, Q, R, horizon=4, u_bounds=(-0.25, 0.25))

    control = mpc.control(np.array([2.0]), x_ref=np.zeros((5, 1)))

    assert -0.25 <= control[0] <= 0.25
    assert mpc.qp_converged_
    assert mpc.last_qp_iterations_ > 0


def test_control_validation_rejects_incompatible_lqr_matrices():
    with pytest.raises(ValueError, match="positive definite"):
        DiscreteLQR(np.eye(2), np.ones((2, 1)), np.eye(2), np.zeros((1, 1)))
    with pytest.raises(ValueError, match="horizon"):
        LinearMPC(np.eye(1), np.ones((1, 1)), np.eye(1), np.eye(1), horizon=0)


def test_cpp_linear_rollout_matches_numpy_when_available():
    cpp = _load_cpp_control("auto")
    if cpp is None:
        pytest.skip("C++ control extension is unavailable")
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    x0 = np.array([1.0, 0.2])
    U = np.arange(12, dtype=float).reshape(6, 2)[:, :1] / 10.0
    old = get_backend()
    try:
        set_backend("numpy")
        reference = rollout_linear_dynamics(A, B, x0, U)
        set_backend("auto")
        accelerated = rollout_linear_dynamics(A, B, x0, U)
        assert np.allclose(reference, accelerated, atol=1e-12)
    finally:
        set_backend(old)
