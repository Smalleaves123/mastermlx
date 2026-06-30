import numpy as np

from mastermlx.control import DiscreteLQR, LinearMPC, PIDController, finite_horizon_lqr, iLQR, rollout_dynamics


def test_pid_controller_basic_response():
    pid = PIDController(kp=2.0, ki=0.5, kd=0.0, setpoint=1.0)

    u1 = pid.update(np.array(0.0), dt=1.0)
    u2 = pid.update(np.array(0.0), dt=1.0)

    assert np.allclose(u1, 2.5)
    assert np.allclose(u2, 3.0)


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
