"""Control tools for robotics and dynamical systems."""

from .pid import PIDController
from .lqr import DiscreteLQR, finite_horizon_lqr, solve_discrete_are
from .mpc import LinearMPC, control_backend_report, iLQR, rollout_dynamics, rollout_linear_dynamics

__all__ = [
    "DiscreteLQR",
    "LinearMPC",
    "PIDController",
    "control_backend_report",
    "finite_horizon_lqr",
    "iLQR",
    "rollout_dynamics",
    "rollout_linear_dynamics",
    "solve_discrete_are",
]
