"""Control tools for robotics and dynamical systems."""

from .pid import PIDController
from .lqr import DiscreteLQR, finite_horizon_lqr, solve_discrete_are
from .mpc import LinearMPC, iLQR, rollout_dynamics

__all__ = [
    "DiscreteLQR",
    "LinearMPC",
    "PIDController",
    "finite_horizon_lqr",
    "iLQR",
    "rollout_dynamics",
    "solve_discrete_are",
]
