from __future__ import annotations

import numpy as np

from ..robotics.model import RobotModel
from ..robotics.visualizer import plot_chain


def step_state(state, action, dt=0.1, damping=0.0):
    """Advance a simple second-order joint state with optional damping."""

    state = np.asarray(state, dtype=float).reshape(-1)
    action = np.asarray(action, dtype=float).reshape(-1)
    if state.size % 2 != 0:
        raise ValueError("state must contain position and velocity blocks")
    n = state.size // 2
    if action.size != n:
        raise ValueError("action dimension must match the number of joints")
    q = state[:n]
    qd = state[n:]
    qdd = action - damping * qd
    qd_next = qd + float(dt) * qdd
    q_next = q + float(dt) * qd_next
    return np.concatenate([q_next, qd_next])


class SimpleRobotSim:
    """Tiny deterministic simulator for serial robots."""

    def __init__(self, robot: RobotModel, state=None, dt=0.1, damping=0.0):
        self.robot = robot
        self.dt = float(dt)
        self.damping = float(damping)
        n = len(robot.links)
        if state is None:
            self.state = np.zeros(2 * n, dtype=float)
        else:
            self.state = np.asarray(state, dtype=float).reshape(-1)
        if self.state.size != 2 * n:
            raise ValueError("state must have shape (2 * n_joints,)")

    @property
    def q(self):
        n = len(self.robot.links)
        return self.state[:n]

    @property
    def qd(self):
        n = len(self.robot.links)
        return self.state[n:]

    def pose(self):
        return self.robot.fk(self.q)

    def step(self, action):
        self.state = step_state(self.state, action, dt=self.dt, damping=self.damping)
        return self.state

    def rollout(self, actions):
        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 2 or actions.shape[1] != len(self.robot.links):
            raise ValueError("actions must have shape (T, n_joints)")
        states = [self.state.copy()]
        poses = [self.pose()]
        for action in actions:
            self.step(action)
            states.append(self.state.copy())
            poses.append(self.pose())
        return np.asarray(states), poses

    def render(self, joint_values=None, ax=None, annotate=False):
        q = self.q if joint_values is None else np.asarray(joint_values, dtype=float).reshape(-1)
        points = self.robot.positions(q)
        return plot_chain(points[:, :2] if points.shape[1] >= 2 else points, ax=ax, annotate=annotate)

    def reset(self, state=None):
        n = len(self.robot.links)
        if state is None:
            self.state = np.zeros(2 * n, dtype=float)
        else:
            self.state = np.asarray(state, dtype=float).reshape(-1)
            if self.state.size != 2 * n:
                raise ValueError("state must have shape (2 * n_joints,)")
        return self.state
