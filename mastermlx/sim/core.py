from __future__ import annotations

from collections.abc import Mapping

import numpy as np

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


def step_state_batch(states, actions, dt=0.1, damping=0.0):
    """Advance a batch of second-order joint states."""

    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=float)
    if states.ndim != 2 or actions.ndim != 2:
        raise ValueError("states and actions must be two-dimensional")
    if states.shape[0] != actions.shape[0] or states.shape[1] % 2 != 0:
        raise ValueError("states and actions must have matching batch dimensions")
    n_joints = states.shape[1] // 2
    if actions.shape[1] != n_joints:
        raise ValueError("action dimension must match the number of joints")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
        raise ValueError("states and actions must contain only finite values")
    q = states[:, :n_joints]
    qd = states[:, n_joints:]
    qdd = actions - float(damping) * qd
    qd_next = qd + float(dt) * qdd
    q_next = q + float(dt) * qd_next
    return np.concatenate([q_next, qd_next], axis=1)


class SimpleRobotSim:
    """Tiny deterministic simulator for serial robots."""

    def __init__(self, robot, state=None, dt=0.1, damping=0.0):
        self.robot = robot
        self.dt = float(dt)
        self.damping = float(damping)
        n = robot.n_joints
        if state is None:
            self.state = np.zeros(2 * n, dtype=float)
        else:
            self.state = np.asarray(state, dtype=float).reshape(-1)
        if self.state.size != 2 * n:
            raise ValueError("state must have shape (2 * n_joints,)")

    @property
    def q(self):
        n = self.robot.n_joints
        return self.state[:n]

    @property
    def qd(self):
        n = self.robot.n_joints
        return self.state[n:]

    def pose(self):
        return self.robot.fk(self.q)

    def step(self, action):
        self.state = step_state(self.state, action, dt=self.dt, damping=self.damping)
        return self.state

    def rollout(self, actions):
        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 2 or actions.shape[1] != self.robot.n_joints:
            raise ValueError("actions must have shape (T, n_joints)")
        states = [self.state.copy()]
        poses = [self.pose()]
        for action in actions:
            self.step(action)
            states.append(self.state.copy())
            poses.append(self.pose())
        return np.asarray(states), poses

    def rollout_batch(self, actions, initial_states=None):
        """Roll out independent second-order episodes without Python loops."""

        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 3 or actions.shape[2] != self.robot.n_joints:
            raise ValueError("actions must have shape (batch, time, n_joints)")
        if initial_states is None:
            states = np.zeros((actions.shape[0], 2 * self.robot.n_joints), dtype=float)
        else:
            states = np.asarray(initial_states, dtype=float).copy()
            expected = (actions.shape[0], 2 * self.robot.n_joints)
            if states.shape != expected:
                raise ValueError(f"initial_states must have shape {expected}")
        if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(states)):
            raise ValueError("actions and initial_states must contain only finite values")
        history = [states.copy()]
        for action in np.moveaxis(actions, 1, 0):
            states = step_state_batch(states, action, dt=self.dt, damping=self.damping)
            history.append(states.copy())
        return np.stack(history, axis=1)

    def render(self, joint_values=None, ax=None, annotate=False):
        q = self.q if joint_values is None else np.asarray(joint_values, dtype=float).reshape(-1)
        points = self.robot.positions(q)
        return plot_chain(points[:, :2] if points.shape[1] >= 2 else points, ax=ax, annotate=annotate)

    def reset(self, state=None):
        n = self.robot.n_joints
        if state is None:
            self.state = np.zeros(2 * n, dtype=float)
        else:
            self.state = np.asarray(state, dtype=float).reshape(-1)
            if self.state.size != 2 * n:
                raise ValueError("state must have shape (2 * n_joints,)")
        return self.state


def track_joint_trajectory(
    robot,
    trajectory,
    *,
    gains=(4.0, 0.4),
    dt=0.1,
    damping=0.0,
    state=None,
):
    """Track joint targets with the shared deterministic PD controller."""

    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] != robot.n_joints:
        raise ValueError("trajectory must have shape (T, n_joints)")
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("trajectory must contain only finite values")
    gains = np.asarray(gains, dtype=float).reshape(-1)
    if gains.shape != (2,) or not np.all(np.isfinite(gains)):
        raise ValueError("gains must contain finite proportional and derivative values")
    kp, kd = gains
    simulation = SimpleRobotSim(robot, state=state, dt=dt, damping=damping)
    states = [simulation.state.copy()]
    poses = [simulation.pose()]
    controls = []
    for target in trajectory:
        action = kp * (target - simulation.q) - kd * simulation.qd
        controls.append(action)
        simulation.step(action)
        states.append(simulation.state.copy())
        poses.append(simulation.pose())
    return np.asarray(states), poses, np.asarray(controls)


class RobotSimulation:
    """Small Gym-like environment for deterministic robot task simulation.

    The environment deliberately has no Gym dependency. ``reset`` returns
    ``(observation, info)`` and ``step`` returns
    ``(observation, reward, terminated, truncated, info)``.
    """

    def __init__(
        self,
        world,
        *,
        dt=0.05,
        damping=0.0,
        max_steps=1000,
        target_position=None,
        target_tolerance=0.05,
        grasp_distance=0.1,
        joint_limits=None,
        terminate_on_collision=True,
    ):
        if not hasattr(world, "robot") or not hasattr(world, "hit"):
            raise TypeError("world must expose robot and hit()")
        self.world = world
        self.robot = world.robot
        self.dt = float(dt)
        self.damping = float(damping)
        self.max_steps = int(max_steps)
        self.target_position = None if target_position is None else self._position(target_position, "target_position")
        self.target_tolerance = float(target_tolerance)
        self.grasp_distance = float(grasp_distance)
        configured_limits = self.robot.joint_limits if joint_limits is None else joint_limits
        from ..robotics.constraints import validate_joint_limits

        self.joint_limits = validate_joint_limits(configured_limits, self.robot.n_joints)
        self.terminate_on_collision = bool(terminate_on_collision)
        if self.dt <= 0.0 or not np.isfinite(self.dt):
            raise ValueError("dt must be positive and finite")
        if self.damping < 0.0 or not np.isfinite(self.damping):
            raise ValueError("damping must be non-negative and finite")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.target_tolerance <= 0.0 or not np.isfinite(self.target_tolerance):
            raise ValueError("target_tolerance must be positive and finite")
        if self.grasp_distance <= 0.0 or not np.isfinite(self.grasp_distance):
            raise ValueError("grasp_distance must be positive and finite")
        self.sim = SimpleRobotSim(self.robot, dt=self.dt, damping=self.damping)
        self.n_steps = 0
        self.time = 0.0
        self.gripper = 1.0
        self.rng = np.random.default_rng()

    @staticmethod
    def _position(value, name):
        position = np.asarray(value, dtype=float).reshape(-1)
        if position.size == 2:
            position = np.concatenate([position, [0.0]])
        if position.size != 3 or not np.all(np.isfinite(position)):
            raise ValueError(f"{name} must be a finite 2D or 3D position")
        return position

    def _parse_action(self, action):
        gripper = None
        if isinstance(action, Mapping):
            joint_action = action.get("joint", action.get("joints", action.get("action")))
            if joint_action is None:
                raise ValueError("action mapping must contain 'joint' or 'action'")
            gripper = action.get("gripper")
        else:
            values = np.asarray(action, dtype=float).reshape(-1)
            if values.size == self.robot.n_joints + 1:
                joint_action, gripper = values[:-1], values[-1]
            else:
                joint_action = values
        joint_action = np.asarray(joint_action, dtype=float).reshape(-1)
        if joint_action.size != self.robot.n_joints or not np.all(np.isfinite(joint_action)):
            raise ValueError(f"joint action must contain {self.robot.n_joints} finite values")
        if gripper is not None:
            gripper = float(gripper)
            if not np.isfinite(gripper):
                raise ValueError("gripper action must be finite")
        return joint_action, gripper

    def _tcp_position(self):
        return np.asarray(self.robot.fk(self.sim.q)[:3, 3], dtype=float)

    def _object_positions(self):
        objects = getattr(self.world, "objects", ())
        if not objects:
            return np.empty((0, 3), dtype=float)
        return np.stack([np.asarray(item.position, dtype=float) for item in objects])

    def observation(self):
        objects = getattr(self.world, "objects", ())
        return {
            "state": self.sim.state.copy(),
            "q": self.sim.q.copy(),
            "qd": self.sim.qd.copy(),
            "tcp_position": self._tcp_position(),
            "tcp_pose": self.robot.fk(self.sim.q),
            "gripper": float(self.gripper),
            "object_positions": self._object_positions(),
            "object_attached": np.asarray([item.attached for item in objects], dtype=bool),
        }

    def _info(self, *, event=None, collision=False, truncated=False):
        info = {
            "step": int(self.n_steps),
            "time": float(self.time),
            "collision": bool(collision),
            "joint_limit_violation": None,
            "attached_object": getattr(self.world, "attached_object_name", lambda: None)(),
        }
        if self.joint_limits is not None:
            from ..robotics.constraints import joint_limit_violation

            info["joint_limit_violation"] = float(joint_limit_violation(self.sim.q, self.joint_limits))
        if self.target_position is not None:
            distance = float(np.linalg.norm(self._tcp_position() - self.target_position))
            info["distance_to_target"] = distance
            info["success"] = bool(distance <= self.target_tolerance)
        else:
            info["success"] = False
        if event is not None:
            info["event"] = event
        if truncated:
            info["termination_reason"] = "max_steps"
        elif collision and self.terminate_on_collision:
            info["termination_reason"] = "collision"
        elif info["success"]:
            info["termination_reason"] = "target_reached"
        return info

    def reset(self, state=None, *, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.sim.reset(state)
        if hasattr(self.world, "reset_objects"):
            self.world.reset_objects()
        self.n_steps = 0
        self.time = 0.0
        self.gripper = 1.0
        return self.observation(), {"seed": seed, "step": 0, "time": 0.0}

    def step(self, action):
        joint_action, gripper = self._parse_action(action)
        previous_distance = None
        if self.target_position is not None:
            previous_distance = float(np.linalg.norm(self._tcp_position() - self.target_position))
        previous_gripper = self.gripper
        if gripper is not None:
            self.gripper = gripper
        self.sim.step(joint_action)
        tcp_position = self._tcp_position()
        event = None
        if gripper is not None and previous_gripper > 0.5 >= self.gripper:
            if hasattr(self.world, "grasp_object"):
                event = self.world.grasp_object(tcp_position, self.grasp_distance)
        elif gripper is not None and previous_gripper <= 0.5 < self.gripper:
            if hasattr(self.world, "release_object"):
                event = self.world.release_object()
        if hasattr(self.world, "sync_attached_objects"):
            self.world.sync_attached_objects(tcp_position)
        collision = bool(self.world.hit(self.sim.q))
        self.n_steps += 1
        self.time += self.dt
        info = self._info(event=event, collision=collision)
        current_distance = info.get("distance_to_target")
        reward = -0.001 * float(np.dot(joint_action, joint_action))
        if previous_distance is not None and current_distance is not None:
            reward += previous_distance - current_distance
        terminated = bool(info["success"] or (collision and self.terminate_on_collision))
        truncated = bool(self.n_steps >= self.max_steps and not terminated)
        if truncated:
            info["termination_reason"] = "max_steps"
        return self.observation(), float(reward), terminated, truncated, info

    def rollout(self, actions, *, reset=True, seed=None):
        """Run one episode and return arrays suitable for analysis."""

        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 2:
            raise ValueError("actions must have shape (time, action_dim)")
        if reset:
            observation, reset_info = self.reset(seed=seed)
        else:
            observation, reset_info = self.observation(), None
        observations = [observation]
        rewards = []
        terminated = []
        truncated = []
        infos = []
        for action in actions:
            observation, reward, done, cutoff, info = self.step(action)
            observations.append(observation)
            rewards.append(reward)
            terminated.append(done)
            truncated.append(cutoff)
            infos.append(info)
            if done or cutoff:
                break
        return {
            "observations": observations,
            "rewards": np.asarray(rewards, dtype=float),
            "terminated": np.asarray(terminated, dtype=bool),
            "truncated": np.asarray(truncated, dtype=bool),
            "infos": infos,
            "reset_info": reset_info,
        }
