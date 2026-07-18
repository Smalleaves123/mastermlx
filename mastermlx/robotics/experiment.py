from __future__ import annotations

import numpy as np

from .model import RobotModel
from .trajectory import plan_joint_trajectory


def _coerce_robot_model(robot):
    if isinstance(robot, RobotModel):
        return robot
    if isinstance(robot, (list, tuple)):
        return RobotModel.from_dh(robot)
    raise TypeError("robot must be a RobotModel or a DH link sequence")


def _trajectory_points(robot, trajectory):
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2:
        raise ValueError("trajectory must have shape (T, n_joints)")
    return np.asarray([robot.fk(q)[:3, 3] for q in trajectory], dtype=float)


class RobotExperiment:
    """High-level robotics workflow for kinematics, planning, and tracking."""

    def __init__(self, robot, pose_estimator=None, name=None):
        self.robot = _coerce_robot_model(robot)
        self.pose_estimator = pose_estimator
        self.name = self.robot.name if name is None else str(name)

    @classmethod
    def from_dh(cls, links, *, name="robot", base=None, tool=None, pose_estimator=None):
        robot = RobotModel.from_dh(links, name=name, base=base, tool=tool)
        return cls(robot, pose_estimator=pose_estimator, name=name)

    @classmethod
    def from_urdf(cls, xml_text, *, name=None, base_link=None, tip_link=None, pose_estimator=None):
        robot = RobotModel.from_urdf(xml_text, name=name, base_link=base_link, tip_link=tip_link)
        return cls(robot, pose_estimator=pose_estimator, name=robot.name if name is None else name)

    @property
    def links(self):
        return self.robot.links

    @property
    def base(self):
        return self.robot.base

    @property
    def tool(self):
        return self.robot.tool

    def fk(self, joint_values=None, return_all=False):
        return self.robot.fk(joint_values=joint_values, return_all=return_all)

    def positions(self, joint_values=None):
        return self.robot.positions(joint_values=joint_values)

    def jacobian(self, joint_values=None):
        return self.robot.jacobian(joint_values=joint_values)

    def ik(self, target, joint_values=None, **kwargs):
        return self.robot.ik(target, joint_values=joint_values, **kwargs)

    def plan_trajectory(
        self,
        q_start,
        q_goal,
        duration,
        num_waypoints=11,
        num_samples_per_segment=100,
        kind="quintic",
        smoothness=0.0,
        via_points=None,
    ):
        return plan_joint_trajectory(
            q_start,
            q_goal,
            duration,
            num_waypoints=num_waypoints,
            num_samples_per_segment=num_samples_per_segment,
            kind=kind,
            smoothness=smoothness,
            via_points=via_points,
        )

    def trajectory_report(self, trajectory):
        trajectory = np.asarray(trajectory, dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[1] != len(self.links):
            raise ValueError("trajectory must have shape (T, n_joints)")
        joint_deltas = np.diff(trajectory, axis=0)
        joint_path_length = float(np.sum(np.linalg.norm(joint_deltas, axis=1))) if joint_deltas.size else 0.0
        joint_max_step = float(np.max(np.linalg.norm(joint_deltas, axis=1))) if joint_deltas.size else 0.0
        end_effector = _trajectory_points(self.robot, trajectory)
        ee_deltas = np.diff(end_effector, axis=0)
        ee_path_length = float(np.sum(np.linalg.norm(ee_deltas, axis=1))) if ee_deltas.size else 0.0
        workspace_min = end_effector.min(axis=0)
        workspace_max = end_effector.max(axis=0)
        return {
            "robot": self.name,
            "n_steps": int(trajectory.shape[0]),
            "n_joints": int(trajectory.shape[1]),
            "joint_path_length": joint_path_length,
            "joint_max_step": joint_max_step,
            "end_effector_path_length": ee_path_length,
            "workspace_min": workspace_min,
            "workspace_max": workspace_max,
            "final_pose": self.robot.fk(trajectory[-1]),
        }

    def track_trajectory(self, trajectory, gains=(4.0, 0.4), dt=0.1, damping=0.0, state=None):
        """Track a joint trajectory using the simple simulator and PD control."""

        trajectory = np.asarray(trajectory, dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[1] != len(self.links):
            raise ValueError("trajectory must have shape (T, n_joints)")

        from ..sim.core import SimpleRobotSim

        kp, kd = gains
        sim = SimpleRobotSim(self.robot, state=state, dt=dt, damping=damping)
        states = [sim.state.copy()]
        poses = [sim.pose()]
        controls = []
        for target in trajectory:
            q_err = target - sim.q
            qd_err = -sim.qd
            action = kp * q_err + kd * qd_err
            controls.append(action)
            sim.step(action)
            states.append(sim.state.copy())
            poses.append(sim.pose())
        return np.asarray(states), poses, np.asarray(controls)

    def estimate_pose(self, odometry, dt, heading=None, position=None, pose=None):
        """Advance the attached planar pose estimator, if present."""

        if self.pose_estimator is None:
            raise RuntimeError("RobotExperiment does not have a pose_estimator")
        if not hasattr(self.pose_estimator, "step"):
            raise AttributeError("pose_estimator must define step()")
        return self.pose_estimator.step(odometry, dt, heading=heading, position=position, pose=pose)

    def reset_pose_estimator(self, x0=None, P0=None):
        if self.pose_estimator is None:
            raise RuntimeError("RobotExperiment does not have a pose_estimator")
        if hasattr(self.pose_estimator, "reset"):
            return self.pose_estimator.reset(x0=x0, P0=P0)
        raise AttributeError("pose_estimator does not define reset()")

    def summary(self):
        return {
            "name": self.name,
            "n_joints": len(self.links),
            "has_pose_estimator": self.pose_estimator is not None,
            "robot": self.robot.name,
        }


def compare_robot_models(models, joint_values, task="kinematics"):
    """Compare several robot models on a shared joint configuration."""

    if not models:
        raise ValueError("models must be non-empty")

    joint_values = np.asarray(joint_values, dtype=float).reshape(-1)
    leaderboard = []
    best_name = None
    best_score = -np.inf
    best_experiment = None

    for name, robot in models:
        experiment = RobotExperiment(robot, name=name)
        pose = experiment.fk(joint_values)
        score = float(np.linalg.norm(pose[:3, 3]))
        leaderboard.append((name, score))
        if score > best_score:
            best_score = score
            best_name = name
            best_experiment = experiment

    leaderboard.sort(key=lambda item: item[1], reverse=True)
    return {
        "task": task,
        "leaderboard": leaderboard,
        "best_name": best_name,
        "best_score": best_score,
        "best_experiment": best_experiment,
    }


__all__ = ["RobotExperiment", "compare_robot_models"]
