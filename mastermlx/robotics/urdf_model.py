"""General spatial robot model backed by a serial URDF chain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..planning import rrt, rrt_star, smooth
from .constraints import clip_joint_values, validate_joint_limits
from .collision import (
    MeshObstacle,
    _path_samples,
    mesh_collision_report,
    path_collision_free,
    path_collision_report,
    path_collision_summary,
    robot_collision_report,
)
from .results import RobotResult
from .dynamics import LinkInertia, normalize_link_inertias
from .spatial_dynamics import (
    spatial_computed_torque,
    spatial_coriolis_forces,
    spatial_forward_dynamics,
    spatial_gravity_forces,
    spatial_inverse_dynamics,
    spatial_mass_matrix,
)
from .urdf_parser import URDFSerialChain


def _pose_orientation_error(current, target):
    """Return a world-frame small-angle orientation error."""

    current_rotation = np.asarray(current, dtype=float)[:3, :3]
    target_rotation = np.asarray(target, dtype=float)[:3, :3]
    return 0.5 * (
        np.cross(current_rotation[:, 0], target_rotation[:, 0])
        + np.cross(current_rotation[:, 1], target_rotation[:, 1])
        + np.cross(current_rotation[:, 2], target_rotation[:, 2])
    )


@dataclass
class URDFRobotModel:
    """Kinematics model for a general serial spatial URDF chain.

    This model preserves URDF joint origins and axes, including non-zero RPY
    origins and joints that rotate or translate about arbitrary directions. It
    deliberately focuses on kinematics and task-space IK; the existing
    :class:`RobotModel` remains the compatibility path for DH dynamics and
    workcell features.
    """

    chain: URDFSerialChain
    name: str = "robot"
    base: np.ndarray | None = None
    tool: np.ndarray | None = None
    joint_limits: np.ndarray | None = None
    resource_dir: str | Path | None = None
    link_inertias: tuple[LinkInertia, ...] | None = None

    def __post_init__(self):
        if not isinstance(self.chain, URDFSerialChain):
            raise TypeError("chain must be a URDFSerialChain")
        if self.base is not None:
            self.base = np.asarray(self.base, dtype=float)
            if self.base.shape != (4, 4):
                raise ValueError("base must have shape (4, 4)")
        if self.tool is not None:
            self.tool = np.asarray(self.tool, dtype=float)
            if self.tool.shape != (4, 4):
                raise ValueError("tool must have shape (4, 4)")
        if self.resource_dir is not None:
            self.resource_dir = Path(self.resource_dir)
        configured_inertias = self.chain.link_inertias if self.link_inertias is None else self.link_inertias
        if configured_inertias and all(value is not None for value in configured_inertias):
            self.link_inertias = normalize_link_inertias(configured_inertias, len(self.chain.joints))
        elif self.link_inertias is not None:
            self.link_inertias = normalize_link_inertias(self.link_inertias, len(self.chain.joints))
        limits = self.chain.joint_limits if self.joint_limits is None else self.joint_limits
        self.joint_limits = validate_joint_limits(limits, self.n_joints)

    @classmethod
    def from_urdf(
        cls,
        xml_text,
        *,
        name=None,
        base_link=None,
        tip_link=None,
        base=None,
        tool=None,
        joint_limits=None,
        resource_dir=None,
        link_inertias=None,
    ):
        chain = URDFSerialChain.from_urdf(
            xml_text, base_link=base_link, tip_link=tip_link
        )
        return cls(
            chain=chain,
            name=chain.tip_link if name is None else str(name),
            base=base,
            tool=tool,
            joint_limits=joint_limits,
            resource_dir=resource_dir,
            link_inertias=link_inertias,
        )

    @property
    def n_joints(self):
        return self.chain.n_joints

    @property
    def joint_names(self):
        return self.chain.joint_names

    @property
    def joint_types(self):
        return self.chain.joint_types

    def default_joint_values(self):
        if self.joint_limits is None:
            return np.zeros(self.n_joints, dtype=float)
        zero = np.zeros(self.n_joints, dtype=float)
        if np.all((zero >= self.joint_limits[:, 0]) & (zero <= self.joint_limits[:, 1])):
            return zero
        return np.mean(self.joint_limits, axis=1)

    def validate_joint_values(self, joint_values=None, *, check_limits=False):
        values = self.chain.validate_joint_values(joint_values)
        if check_limits:
            if self.joint_limits is not None and (
                np.any(values < self.joint_limits[:, 0])
                or np.any(values > self.joint_limits[:, 1])
            ):
                raise ValueError("joint_values exceeds configured joint_limits")
        return values

    def clip_joint_values(self, joint_values):
        values = self.validate_joint_values(joint_values)
        return clip_joint_values(values, self.joint_limits)

    def fk(self, joint_values=None, *, return_all=False):
        values = self.default_joint_values() if joint_values is None else joint_values
        values = self.validate_joint_values(values, check_limits=self.joint_limits is not None)
        return self.chain.forward_kinematics(
            values, base=self.base, tool=self.tool, return_all=return_all
        )

    def forward_kinematics(self, joint_values=None, *, return_all=False):
        return self.fk(joint_values=joint_values, return_all=return_all)

    def fk_batch(self, joint_values):
        values = np.asarray(joint_values, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_joints:
            raise ValueError(f"joint_values must have shape (n_samples, {self.n_joints})")
        if self.joint_limits is not None:
            for row in values:
                self.validate_joint_values(row, check_limits=True)
        return self.chain.forward_kinematics_batch(values, base=self.base, tool=self.tool)

    def positions(self, joint_values=None):
        values = self.default_joint_values() if joint_values is None else joint_values
        self.validate_joint_values(values, check_limits=self.joint_limits is not None)
        return self.chain.positions(values, base=self.base, tool=self.tool)

    def frame_positions(self, joint_values=None):
        """Return the base, intermediate, and tool frame positions."""

        return self.positions(joint_values=joint_values)

    def frame_positions_batch(self, joint_values):
        """Return all chain frame positions for a batch of configurations."""

        values = np.asarray(joint_values, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_joints:
            raise ValueError(f"joint_values must have shape (n_samples, {self.n_joints})")
        if self.joint_limits is not None:
            for row in values:
                self.validate_joint_values(row, check_limits=True)
        return np.asarray(
            [self.frame_positions(row) for row in values], dtype=float
        )

    def collision_meshes(self, joint_values=None):
        """Return transformed collision meshes for all links in the chain."""

        from .urdf_parser import URDFCollision

        values = self.default_joint_values() if joint_values is None else joint_values
        values = self.validate_joint_values(values, check_limits=self.joint_limits is not None)
        _, frames = self.chain.forward_kinematics(
            values, base=self.base, tool=None, return_all=True
        )
        meshes = []
        for link_frame, collisions in zip(frames, self.chain.link_collisions):
            for collision in collisions:
                if not isinstance(collision, URDFCollision):
                    raise TypeError("chain link collisions must contain URDFCollision values")
                mesh = self._collision_mesh(collision)
                origin = np.eye(4, dtype=float)
                origin[:3, :3] = self._rpy_matrix(collision.origin_rpy)
                origin[:3, 3] = collision.origin_xyz
                meshes.append(mesh.transformed(link_frame @ origin))
        return tuple(meshes)

    @staticmethod
    def _rpy_matrix(values):
        from .transforms import rpy_to_matrix

        return rpy_to_matrix(*values)

    def _collision_mesh(self, collision):
        kind = collision.geometry_type
        if kind == "mesh":
            filename = Path(collision.filename)
            if not filename.is_absolute() and self.resource_dir is not None:
                filename = Path(self.resource_dir) / filename
            mesh = MeshObstacle.from_file(filename)
            return MeshObstacle(mesh.vertices * np.asarray(collision.scale), mesh.faces)
        if kind == "box":
            size = np.asarray(collision.size, dtype=float)
            vertices = np.asarray([
                [x * size[0] / 2.0, y * size[1] / 2.0, z * size[2] / 2.0]
                for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)
            ])
            faces = np.asarray([
                [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
            ])
            return MeshObstacle(vertices, faces)
        if kind == "sphere":
            return self._sphere_mesh(float(collision.radius))
        if kind in {"cylinder", "capsule"}:
            return self._cylinder_mesh(float(collision.radius), float(collision.length))
        raise ValueError(f"unsupported URDF collision geometry: {kind!r}")

    @staticmethod
    def _sphere_mesh(radius, latitude=8, longitude=16):
        vertices = []
        for i in range(latitude + 1):
            phi = np.pi * i / latitude
            for j in range(longitude):
                theta = 2.0 * np.pi * j / longitude
                vertices.append([
                    radius * np.sin(phi) * np.cos(theta),
                    radius * np.sin(phi) * np.sin(theta),
                    radius * np.cos(phi),
                ])
        faces = []
        for i in range(latitude):
            for j in range(longitude):
                next_j = (j + 1) % longitude
                a = i * longitude + j
                b = i * longitude + next_j
                c = (i + 1) * longitude + j
                d = (i + 1) * longitude + next_j
                faces.extend([[a, c, b], [b, c, d]])
        return MeshObstacle(np.asarray(vertices), np.asarray(faces))

    @staticmethod
    def _cylinder_mesh(radius, length, segments=16):
        vertices = []
        for z in (-length / 2.0, length / 2.0):
            vertices.extend([
                [radius * np.cos(2.0 * np.pi * i / segments),
                 radius * np.sin(2.0 * np.pi * i / segments), z]
                for i in range(segments)
            ])
        vertices.extend([[0.0, 0.0, -length / 2.0], [0.0, 0.0, length / 2.0]])
        bottom, top = 2 * segments, 2 * segments + 1
        faces = []
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.extend([
                [i, next_i, segments + next_i],
                [i, segments + next_i, segments + i],
                [bottom, next_i, i],
                [top, segments + i, segments + next_i],
            ])
        return MeshObstacle(np.asarray(vertices), np.asarray(faces))

    def collision_report(
        self,
        joint_values=None,
        obstacles=(),
        *,
        link_radius=0.0,
        occupancy_grid=None,
    ):
        """Return geometric and optional occupancy-grid collision diagnostics."""

        obstacles = list(obstacles)
        report = robot_collision_report(
            self,
            joint_values,
            obstacles,
            link_radius=link_radius,
            occupancy_grid=occupancy_grid,
        )
        mesh_report = mesh_collision_report(
            self.collision_meshes(joint_values), obstacles, link_radius=link_radius
        )
        if mesh_report["minimum_clearance"] < report["minimum_clearance"]:
            report["minimum_clearance"] = mesh_report["minimum_clearance"]
            report["closest"] = mesh_report["closest"]
        report["collision"] = bool(report["collision"] or mesh_report["collision"])
        report["hits"].extend(mesh_report["hits"])
        return report

    def path_collision_free(
        self,
        joint_path,
        obstacles=(),
        *,
        clearance=0.0,
        link_radius=0.0,
        interpolation_step=0.05,
        occupancy_grid=None,
    ):
        """Return whether a joint path clears geometry and an optional voxel map."""

        if any(self.chain.link_collisions):
            samples = _path_samples(self, joint_path, interpolation_step)
            def sample_free(sample):
                report = self.collision_report(
                    sample,
                    obstacles,
                    link_radius=link_radius,
                    occupancy_grid=occupancy_grid,
                )
                return bool(not report["collision"] and report["minimum_clearance"] >= clearance)
            return bool(all(sample_free(sample) for sample in samples))
        return path_collision_free(
            self,
            joint_path,
            obstacles,
            clearance=clearance,
            link_radius=link_radius,
            interpolation_step=interpolation_step,
            occupancy_grid=occupancy_grid,
        )

    def path_collision_summary(
        self,
        joint_path,
        obstacles=(),
        *,
        link_radius=0.0,
        interpolation_step=0.05,
        occupancy_grid=None,
    ):
        """Return batched collision and clearance data for a joint path."""

        if any(self.chain.link_collisions):
            report = self.path_collision_report(
                joint_path,
                obstacles,
                link_radius=link_radius,
                interpolation_step=interpolation_step,
                occupancy_grid=occupancy_grid,
            )
            clearances = report["clearances"]
            kind_codes = {None: 0, "point": 1, "segment": 2, "mesh": 3, "occupancy": 4}
            closest = [item["closest"] for item in report["reports"]]
            return RobotResult({
                "collision": bool(report["collision"]),
                "minimum_clearance": float(np.min(clearances)),
                "first_collision_index": report["first_collision_index"],
                "n_samples": report["n_samples"],
                "samples": report["samples"],
                "clearances": clearances,
                "closest_kind": np.asarray(
                    [kind_codes.get(item["kind"], 0) for item in closest], dtype=np.int8
                ),
                "closest_index": np.asarray(
                    [-1 if item["index"] is None else item["index"] for item in closest],
                    dtype=np.int64,
                ),
                "closest_obstacle_index": np.asarray(
                    [-1 if item["obstacle_index"] is None else item["obstacle_index"] for item in closest],
                    dtype=np.int64,
                ),
            })
        return path_collision_summary(
            self,
            joint_path,
            obstacles,
            link_radius=link_radius,
            interpolation_step=interpolation_step,
            occupancy_grid=occupancy_grid,
        )

    def path_collision_report(
        self,
        joint_path,
        obstacles=(),
        *,
        link_radius=0.0,
        interpolation_step=0.05,
        occupancy_grid=None,
    ):
        """Return detailed collision reports for a joint path."""

        if any(self.chain.link_collisions):
            obstacles = list(obstacles)
            samples = _path_samples(self, joint_path, interpolation_step)
            reports = [
                self.collision_report(
                    sample,
                    obstacles,
                    link_radius=link_radius,
                    occupancy_grid=occupancy_grid,
                )
                for sample in samples
            ]
            clearances = np.asarray([item["minimum_clearance"] for item in reports])
            first = next((index for index, item in enumerate(reports) if item["collision"]), None)
            return RobotResult({
                "collision": bool(first is not None),
                "minimum_clearance": float(np.min(clearances)),
                "first_collision_index": first,
                "n_samples": samples.shape[0],
                "samples": samples,
                "clearances": clearances,
                "reports": reports,
            })
        return path_collision_report(
            self,
            joint_path,
            obstacles,
            link_radius=link_radius,
            interpolation_step=interpolation_step,
            occupancy_grid=occupancy_grid,
        )

    def plan_joint_path(
        self,
        q_start,
        q_goal,
        bounds=None,
        *,
        planner="rrt",
        obstacles=(),
        occupancy_grid=None,
        smooth_path=True,
        shortcut_attempts=100,
        collision_step=0.05,
        clearance=0.0,
        link_radius=0.0,
        workers=1,
        **planner_kwargs,
    ):
        """Plan a collision-free spatial-URDF joint path with RRT or RRT*."""

        q_start = self.validate_joint_values(q_start, check_limits=self.joint_limits is not None)
        q_goal = self.validate_joint_values(q_goal, check_limits=self.joint_limits is not None)
        if bounds is None:
            if self.joint_limits is None:
                raise ValueError("bounds are required when joint_limits are not configured")
            bounds = self.joint_limits.copy()
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (self.n_joints, 2) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("bounds must have shape (n_joints, 2) with lower < upper")
        if self.joint_limits is not None:
            bounds = np.column_stack([
                np.maximum(bounds[:, 0], self.joint_limits[:, 0]),
                np.minimum(bounds[:, 1], self.joint_limits[:, 1]),
            ])
            if np.any(bounds[:, 0] >= bounds[:, 1]):
                raise ValueError("bounds do not overlap configured joint_limits")
        if np.any(q_start < bounds[:, 0]) or np.any(q_start > bounds[:, 1]):
            raise ValueError("q_start must be inside bounds")
        if np.any(q_goal < bounds[:, 0]) or np.any(q_goal > bounds[:, 1]):
            raise ValueError("q_goal must be inside bounds")
        clearance = float(clearance)
        link_radius = float(link_radius)
        collision_step = float(collision_step)
        if clearance < 0.0 or link_radius < 0.0 or collision_step <= 0.0:
            raise ValueError("clearance and link_radius must be non-negative; collision_step must be positive")
        obstacles = list(obstacles)

        def hit(values):
            report = self.collision_report(
                values, obstacles, link_radius=link_radius, occupancy_grid=occupancy_grid
            )
            return bool(report["collision"] or report["minimum_clearance"] < clearance)

        def edge_free(start, end, step):
            return self.path_collision_free(
                np.vstack([start, end]),
                obstacles,
                clearance=clearance,
                link_radius=link_radius,
                interpolation_step=step,
                occupancy_grid=occupancy_grid,
            )

        if self.path_collision_free(
            np.vstack([q_start, q_goal]), obstacles, clearance=clearance,
            link_radius=link_radius, interpolation_step=collision_step,
            occupancy_grid=occupancy_grid,
        ):
            return np.vstack([q_start, q_goal])
        planner_name = str(planner).lower()
        options = dict(planner_kwargs)
        options.setdefault("collision_step", collision_step)
        options.setdefault("edge_free", edge_free)
        options.setdefault("workers", workers)
        if planner_name == "rrt":
            path = rrt(q_start, q_goal, bounds, hit=hit, **options)
        elif planner_name in {"rrt_star", "rrt*"}:
            path = rrt_star(q_start, q_goal, bounds, hit=hit, **options)
        else:
            raise ValueError("planner must be one of: rrt, rrt_star")
        if path is None:
            raise RuntimeError(f"{planner_name} could not find a collision-free joint-space path")
        if smooth_path:
            candidate = smooth(
                path,
                hit=hit,
                n=int(shortcut_attempts),
                random_state=options.get("random_state"),
                edge_free=edge_free,
                workers=workers,
            )
            if self.path_collision_free(
                candidate, obstacles, clearance=clearance, link_radius=link_radius,
                interpolation_step=collision_step, occupancy_grid=occupancy_grid,
            ):
                path = candidate
        if not self.path_collision_free(
            path, obstacles, clearance=clearance, link_radius=link_radius,
            interpolation_step=collision_step, occupancy_grid=occupancy_grid,
        ):
            raise RuntimeError("planner returned a path that does not satisfy collision checks")
        return path

    def jacobian(self, joint_values=None):
        values = self.default_joint_values() if joint_values is None else joint_values
        values = self.validate_joint_values(values, check_limits=self.joint_limits is not None)
        return self.chain.geometric_jacobian(values, base=self.base, tool=self.tool)

    def geometric_jacobian(self, joint_values=None):
        return self.jacobian(joint_values=joint_values)

    def mass_matrix(self, joint_values=None, *, link_inertias=None):
        """Return the spatial URDF joint-space mass matrix."""

        return spatial_mass_matrix(
            self, joint_values, link_inertias=self.link_inertias if link_inertias is None else link_inertias
        )

    def gravity_forces(
        self, joint_values=None, *, gravity=(0.0, 0.0, -9.81), link_inertias=None
    ):
        """Return gravity forces for a spatial URDF chain."""

        return spatial_gravity_forces(
            self,
            joint_values,
            gravity=gravity,
            link_inertias=self.link_inertias if link_inertias is None else link_inertias,
        )

    def coriolis_forces(
        self, joint_values, joint_velocities, *, link_inertias=None, epsilon=1e-6
    ):
        """Return Coriolis and centrifugal forces for a spatial URDF chain."""

        return spatial_coriolis_forces(
            self,
            joint_values,
            joint_velocities,
            link_inertias=self.link_inertias if link_inertias is None else link_inertias,
            epsilon=epsilon,
        )

    def inverse_dynamics(
        self,
        joint_values,
        joint_velocities,
        joint_accelerations,
        *,
        gravity=(0.0, 0.0, -9.81),
        link_inertias=None,
        include_coriolis=True,
    ):
        """Return spatial inverse-dynamics torques."""

        return spatial_inverse_dynamics(
            self,
            joint_values,
            joint_velocities,
            joint_accelerations,
            gravity=gravity,
            link_inertias=self.link_inertias if link_inertias is None else link_inertias,
            include_coriolis=include_coriolis,
        )

    def forward_dynamics(
        self,
        joint_values,
        joint_velocities,
        joint_torques,
        *,
        gravity=(0.0, 0.0, -9.81),
        link_inertias=None,
        include_coriolis=True,
    ):
        """Return spatial joint accelerations from applied torques."""

        return spatial_forward_dynamics(
            self,
            joint_values,
            joint_velocities,
            joint_torques,
            gravity=gravity,
            link_inertias=self.link_inertias if link_inertias is None else link_inertias,
            include_coriolis=include_coriolis,
        )

    def computed_torque_control(
        self,
        joint_values,
        joint_velocities,
        desired_positions,
        desired_velocities=None,
        desired_accelerations=None,
        *,
        kp=25.0,
        kd=8.0,
        gravity=(0.0, 0.0, -9.81),
        link_inertias=None,
        torque_limits=None,
    ):
        """Return a computed-torque control command for a desired joint state."""

        return spatial_computed_torque(
            self,
            joint_values,
            joint_velocities,
            desired_positions,
            desired_velocities,
            desired_accelerations,
            kp=kp,
            kd=kd,
            gravity=gravity,
            link_inertias=self.link_inertias if link_inertias is None else link_inertias,
            torque_limits=torque_limits,
        )

    def jacobian_batch(self, joint_values):
        values = np.asarray(joint_values, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_joints:
            raise ValueError(f"joint_values must have shape (n_samples, {self.n_joints})")
        if self.joint_limits is not None:
            for row in values:
                self.validate_joint_values(row, check_limits=True)
        return self.chain.geometric_jacobian_batch(values, base=self.base, tool=self.tool)

    def end_effector_velocity(self, joint_values, joint_velocities, *, translational=False):
        q = self.validate_joint_values(joint_values, check_limits=self.joint_limits is not None)
        qd = self.validate_joint_values(joint_velocities)
        jacobian = self.jacobian(q)
        return (jacobian[:3] if translational else jacobian) @ qd

    def differential_ik(
        self,
        joint_values,
        task_velocity,
        *,
        damping=1e-4,
        translational=False,
    ):
        q = self.validate_joint_values(joint_values, check_limits=self.joint_limits is not None)
        velocity = np.asarray(task_velocity, dtype=float).reshape(-1)
        expected = 3 if translational else 6
        if velocity.size != expected:
            raise ValueError(f"task_velocity must contain {expected} values")
        damping = float(damping)
        if not np.isfinite(damping) or damping < 0.0:
            raise ValueError("damping must be finite and non-negative")
        jacobian = self.jacobian(q)
        if translational:
            jacobian = jacobian[:3]
        gram = jacobian @ jacobian.T
        regularized = gram + damping**2 * np.eye(gram.shape[0], dtype=float)
        return jacobian.T @ np.linalg.solve(regularized, velocity)

    def inverse_kinematics(
        self,
        target,
        joint_values=None,
        *,
        max_iter=200,
        tol=1e-6,
        damping=1e-4,
        step_size=1.0,
        position_weight=1.0,
        orientation_weight=1.0,
        return_info=False,
    ):
        """Solve a position or full-pose target with damped least squares."""

        target = np.asarray(target, dtype=float)
        position_only = target.shape == (3,)
        if not position_only and target.shape != (4, 4):
            raise ValueError("target must have shape (3,) or (4, 4)")
        if not np.all(np.isfinite(target)):
            raise ValueError("target must contain only finite values")
        max_iter = int(max_iter)
        tol = float(tol)
        damping = float(damping)
        step_size = float(step_size)
        if max_iter < 1 or tol <= 0.0 or damping < 0.0 or step_size <= 0.0:
            raise ValueError("max_iter, tol, damping, and step_size must be valid positive values")
        if position_weight <= 0.0 or orientation_weight <= 0.0:
            raise ValueError("position_weight and orientation_weight must be positive")

        q = self.default_joint_values() if joint_values is None else joint_values
        q = self.validate_joint_values(q, check_limits=self.joint_limits is not None).copy()
        target_position = target if position_only else target[:3, 3]
        target_pose = None if position_only else target
        converged = False
        error_norm = np.inf
        iterations = 0
        for iterations in range(1, max_iter + 1):
            current = self.fk(q)
            error = target_position - current[:3, 3]
            jacobian = self.jacobian(q)[:3]
            if target_pose is not None:
                orientation_error = _pose_orientation_error(current, target_pose)
                error = np.concatenate(
                    [position_weight * error, orientation_weight * orientation_error]
                )
                full_jacobian = self.jacobian(q)
                full_jacobian = np.vstack(
                    [position_weight * full_jacobian[:3], orientation_weight * full_jacobian[3:]]
                )
            else:
                full_jacobian = position_weight * jacobian
            error_norm = float(np.linalg.norm(error))
            if error_norm <= tol:
                converged = True
                break
            gram = full_jacobian @ full_jacobian.T
            regularized = gram + damping**2 * np.eye(gram.shape[0], dtype=float)
            q += step_size * full_jacobian.T @ np.linalg.solve(regularized, error)
            if self.joint_limits is not None:
                q = self.clip_joint_values(q)

        if return_info:
            return RobotResult({
                "joint_values": q,
                "converged": converged,
                "iterations": iterations,
                "error_norm": error_norm,
                "position_only": position_only,
            })
        if not converged:
            raise RuntimeError(
                f"inverse kinematics did not converge within {max_iter} iterations "
                f"(error_norm={error_norm:.3e})"
            )
        return q


__all__ = ["URDFRobotModel"]
