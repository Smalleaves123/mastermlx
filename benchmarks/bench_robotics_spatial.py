"""Benchmark compiled kinematics and dynamics for general serial URDF chains."""

from __future__ import annotations

import time

import numpy as np

from mastermlx import get_backend, set_backend
from mastermlx.robotics import URDFRobotModel
from mastermlx.robotics.urdf_parser import _load_cpp_spatial


def _robot():
    links = ['<link name="base" />']
    for index in range(1, 8):
        links.append(
            f'''<link name="link{index}">
              <inertial>
                <origin xyz="-0.12 0.02 0.01" rpy="0.03 0.02 0.01" />
                <mass value="{1.0 + 0.1 * index}" />
                <inertia ixx="0.02" ixy="0.001" ixz="0.0005" iyy="0.025" iyz="0.0008" izz="0.03" />
              </inertial>
            </link>'''
        )
    joints = []
    joint_types = ("revolute", "fixed", "prismatic", "revolute", "continuous", "revolute", "prismatic")
    axes = ((1, 0, 1), (0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 0), (0, 0, 1), (1, 1, 0))
    for index, (joint_type, axis) in enumerate(zip(joint_types, axes), start=1):
        parent = "base" if index == 1 else f"link{index - 1}"
        axis_xml = "" if joint_type == "fixed" else f'<axis xyz="{axis[0]} {axis[1]} {axis[2]}" />'
        joints.append(
            f'''<joint name="joint{index}" type="{joint_type}">
              <parent link="{parent}" /><child link="link{index}" />
              <origin xyz="0.22 0.03 0.04" rpy="0.05 -0.03 0.02" />
              {axis_xml}
            </joint>'''
        )
    return URDFRobotModel.from_urdf(
        f'<robot name="spatial-benchmark">{"".join(links)}{"".join(joints)}</robot>'
    )


def _measure(function, repeats=3):
    function()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        timings.append(time.perf_counter() - start)
    return float(np.mean(timings))


def main():
    if _load_cpp_spatial("auto") is None:
        raise SystemExit("C++ spatial extension is unavailable; build with `python setup.py build_ext --inplace`")
    robot = _robot()
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 0.3, size=(2_000, robot.n_joints))
    qd = np.full(robot.n_joints, 0.15)
    qdd = np.full(robot.n_joints, 0.1)
    torques = np.full(robot.n_joints, 0.2)
    batch_values = values[:256]
    batch_velocities = np.full_like(batch_values, 0.15)
    batch_accelerations = np.full_like(batch_values, 0.1)
    old = get_backend()
    try:
        print(f"C++ spatial backend available: {_load_cpp_spatial('auto') is not None}")
        results = {}
        for backend in ("numpy", "auto"):
            set_backend(backend)
            results[backend] = {
                "fk_batch": _measure(lambda: robot.fk_batch(values)),
                "frame_positions_batch": _measure(lambda: robot.frame_positions_batch(values)),
                "jacobian_batch": _measure(lambda: robot.jacobian_batch(values)),
                "inverse_dynamics_batch_256": _measure(
                    lambda: robot.inverse_dynamics_batch(
                        batch_values,
                        batch_velocities,
                        batch_accelerations,
                    ),
                    repeats=3,
                ),
                "dynamics_64": _measure(
                    lambda: [
                        (
                            robot.mass_matrix(row),
                            robot.gravity_forces(row),
                            robot.coriolis_forces(row, qd),
                            robot.inverse_dynamics(row, qd, qdd),
                            robot.forward_dynamics(row, qd, torques),
                        )
                        for row in values[:64]
                    ],
                    repeats=1,
                ),
            }
            print(f"{backend:>5} {results[backend]}")
        print("speedups")
        for name in results["numpy"]:
            print(f"  {name}: {results['numpy'][name] / results['auto'][name]:.2f}x")
    finally:
        set_backend(old)


if __name__ == "__main__":
    main()
