"""URDF parsing and the RobotExperiment convenience wrapper."""

import numpy as np

from mastermlx.robotics import (
    RobotExperiment,
    RobotModel,
    compare_robot_models,
    parse_urdf,
    urdf_to_dh_chain,
)


xml = """
<robot name="example_planar2r">
  <link name="base" />
  <link name="link1" />
  <link name="link2" />
  <joint name="joint1" type="revolute">
    <parent link="base" />
    <child link="link1" />
    <origin xyz="1 0 0" rpy="0 0 0" />
    <axis xyz="0 0 1" />
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1" />
    <child link="link2" />
    <origin xyz="1 0 0" rpy="0 0 0" />
    <axis xyz="0 0 1" />
  </joint>
</robot>
"""
links, joints = parse_urdf(xml)
dh_chain = urdf_to_dh_chain(xml)
print("urdf_counts:", len(links), len(joints), len(dh_chain))

robot = RobotModel.from_urdf(xml, name="urdf-robot")
experiment = RobotExperiment.from_urdf(xml, name="urdf-experiment")
q = np.array([0.2, -0.1])
print("model_pose:\n", robot.fk(q))
print("experiment_pose:\n", experiment.fk(q))
comparison = compare_robot_models(
    [("robot", robot), ("experiment", experiment.model)],
    q,
)
print("comparison:", comparison)

print("model_name:", robot.name)
print("comparison_entries:", len(comparison["leaderboard"]))
