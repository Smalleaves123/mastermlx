import numpy as np

from mastermlx import get_backend, set_backend
from mastermlx.robotics import URDFRobotModel, URDFSerialChain
from mastermlx.robotics.urdf_parser import _load_cpp_spatial


def _spatial_urdf(joint_order="normal"):
    joints = {
        "joint1": """
          <joint name="joint1" type="revolute">
            <parent link="base" /><child link="link1" />
            <origin xyz="0 0 0.4" rpy="0 0 0" />
            <axis xyz="0 0 1" /><limit lower="-1.2" upper="1.2" />
          </joint>
        """,
        "joint2": """
          <joint name="joint2" type="revolute">
            <parent link="link1" /><child link="link2" />
            <origin xyz="0 0 0" rpy="0 0.4 0" />
            <axis xyz="0 1 1" /><limit lower="-1.0" upper="1.0" />
          </joint>
        """,
        "joint3": """
          <joint name="joint3" type="fixed">
            <parent link="link2" /><child link="link3" />
            <origin xyz="0.35 0 0.1" rpy="0.2 0 0" />
          </joint>
        """,
        "joint4": """
          <joint name="joint4" type="prismatic">
            <parent link="link3" /><child link="tool" />
            <origin xyz="0 0 0.3" rpy="0 0 0.2" />
            <axis xyz="1 0 0" /><limit lower="0.0" upper="0.4" />
          </joint>
        """,
    }
    order = ("joint2", "joint1", "joint3", "joint4") if joint_order == "shuffled" else tuple(joints)
    return """
    <robot name="spatial_arm">
      <link name="base" /><link name="link1" /><link name="link2" />
      <link name="link3" /><link name="tool" />
      {joints}
    </robot>
    """.format(joints="\n".join(joints[name] for name in order))


def test_spatial_urdf_preserves_joint_axes_origins_and_path_order():
    chain = URDFSerialChain.from_urdf(_spatial_urdf("shuffled"))
    assert chain.base_link == "base"
    assert chain.tip_link == "tool"
    assert chain.joint_names == ("joint1", "joint2", "joint4")
    assert chain.joint_types == ("revolute", "revolute", "prismatic")
    assert chain.n_joints == 3
    assert np.allclose(chain.joint_limits, [[-1.2, 1.2], [-1.0, 1.0], [0.0, 0.4]])


def test_cpp_spatial_urdf_batch_kinematics_matches_numpy():
    if _load_cpp_spatial("auto") is None:
        return
    chain = URDFSerialChain.from_urdf(_spatial_urdf("shuffled"))
    values = np.array([[0.2, -0.3, 0.1], [-0.4, 0.5, 0.2]])
    base = np.array(
        [[0.0, -1.0, 0.0, 0.2], [1.0, 0.0, 0.0, -0.1], [0.0, 0.0, 1.0, 0.3], [0.0, 0.0, 0.0, 1.0]]
    )
    tool = np.eye(4)
    tool[:3, 3] = [0.1, 0.0, 0.2]
    old = get_backend()
    try:
        set_backend("numpy")
        reference = (
            chain.forward_kinematics_batch(values, base=base, tool=tool),
            chain.positions_batch(values, base=base, tool=tool),
            chain.geometric_jacobian_batch(values, base=base, tool=tool),
        )
        set_backend("auto")
        accelerated = (
            chain.forward_kinematics_batch(values, base=base, tool=tool),
            chain.positions_batch(values, base=base, tool=tool),
            chain.geometric_jacobian_batch(values, base=base, tool=tool),
        )
    finally:
        set_backend(old)

    for actual, expected in zip(accelerated, reference):
        assert np.allclose(actual, expected, atol=1e-12)


def test_spatial_urdf_jacobian_matches_position_finite_difference():
    robot = URDFRobotModel.from_urdf(_spatial_urdf(), name="spatial")
    q = np.array([0.25, -0.3, 0.12])
    pose, frames = robot.fk(q, return_all=True)
    jacobian = robot.jacobian(q)
    eps = 1e-6
    numerical = np.zeros((3, robot.n_joints))
    for index in range(robot.n_joints):
        delta = np.zeros(robot.n_joints)
        delta[index] = eps
        numerical[:, index] = (robot.fk(q + delta)[:3, 3] - robot.fk(q - delta)[:3, 3]) / (2 * eps)

    assert pose.shape == (4, 4)
    assert len(frames) == 5
    assert jacobian.shape == (6, 3)
    assert np.allclose(jacobian[:3], numerical, atol=2e-6)


def test_spatial_urdf_batch_and_position_ik():
    robot = URDFRobotModel.from_urdf(_spatial_urdf())
    q = np.array([0.35, -0.2, 0.18])
    target = robot.fk(q)[:3, 3]
    result = robot.inverse_kinematics(
        target,
        joint_values=[0.2, -0.1, 0.1],
        max_iter=300,
        tol=1e-8,
        return_info=True,
    )
    batch = robot.fk_batch(np.vstack([q, result.joint_values]))
    jacobian_batch = robot.jacobian_batch(np.vstack([q, result.joint_values]))

    assert result.converged
    assert result.error_norm < 1e-8
    assert np.allclose(batch[1], robot.fk(result.joint_values))
    assert jacobian_batch.shape == (2, 6, 3)
    assert np.allclose(batch[1, :3, 3], target, atol=1e-7)

    clipped_seed = robot.inverse_kinematics(
        target,
        joint_values=[5.0, -5.0, 5.0],
        max_iter=300,
        return_info=True,
    )
    assert np.all(clipped_seed.joint_values >= robot.joint_limits[:, 0])
    assert np.all(clipped_seed.joint_values <= robot.joint_limits[:, 1])


def test_spatial_urdf_full_pose_ik_and_joint_clipping():
    xml = _spatial_urdf().replace(
        '<joint name="joint3" type="fixed">',
        '<joint name="joint3" type="revolute">\n'
        '            <axis xyz="0 0 1" /><limit lower="-1.0" upper="1.0" />'
    ).replace(
        '<axis xyz="1 0 0" /><limit lower="0.0" upper="0.4" />',
        '<axis xyz="1 0 0" /><limit lower="-1.0" upper="1.0" />',
    )
    robot = URDFRobotModel.from_urdf(xml)
    q = np.array([0.18, -0.15, 0.2, 0.12])
    target = robot.fk(q)
    result = robot.inverse_kinematics(
        target,
        joint_values=q + np.array([0.02, -0.02, 0.01, -0.01]),
        max_iter=300,
        tol=1e-8,
        return_info=True,
    )

    assert result.converged
    assert result.error_norm < 1e-8
    assert np.allclose(robot.fk(result.joint_values), target, atol=1e-6)
    assert np.allclose(robot.clip_joint_values([3.0, -2.0, 2.0, 2.0]), [1.2, -1.0, 1.0, 1.0])
