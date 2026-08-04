import numpy as np

from mastermlx.robotics import (
    MeshObstacle,
    SphereObstacle,
    URDFRobotModel,
    parse_urdf,
)


def _planar_urdf_with_mesh(filename):
    return f"""
    <robot name="mesh_arm">
      <link name="base" />
      <link name="link1">
        <collision>
          <origin xyz="0 0 0" rpy="0 0 0" />
          <geometry><mesh filename="{filename}" scale="1 1 1" /></geometry>
        </collision>
      </link>
      <link name="link2" />
      <link name="tip" />
      <joint name="joint1" type="revolute">
        <parent link="base" /><child link="link1" />
        <origin xyz="0 0 0" rpy="0 0 0" />
        <axis xyz="0 0 1" /><limit lower="-3.1415926535" upper="3.1415926535" />
      </joint>
      <joint name="joint2" type="revolute">
        <parent link="link1" /><child link="link2" />
        <origin xyz="1 0 0" rpy="0 0 0" />
        <axis xyz="0 0 1" /><limit lower="-3.1415926535" upper="3.1415926535" />
      </joint>
      <joint name="tip_fixed" type="fixed">
        <parent link="link2" /><child link="tip" />
        <origin xyz="1 0 0" rpy="0 0 0" />
      </joint>
    </robot>
    """


def _cube_obj():
    return """\
v -0.05 -0.05 -0.05
v 0.05 -0.05 -0.05
v 0.05 0.05 -0.05
v -0.05 0.05 -0.05
v -0.05 -0.05 0.05
v 0.05 -0.05 0.05
v 0.05 0.05 0.05
v -0.05 0.05 0.05
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 5 1 4 8
"""


def test_urdf_collision_geometry_and_obj_loader(tmp_path):
    mesh_path = tmp_path / "link.obj"
    mesh_path.write_text(_cube_obj(), encoding="utf-8")
    stl_path = tmp_path / "triangle.stl"
    stl_path.write_text(
        "solid triangle\n"
        " facet normal 0 0 1\n"
        "  outer loop\n"
        "   vertex 0 0 0\n"
        "   vertex 1 0 0\n"
        "   vertex 0 1 0\n"
        "  endloop\n"
        " endfacet\n"
        "endsolid triangle\n",
        encoding="utf-8",
    )
    xml = _planar_urdf_with_mesh(mesh_path.name)
    links, _ = parse_urdf(xml)
    assert len(links[1].collisions) == 1
    assert links[1].collisions[0].geometry_type == "mesh"

    robot = URDFRobotModel.from_urdf(xml, resource_dir=tmp_path)
    meshes = robot.collision_meshes([0.0, 0.0])
    assert len(meshes) == 1
    assert meshes[0].vertices.shape == (8, 3)
    assert isinstance(MeshObstacle.from_obj(mesh_path), MeshObstacle)
    assert MeshObstacle.from_stl(stl_path).faces.shape == (1, 3)

    report = robot.collision_report([0.0, 0.0], [SphereObstacle((0.0, 0.0, 0.0), 0.08)])
    assert report["collision"]
    assert any(hit["kind"] == "mesh" for hit in report["hits"])


def test_spatial_urdf_exposes_rrt_joint_planning():
    xml = _planar_urdf_with_mesh("unused.obj").replace(
        '''      <link name="link1">
        <collision>
          <origin xyz="0 0 0" rpy="0 0 0" />
          <geometry><mesh filename="unused.obj" scale="1 1 1" /></geometry>
        </collision>
      </link>''',
        '      <link name="link1" />',
    )
    robot = URDFRobotModel.from_urdf(xml)
    obstacles = [SphereObstacle((1.5, 0.0, 0.0), 0.15)]
    path = robot.plan_joint_path(
        [np.pi / 2.0, 0.0],
        [-np.pi / 2.0, 0.0],
        bounds=[[-np.pi, np.pi], [-np.pi, np.pi]],
        obstacles=obstacles,
        step=0.15,
        goal_rate=0.25,
        max_iter=3000,
        random_state=0,
        check_self_collision=True,
    )

    assert path.shape[0] > 2
    assert np.allclose(path[0], [np.pi / 2.0, 0.0])
    assert np.allclose(path[-1], [-np.pi / 2.0, 0.0])
    assert robot.path_collision_free(path, obstacles, interpolation_step=0.05)
