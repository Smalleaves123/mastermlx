import numpy as np

from mastermlx.robotics import (
    BoxObstacle,
    CapsuleObstacle,
    SphereObstacle,
    URDFRobotModel,
    VoxelOccupancyGrid,
    path_collision_free,
    path_collision_summary,
)


def _prismatic_robot():
    xml = """
    <robot name="voxel_arm">
      <link name="base" />
      <link name="tip" />
      <joint name="slide" type="prismatic">
        <parent link="base" /><child link="tip" />
        <origin xyz="0 0 0" rpy="0 0 0" />
        <axis xyz="1 0 0" />
        <limit lower="0" upper="1" />
      </joint>
    </robot>
    """
    return URDFRobotModel.from_urdf(xml)


def test_voxel_grid_point_cloud_queries_and_clear():
    grid = VoxelOccupancyGrid.from_point_cloud(
        np.array([[0.24, 0.24, 0.24], [2.0, 2.0, 2.0]]),
        bounds=([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        resolution=0.25,
    )

    assert grid.shape == (4, 4, 4)
    assert np.array_equal(grid.world_to_grid([0.24, 0.24, 0.24]), [0, 0, 0])
    assert np.allclose(grid.grid_to_world([0, 0, 0]), [0.125, 0.125, 0.125])
    assert grid.is_occupied([0.24, 0.24, 0.24])
    assert np.array_equal(
        grid.is_occupied([[0.24, 0.24, 0.24], [0.8, 0.8, 0.8]]), [True, False]
    )
    assert not grid.is_occupied([1.2, 0.5, 0.5])
    assert not grid.collision_free([0.24, 0.24, 0.24])
    grid.clear_occupied([0.24, 0.24, 0.24])
    assert grid.collision_free([0.24, 0.24, 0.24])


def test_voxel_grid_radius_and_polyline_collision():
    grid = VoxelOccupancyGrid.from_point_cloud(
        [[0.5, 0.5, 0.5]],
        bounds=([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        resolution=0.1,
    )

    assert grid.collision_free([[0.1, 0.1, 0.1]])
    assert not grid.collision_free([[0.5, 0.5, 0.35]], radius=0.1)
    assert not grid.polyline_collision_free([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]])
    assert grid.minimum_clearance([[0.5, 0.5, 0.5]]) <= 1e-12


def test_spatial_robot_geometry_and_voxel_path_check():
    robot = _prismatic_robot()
    obstacles = [
        SphereObstacle((0.5, 0.0, 0.0), 0.05),
        BoxObstacle((0.7, -0.05, -0.05), (0.8, 0.05, 0.05)),
        CapsuleObstacle((0.2, 0.2, 0.0), (0.8, 0.2, 0.0), 0.03),
    ]
    report = robot.collision_report([0.5], obstacles)
    assert report["collision"]
    assert report["minimum_clearance"] < 0.0

    grid = VoxelOccupancyGrid.from_point_cloud(
        [[0.5, 0.0, 0.0]],
        bounds=([-0.1, -0.2, -0.2], [1.1, 0.2, 0.2]),
        resolution=0.05,
    )
    path = np.array([[0.1], [0.9]])
    assert not path_collision_free(robot, path, [], occupancy_grid=grid, interpolation_step=0.1)
    summary = path_collision_summary(
        robot, path, [], occupancy_grid=grid, interpolation_step=0.1
    )
    assert summary["collision"]
    assert summary["first_collision_index"] is not None
    assert "occupancy_clearances" in summary
    assert not robot.path_collision_free(path, occupancy_grid=grid, interpolation_step=0.1)
