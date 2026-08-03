# Spatial URDF robotics

`URDFRobotModel` provides a general serial-chain kinematics path for spatial
robots. It preserves URDF joint origins and arbitrary joint axes instead of
reducing the model to the legacy axis-aligned DH conversion.

```python
from mastermlx.robotics import URDFRobotModel

robot = URDFRobotModel.from_urdf(
    xml_text,
    base_link="base_link",
    tip_link="tool0",
)

q = robot.default_joint_values()
pose = robot.fk(q)
jacobian = robot.jacobian(q)
q_target = robot.inverse_kinematics(target_pose, joint_values=q)
```

Supported serial-chain joints are:

- `fixed`
- `revolute`
- `continuous`
- `prismatic`

The model supports non-zero joint-origin RPY, arbitrary joint axes, batches of
FK/Jacobian evaluations, differential IK, position IK, and full-pose IK. It is
currently a kinematics-focused model. The existing `RobotModel` remains the
compatibility path for DH-based dynamics and workcell workflows.

## Spatial collision and occupancy maps

The same spatial chain can be checked against 3D analytic obstacles or a
voxelized point cloud. `VoxelOccupancyGrid` is NumPy-only and uses `(x, y, z)`
voxel indices:

```python
from mastermlx.robotics import VoxelOccupancyGrid

occupancy = VoxelOccupancyGrid.from_point_cloud(
    point_cloud,
    bounds=([-1, -1, -0.2], [1, 1, 1.5]),
    resolution=0.05,
)
safe = robot.path_collision_free(
    joint_path,
    obstacles,
    occupancy_grid=occupancy,
    interpolation_step=0.05,
)
report = robot.path_collision_summary(
    joint_path, obstacles, occupancy_grid=occupancy
)
```

The map supports world/grid coordinate conversion, marking and clearing
occupied voxels, point queries, conservative radius checks, and polyline
checks. Points outside the map are treated as free for collision queries and
ignored when voxelizing a point cloud.

## URDF collision geometry and sampling planners

URDF links may contain collision boxes, spheres, cylinders, or OBJ/STL meshes.
Pass `resource_dir` when mesh filenames are relative:

```python
robot = URDFRobotModel.from_urdf(
    xml_text,
    resource_dir="path/to/meshes",
)
meshes = robot.collision_meshes(q)
report = robot.collision_report(q, obstacles)
```

Spatial models also expose the existing NumPy RRT and RRT* planners:

```python
path = robot.plan_joint_path(
    q_start,
    q_goal,
    bounds=joint_bounds,
    planner="rrt_star",
    obstacles=obstacles,
    random_state=0,
)
```

The planner checks every interpolated edge against analytic obstacles and the
optional occupancy grid, then performs a final safety check before returning.

## Design boundary

The current implementation intentionally rejects branching chains and URDF
`floating`, `planar`, and `spherical` joints. The spatial model remains
kinematics-focused: it does not yet parse URDF visual/collision meshes or
provide dynamics for arbitrary spatial chains.

## Joint-path optimization

For an existing collision-free joint path, the robotics package also provides
`optimize_joint_path()`. It keeps the endpoints fixed, reduces path curvature,
projects intermediate waypoints into joint limits, and accepts an optional
cost for the complete path:

```python
from mastermlx.robotics import optimize_joint_path

result = optimize_joint_path(
    path,
    smoothness=1.0,
    reference_weight=0.2,
    joint_limits=limits,
)
optimized_path = result["path"]
```

`RobotWorkcell.optimize_joint_path()` supplies a clearance barrier based on
the workcell obstacles and returns `collision_free`, `minimum_clearance`, and
the detailed collision summary. `RobotWorkcell.plan_motion()` can run this
step with `optimize_path=True`; it refuses to execute an optimized path that
does not pass the final collision check.

## Closed-loop tracking

`RobotWorkcell.simulate_tracking()` keeps PD tracking as the default and can
run the same virtual second-order joint simulator with the existing linear MPC
controller:

```python
tracking = workcell.simulate_tracking(
    trajectory,
    controller="mpc",
    control_limits=5.0,
)
```

The result contains the controller name, simulated states, controls, poses,
and per-step joint error. MPC control limits are expanded across the complete
prediction horizon before solving the box-constrained quadratic problem.
