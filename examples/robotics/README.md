# Robotics Examples

This folder contains robotics demos for the `mastermlx.robotics` package.

## Intended demos

- forward and inverse kinematics on a small serial arm
- joint-space trajectory planning and tracking
- URDF-based model construction
- spatial URDF kinematics with arbitrary joint axes and full-pose IK
- spatial collision checks against analytic obstacles and point-cloud voxels
- URDF collision geometry and spatial RRT/RRT* planning
- planar pose estimation with `PlanarPoseEKF`
- the higher-level `RobotExperiment` workflow

## Good example stories

- a minimal serial-arm planning loop
- a trajectory report that summarizes joint motion and end-effector travel
- a small pose-estimation update loop for planar navigation
- a workcell motion-planning loop with obstacle clearance, retiming, tracking, and safety reports

## Demo link

The high-level workflow demo lives in [`experiment_demo.py`](experiment_demo.py).
The workcell planning workflow lives in [`workcell_planning_demo.py`](workcell_planning_demo.py).
The general spatial URDF API is documented in
[`docs/robotics_3d.md`](../../docs/robotics_3d.md).
