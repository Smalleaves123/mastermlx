# Robotics Examples

This folder contains copy-and-run robotics and simulation demos for
`mastermlx.robotics` and `mastermlx.sim`.

## Intended demos

- forward and inverse kinematics on a small serial arm
- joint-space trajectory planning and tracking
- URDF-based model construction
- spatial URDF kinematics with arbitrary joint axes and full-pose IK
- spatial collision checks against analytic obstacles and point-cloud voxels
- URDF collision geometry and spatial RRT/RRT* planning
- planar pose estimation with `PlanarPoseEKF`
- the higher-level `RobotExperiment` workflow
- the standard `RobotSimulation` environment loop
- lightweight object grasp, transport, and release
- batch rollout speed and parity comparison

## Good example stories

- a minimal serial-arm planning loop
- a trajectory report that summarizes joint motion and end-effector travel
- a small pose-estimation update loop for planar navigation
- a workcell motion-planning loop with obstacle clearance, retiming, tracking, and safety reports
- a closed-loop controller moving a TCP to a target inside a simulated world
- a pick-and-place task with object attachment state

## Demo link

## Run order

```bash
python examples/robotics/00_quickstart.py
python examples/robotics/workcell_planning_demo.py
python examples/robotics/simulation_loop_demo.py
python examples/robotics/pick_place_simulation.py
python examples/robotics/batch_rollout_demo.py
```

The high-level workflow demo lives in [`experiment_demo.py`](experiment_demo.py).
The workcell planning workflow lives in [`workcell_planning_demo.py`](workcell_planning_demo.py).
The standard simulation loop lives in [`simulation_loop_demo.py`](simulation_loop_demo.py).
Plots are written to `examples/outputs/robotics/`.
The general spatial URDF API is documented in
[`docs/robotics_3d.md`](../../docs/robotics_3d.md).
The simulation contract is documented in
[`docs/robotics_simulation.md`](../../docs/robotics_simulation.md).
