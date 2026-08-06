# Examples

`examples/` is the single canonical example directory. The former top-level
`example/` directory has been merged here so tutorials no longer have two
different layouts or import conventions.

## Install and run

From the repository root:

```bash
python -m pip install -e ".[viz]"
python examples/robotics/simulation_loop_demo.py
```

Plotting examples create PNG files under `examples/outputs/`. The output
directory is generated on demand and is safe to remove before another run.

## Directory guide

- `robotics/` contains kinematics, planning, workcell, state estimation, and
  task-level simulation examples.
- `signal/` contains Fourier, streaming, monitoring, and health workflows.
- `classification/`, `clustering/`, `neural_networks/`, and `nlp/` contain
  model demonstrations with visual diagnostics.
- `quickstart/` contains short cross-domain tutorials.
- `tabular/`, `tools/`, `rl/`, `bandits/`, and `probabilistic/` contain focused
  API examples and extension points.

## Recommended robotics path

Run these in order when learning the simulation stack:

1. `robotics/00_quickstart.py` — construct a two-link robot and evaluate FK.
2. `robotics/kinematics_demo.py` — Jacobians, batches, and IK.
3. `robotics/trajectory_demo.py` — generate and plot joint trajectories.
4. `robotics/workcell_planning_demo.py` — plan around an obstacle and plot the
   reference versus tracked TCP path.
5. `robotics/simulation_loop_demo.py` — execute a controller through
   `RobotSimulation` and inspect termination metrics.
6. `robotics/pick_place_simulation.py` — attach, transport, and release an
   object while plotting TCP and object traces.
7. `robotics/batch_rollout_demo.py` — compare independent and batched rollouts.

The standard environment contract and its intentional lightweight physics
boundary are documented in [`docs/robotics_simulation.md`](../docs/robotics_simulation.md).
