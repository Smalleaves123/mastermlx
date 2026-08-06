# Robot Simulation

`mastermlx.sim` provides a dependency-free simulation layer for controller,
planning, and reinforcement-learning experiments. It intentionally keeps the
core deterministic and NumPy-first; external physics engines can be added as
adapters without changing the robot model or workcell APIs.

## Standard environment loop

`RobotSimulation` follows a small Gym-like contract without requiring Gym:

The runnable examples are:

- `examples/robotics/simulation_loop_demo.py` for target tracking
- `examples/robotics/pick_place_simulation.py` for attachment state
- `examples/robotics/batch_rollout_demo.py` for vectorized dynamics
- `examples/robotics/inspection_simulation.py` for coverage and scan timing

```python
from mastermlx.sim import RobotSimulation, SimpleWorld

world = SimpleWorld(robot)
world.add_object("part", [2.0, 0.0, 0.0])
env = RobotSimulation(world, max_steps=500)

observation, info = env.reset(seed=0)
for _ in range(500):
    observation, reward, terminated, truncated, info = env.step(
        {"joint": [0.0, 0.0], "gripper": 1.0}
    )
    if terminated or truncated:
        break
```

The observation mapping contains joint state, TCP pose, gripper value, object
positions, and attachment flags. Joint actions are accelerations. A mapping
may include a scalar `gripper` command; values at or below `0.5` close the
gripper and values above `0.5` open it.

## Movable objects

`SimpleWorld.add_object()` creates a small task-level object. Closing near a
graspable object attaches it to the TCP, and opening releases it. The object
state is intentionally lightweight: it models attachment and TCP-relative
motion, not rigid-body contact dynamics.

## Batched dynamics

`SimpleRobotSim.rollout_batch()` and `step_state_batch()` evaluate independent
second-order episodes with NumPy batch operations. Run
`python benchmarks/bench_sim.py` to measure the local batch speedup and parity
against the existing single-episode rollout.
