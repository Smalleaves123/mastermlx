# Control module

`mastermlx.control` keeps Python APIs and NumPy fallbacks while using compiled
kernels for workloads that do not require Python callbacks.

## Linear rollout

```python
from mastermlx.control import rollout_linear_dynamics

states = rollout_linear_dynamics(A, B, x0, controls)
```

This path uses `mastermlx.control._control_cpp` when the backend is `auto` and
falls back to the same recurrence in NumPy when the extension is unavailable.
Use `control_backend_report()` to inspect the active capability.

## Linear MPC constraints

`LinearMPC(..., u_bounds=(lower, upper))` solves the finite-horizon condensed
quadratic problem with projected-gradient iterations. Bounds can be scalars or
vectors with one value per control channel. A state reference can be a single
vector or a full `(horizon + 1, state_dim)` trajectory.

The controller exposes `qp_converged_` and `last_qp_iterations_` for runtime
diagnostics. Call `reset()` when starting a new episode to clear the warm-start
sequence.

The unconstrained case continues to use finite-horizon LQR feedback. Clipping
is not used as a substitute for constrained optimization when `u_bounds` are
provided.

## Hardware-neutral joint controllers

`Controller` defines the small execution contract used by offline workflows:

```python
from mastermlx.control import JointPDController

controller = JointPDController(
    n_joints=2,
    kp=2.0,
    kd=0.1,
    output_limits=(-1.0, 1.0),
)
controller.reset()
command = controller.update(reference, state, dt=0.05)
status = controller.status()
```

`reference` is a joint-position vector and `state` is the concatenation of
joint positions and velocities. The status mapping contains step count,
command saturation, and the last command, so an application can decide how to
handle faults or limits without depending on a transport layer.

`JointMPCController` provides the same interface and reports QP convergence
diagnostics. `ComputedTorqueController` produces torque commands using the
robot's DH or URDF dynamics. These classes have no ROS2, message-bus, or
hardware-driver dependency; a device adapter can consume the returned NumPy
command while NumPy remains the fallback implementation for the library's
accelerated paths.

`RobotWorkcell.simulate_tracking()` accepts the legacy strings `"pd"`,
`"mpc"`, and `"computed_torque"`, or an object implementing the controller
contract. Tracking results include `controller_status`. Computed-torque
commands are passed through the robot's forward dynamics in the simulator;
PD and MPC commands use the virtual second-order acceleration model.
