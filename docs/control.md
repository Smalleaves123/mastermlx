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
