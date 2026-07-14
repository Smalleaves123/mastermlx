# mastermlx Architecture Plan

This document captures the design ideas that differentiate `mastermlx` from a basic from-scratch ML reimplementation.

## Design Goals

- Keep a pure-Python/NumPy path that is easy to read in interviews
- Add an optional acceleration layer that can switch to Cython without changing public APIs
- Separate algorithm code from compute kernels so the project can evolve into a real package
- Make each subsystem usable on its own: `clustering`, `probabilistic`, `signal`, `nlp`, `neural_networks`

## Distinctive Ideas

### 1. Dual-backend execution

`mastermlx` now has a compute backend layer:

- default backend: `numpy`
- optional backend: `cython`
- runtime switch via `mastermlx.set_backend("numpy" | "cython" | "auto")`

This is meant to show engineering thinking beyond algorithm reproduction.

### 2. Layered package design

The codebase is organized into:

- public APIs in each domain package
- shared contracts in `mastermlx.base`
- shared validation and metrics in `mastermlx.utils`
- optional acceleration in `mastermlx.accel`

This makes the library easier to publish, test, benchmark, and explain.

### 3. Implementation modules and export facades

Domain package `__init__.py` files should stay small and expose stable public
names. Workflow and reporting logic belongs in focused implementation modules:

- `mastermlx.tabular.workflow` contains the tabular experiment implementation.
- `mastermlx.data.quality` contains row/column quality summaries.
- `mastermlx.data.schema` contains train/test schema checks.
- `mastermlx.data.drift` contains distribution drift checks.

This keeps compatibility imports stable while making individual areas easier
to test, profile, and extend.

### 3. Interview-ready technical story

The project is designed to support a strong narrative:

- pure NumPy implementations prove algorithm understanding
- optional Cython acceleration proves systems awareness
- modular architecture proves package design ability
- broad tests prove engineering discipline

## Next Technical Milestones

1. Move more heavy kernels into `mastermlx.accel`
2. Add benchmark reports that compare NumPy vs Cython backends
3. Expand compiled coverage in `mastermlx.control`, `mastermlx.robotics`, and `mastermlx.math_tools`
4. Add packaging automation for wheel/sdist validation
5. Add API docs and stable versioning policy

For the implementation-level plan, see [`docs/cython_roadmap.md`](cython_roadmap.md).
