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

- default backend: `auto` (uses compiled kernels when available)
- explicit compatibility backend: `numpy`
- explicit compiled preference: `cython`
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
- `mastermlx.data.contract` contains explicit training/inference data contracts.
- `mastermlx.data.evaluation` contains OOF, cross-validation, uncertainty, and model comparison reports.
- `mastermlx.data.online` contains incremental, sliding-window, drift, and delayed-label workflows.

This keeps compatibility imports stable while making individual areas easier
to test, profile, and extend.

### 4. Lazy public facade

The broad compatibility namespace is now lazy: `import mastermlx` loads only
version/configuration metadata and a static export registry. A domain package
loads only when one of its public names is accessed. The registry is generated
from subpackage `__all__` values and guarded in CI, so import performance does
not trade away API compatibility.

### 5. Interview-ready technical story

The project is designed to support a strong narrative:

- pure NumPy implementations prove algorithm understanding
- optional Cython acceleration proves systems awareness
- modular architecture proves package design ability
- broad tests prove engineering discipline

## Current Technical Milestones

Completed foundations:

- wheel/sdist build, install smoke tests, artifact validation, and release
  publication are automated in one workflow;
- public examples have an executable, headless CI smoke suite;
- examples provide a task-oriented API guide and copy-and-run tutorials;
- import performance has a deterministic eager-load contract and timing budget.

Next priorities:

1. Define Stable, Beta, Experimental, and Internal API tiers.
2. Raise contract and branch coverage for Stable packages rather than adding
   unstructured test count.
3. Expand type annotations from `base`, validation, preprocessing, and data
   into the stable estimator families.
4. Track performance and parity baselines for existing hot paths before adding
   further compiled kernels.
5. Complete documented URDF semantics and workcell boundaries before expanding
   the robotics feature surface.

For the implementation-level plan, see [`docs/cython_roadmap.md`](cython_roadmap.md).
