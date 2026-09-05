# Maintenance Guide

This guide keeps the project easier to extend as the algorithm and workflow
surface grows.

## Before Adding A Feature

1. Choose whether it is an algorithm primitive, workflow, report, or benchmark.
2. Reuse domain helpers and `BaseResult` before adding a new result shape.
3. Keep optional heavy dependencies behind extras.
4. Add a narrow unit test and, for workflow features, one business-facing test.
5. Classify every new public name in `mastermlx.api` before release.

## API Stability

Run the API contract test after changing public exports:

```bash
python -m pytest tests/test_api_compat.py tests/test_api_stability_bundle.py -q
```

Package defaults and explicit overrides are kept in `mastermlx/api.py`.
Stable names require compatibility tests and a documented deprecation path.

## Test Tiers

- Quick: unit tests and API compatibility checks.
- Full: the full `tests/` suite.
- Compiled: backend capability and NumPy/compiled parity checks.
- Benchmark smoke: short deterministic benchmark scripts.
- Release: sdist/wheel build and install smoke tests.

Use `python -m pytest` in local and CI commands so the selected interpreter
and the test runner cannot drift apart.

## Top-Level Import Budget

`import mastermlx` is a lazy facade. It may load version metadata, backend
configuration, and the static export registry, but it must not import NumPy or
domain packages until a public symbol is accessed.

Run the cold-import guard locally with:

```bash
python scripts/check_import_budget.py --budget-ms 250 --runs 5
```

The CI budget uses the median of fresh interpreter runs. Deterministic tests
also assert the exact initial `mastermlx.*` module set, so a regression cannot
hide behind a generous timing threshold. Add new top-level exports to
the owning package's `__all__`, then regenerate and validate the static
registry:

```bash
python scripts/generate_lazy_exports.py
python scripts/generate_lazy_exports.py --check
```

CI runs the check mode, and `tests/test_api_compat.py` compares the combined
registry against every public package's `__all__`.

## Examples Smoke Suite

Public examples are executable documentation. Run the curated deterministic
subset from the repository root before release:

```bash
MPLBACKEND=Agg python scripts/run_examples_smoke.py
```

The suite covers quickstart, regression, probabilistic models, bandits, RL,
tabular readiness, classification visualization, and signal visualization. It
uses a headless Matplotlib backend and verifies that the plotting examples
create their expected files. Add a new example to this suite only when it is
fast, deterministic, and representative of a supported public workflow.

## Dependency Policy

Core dependencies should stay small. Visualization belongs in the `viz` extra,
comparison baselines belong in `compare`, and build/test tools belong in `dev`.

## Report Policy

Workflow reports should be JSON-safe or exportable through `BaseResult`.
Prefer flat summaries with nested details rather than ad hoc tuples of arrays.
