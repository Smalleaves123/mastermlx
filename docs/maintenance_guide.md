# Maintenance Guide

This guide keeps the project easier to extend as the algorithm and workflow
surface grows.

## Before Adding A Feature

1. Choose whether it is an algorithm primitive, workflow, report, or benchmark.
2. Reuse domain helpers and `BaseResult` before adding a new result shape.
3. Keep optional heavy dependencies behind extras.
4. Add a narrow unit test and, for workflow features, one business-facing test.

## Test Tiers

- Quick: unit tests and API compatibility checks.
- Full: the full `tests/` suite.
- Compiled: backend capability and NumPy/compiled parity checks.
- Benchmark smoke: short deterministic benchmark scripts.
- Release: sdist/wheel build and install smoke tests.

Use `python -m pytest` in local and CI commands so the selected interpreter
and the test runner cannot drift apart.

## Dependency Policy

Core dependencies should stay small. Visualization belongs in the `viz` extra,
comparison baselines belong in `compare`, and build/test tools belong in `dev`.

## Report Policy

Workflow reports should be JSON-safe or exportable through `BaseResult`.
Prefer flat summaries with nested details rather than ad hoc tuples of arrays.
