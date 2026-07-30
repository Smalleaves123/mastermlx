# Workflow Interfaces

Business workflows should expose dictionary-compatible reports while sharing a
small maintenance contract.

## Result Objects

Use `BaseResult` or `BaseReport` for public workflow outputs. They preserve
normal mapping access while adding:

- attribute access such as `report.status`
- `as_dict(json_safe=True)` for NumPy-safe serialization
- `to_json(path)` and compact `to_csv(path)` exports

Domain aliases such as `RobotResult` should remain available when they are
already public.

## Experiment Objects

Use `BaseExperiment` for high-level workflows that produce reports or
artifacts. The shared fields are:

- `report_` for the latest report
- `artifacts_` for exported or generated assets
- `export_report(path)` for JSON output

Existing `fit`, `run`, `predict`, and `score` semantics should remain
domain-specific. The base class is intentionally small so workflows do not
inherit unnecessary behavior.

## Current Business Workflows

- `mastermlx.tabular.DataReadinessReport`
- `mastermlx.tabular.TabularExperiment`
- `mastermlx.signal.SignalHealthExperiment`
- `mastermlx.signal.SignalExperiment`
- `mastermlx.robotics.RobotWorkcell`
