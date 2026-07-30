# Business Use Cases

`mastermlx` is strongest when its from-scratch algorithms are packaged into
repeatable workflow reports.  The recommended business surfaces are small,
auditable, and NumPy-first.

## Industrial Signal Health

Use `SignalHealthExperiment` for sensor and vibration monitoring:

1. validate raw signal quality
2. extract interpretable vibration features
3. compare features against threshold limits
4. summarize status, health score, alerts, and windowed trends
5. export the report as JSON for dashboards or batch jobs

Good fits include vibration checks, motor health, sensor saturation detection,
and predictive-maintenance feature pipelines.

## Robot Workcell Planning

Use `RobotWorkcell.plan_motion()` for offline motion checks:

1. load or construct a serial robot and workcell
2. plan a collision-free joint path
3. retime under velocity, acceleration, and jerk limits
4. optionally simulate tracking
5. report clearance, motion limits, singularity, and tracking diagnostics

Good fits include offline programming demos, path feasibility checks, and
robotics interview or portfolio walkthroughs.

For material-handling cycles, use `RobotWorkcell.plan_pick_and_place()` with
world-frame approach and retreat offsets. It preserves Cartesian contact
motions, plans the loaded transfer through the workcell, retimes the full
cycle, and returns time-aligned open/close gripper events for an execution
adapter.

## Tabular Data Readiness

Use `DataReadinessReport` before training or inference:

1. summarize missing values, duplicates, outliers, and target quality
2. compare new data against a reference distribution
3. validate explicit `DataContract` rules when provided
4. return a ready/review status with concrete issue labels

Good fits include enterprise data intake, model handoff checklists, and
lightweight deployment risk reports.

## Workflow Benchmark Smoke Test

Run `python benchmarks/bench_workflows.py` to exercise the three business
surfaces together and print compact runtime/status summaries. Add
`--output outputs/workflows` to export one JSON report per workflow plus a
manifest.
