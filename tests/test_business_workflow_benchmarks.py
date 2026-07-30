from benchmarks.bench_workflows import (
    benchmark_robot_workcell,
    benchmark_signal_health,
    benchmark_tabular_readiness,
)


def test_business_workflow_benchmarks_return_reports():
    tabular = benchmark_tabular_readiness()
    signal = benchmark_signal_health()
    robot = benchmark_robot_workcell()

    assert tabular.status == "review"
    assert signal.summary["status"] == "healthy"
    assert robot.planning_report["n_waypoints"] >= 2
