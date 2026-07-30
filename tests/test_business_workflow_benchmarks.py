from benchmarks.bench_workflows import (
    benchmark_robot_workcell,
    benchmark_signal_health,
    benchmark_tabular_readiness,
    run_workflow_suite,
)


def test_business_workflow_benchmarks_return_reports():
    _, tabular = benchmark_tabular_readiness(verbose=False)
    _, signal = benchmark_signal_health(verbose=False)
    _, robot = benchmark_robot_workcell(verbose=False)

    assert tabular.status == "review"
    assert signal.summary["status"] == "healthy"
    assert robot.planning_report["n_waypoints"] >= 2


def test_business_workflow_suite_exports_reports(tmp_path):
    result = run_workflow_suite(output_dir=tmp_path, verbose=False)

    assert result.reports.tabular_readiness.status == "review"
    assert result.timings.signal_health >= 0.0
    assert result.artifacts.manifest.is_file()
    assert result.artifacts.tabular_readiness.is_file()
    assert result.artifacts.signal_health.is_file()
    assert result.artifacts.robot_workcell.is_file()
