import numpy as np

from mastermlx.robotics import RobotModel, RobotWorkcell, optimize_joint_path
from mastermlx.sim import SimpleWorld


def _planar_workcell():
    robot = RobotModel.from_dh(
        [
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta": 0.0},
        ],
        name="planar2r",
    )
    return RobotWorkcell(
        robot,
        SimpleWorld(robot),
        joint_limits=[[-1.5, 1.5], [-1.2, 1.2]],
    )


def _curvature_cost(path):
    difference = path[2:] - 2.0 * path[1:-1] + path[:-2]
    return float(np.sum(difference**2))


def test_optimize_joint_path_reduces_curvature_and_respects_bounds():
    path = np.array([
        [-1.0, -0.5],
        [-0.2, 1.0],
        [0.4, -0.9],
        [0.9, 0.8],
        [1.2, 0.2],
    ])
    result = optimize_joint_path(
        path,
        smoothness=1.0,
        reference_weight=0.1,
        joint_limits=[[-1.5, 1.5], [-1.0, 1.0]],
        max_iter=200,
        step_size=0.1,
    )

    optimized = result["path"]
    assert result["final_cost"] < result["initial_cost"]
    assert _curvature_cost(optimized) < _curvature_cost(path)
    assert np.array_equal(optimized[0], path[0])
    assert np.array_equal(optimized[-1], path[-1])
    assert np.all(optimized >= [-1.5, -1.0])
    assert np.all(optimized <= [1.5, 1.0])
    assert np.all(np.diff(result["history"]) <= 1e-12)


def test_workcell_optimizer_reports_clearance_and_plan_motion_integration():
    workcell = _planar_workcell()
    path = np.array([
        [-0.4, 0.2],
        [-0.1, 0.5],
        [0.2, 0.1],
        [0.5, -0.2],
    ])
    result = workcell.optimize_joint_path(
        path,
        bounds=[[-1.0, 1.0], [-1.0, 1.0]],
        smoothness=0.5,
        reference_weight=0.2,
        max_iter=80,
    )

    assert result["collision_free"]
    assert np.isinf(result["minimum_clearance"])
    assert np.allclose(result["path"][0], path[0])
    assert np.allclose(result["path"][-1], path[-1])

    motion = workcell.plan_motion(
        path[0],
        path[-1],
        bounds=[[-1.0, 1.0], [-1.0, 1.0]],
        velocity_limits=0.8,
        optimize_path=True,
        optimization_kwargs={"max_iter": 20},
    )
    assert motion["planning_report"]["optimized"]
    assert motion["optimization"]["collision_free"]
    assert motion["trajectory"]["position"].shape[1] == 2

    trajectory = workcell.retime_joint_path(path[:2], velocity_limits=0.8)
    tracking = workcell.simulate_tracking(
        trajectory,
        controller="mpc",
        control_limits=5.0,
    )
    assert tracking["controller"] == "mpc"
    assert tracking["actual"].shape == tracking["reference"].shape
    assert tracking["controls"].shape[1] == 2
    assert np.all(np.isfinite(tracking["joint_error"]))
