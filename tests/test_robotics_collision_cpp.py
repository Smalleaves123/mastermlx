import numpy as np
import pytest

from mastermlx import get_backend, set_backend
from mastermlx.robotics import (
    BoxObstacle,
    CapsuleObstacle,
    SphereObstacle,
    chain_clearance_batch,
    chain_collision_free_batch,
    chain_collision_report,
    robotics_backend_report,
)
from mastermlx.robotics.collision import _load_cpp_collision


def _obstacles():
    return [
        SphereObstacle((0.8, 0.1), 0.12),
        BoxObstacle((1.1, -0.2), (1.4, 0.15)),
        CapsuleObstacle((0.0, 0.5), (1.8, 0.5), 0.08),
    ]


def test_robotics_backend_report_includes_collision_kernel():
    report = robotics_backend_report()
    assert "cpp_collision" in report
    assert isinstance(report["cpp_collision"], bool)


def test_cpp_batch_clearance_matches_single_chain_reports():
    cpp = _load_cpp_collision("auto")
    if cpp is None:
        pytest.skip("C++ collision extension is unavailable")

    points = np.array(
        [
            [[0.0, 0.0], [0.8, 0.0], [1.6, 0.0]],
            [[0.0, 0.0], [0.4, 0.4], [1.2, 0.8]],
            [[0.0, 0.0], [0.2, -0.4], [0.9, -0.8]],
        ],
        dtype=float,
    )
    obstacles = _obstacles()
    old = get_backend()
    try:
        set_backend("numpy")
        reference = np.asarray(
            [chain_collision_report(chain, obstacles, link_radius=0.03)["minimum_clearance"] for chain in points]
        )
        set_backend("auto")
        accelerated = chain_clearance_batch(points, obstacles, link_radius=0.03)
    finally:
        set_backend(old)

    assert np.allclose(accelerated, reference, atol=1e-12)


def test_batch_clearance_supports_three_dimensional_capsule():
    cpp = _load_cpp_collision("auto")
    if cpp is None:
        pytest.skip("C++ collision extension is unavailable")
    points = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    obstacle = CapsuleObstacle((0.5, 0.3, 0.0), (0.5, 0.3, 1.0), 0.1)
    expected = chain_collision_report(points[0], [obstacle])["minimum_clearance"]
    assert np.allclose(chain_clearance_batch(points, [obstacle]), [expected])


def test_cpp_broadphase_matches_exact_clearance_thresholds():
    cpp = _load_cpp_collision("auto")
    if cpp is None or not hasattr(cpp, "chain_collision_free_batch"):
        pytest.skip("C++ collision broad-phase extension is unavailable")
    points = np.array(
        [
            [[0.0, 0.0], [0.8, 0.0], [1.6, 0.0]],
            [[0.0, 0.8], [0.8, 0.8], [1.6, 0.8]],
            [[0.0, 2.0], [0.8, 2.0], [1.6, 2.0]],
        ]
    )
    obstacles = _obstacles() + [SphereObstacle((20.0, 20.0), 0.5)]
    exact = chain_clearance_batch(points, obstacles, link_radius=0.03)
    for threshold in (0.0, 0.05, 0.3):
        expected = exact >= threshold
        assert np.array_equal(
            chain_collision_free_batch(points, obstacles, clearance=threshold, link_radius=0.03),
            expected,
        )
