import numpy as np
import pytest

from mastermlx import get_backend, set_backend
from mastermlx.robotics import (
    BoxObstacle,
    CapsuleObstacle,
    SphereObstacle,
    chain_clearance_batch,
    chain_collision_details_batch,
    chain_collision_free_batch,
    chain_collision_summary_batch,
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


def test_cpp_collision_summary_matches_detailed_reports():
    cpp = _load_cpp_collision("auto")
    if cpp is None or not hasattr(cpp, "chain_collision_summary_batch"):
        pytest.skip("C++ collision summary extension is unavailable")
    points = np.array(
        [
            [[0.0, 0.0], [0.8, 0.0], [1.6, 0.0]],
            [[0.0, 2.0], [0.8, 2.0], [1.6, 2.0]],
        ]
    )
    obstacles = _obstacles()
    old = get_backend()
    try:
        set_backend("numpy")
        reference = [
            chain_collision_report(chain, obstacles, link_radius=0.03) for chain in points
        ]
        set_backend("auto")
        summary = chain_collision_summary_batch(points, obstacles, link_radius=0.03)
    finally:
        set_backend(old)

    assert np.allclose(summary["minimum_clearance"], [item["minimum_clearance"] for item in reference])
    assert np.array_equal(summary["collision"], [item["collision"] for item in reference])
    expected_kinds = [{None: 0, "point": 1, "segment": 2}[item["closest"]["kind"]] for item in reference]
    expected_indices = [
        item["closest"]["index"] if item["closest"]["index"] is not None else -1
        for item in reference
    ]
    expected_obstacles = [
        item["closest"]["obstacle_index"]
        if item["closest"]["obstacle_index"] is not None
        else -1
        for item in reference
    ]
    assert np.array_equal(summary["closest_kind"], expected_kinds)
    assert np.array_equal(summary["closest_index"], expected_indices)
    assert np.array_equal(summary["closest_obstacle_index"], expected_obstacles)


def test_cpp_collision_details_match_hit_reports_and_reuse_buffers():
    cpp = _load_cpp_collision("auto")
    if cpp is None or not hasattr(cpp, "chain_collision_details_batch"):
        pytest.skip("C++ collision details extension is unavailable")
    points = np.array(
        [
            [[0.0, 0.0], [0.8, 0.0], [1.6, 0.0]],
            [[0.0, 2.0], [0.8, 2.0], [1.6, 2.0]],
        ]
    )
    obstacles = _obstacles()
    old = get_backend()
    try:
        set_backend("numpy")
        reference = [chain_collision_report(chain, obstacles, link_radius=0.03) for chain in points]
        set_backend("auto")
        details = chain_collision_details_batch(points, obstacles, link_radius=0.03)
    finally:
        set_backend(old)

    assert np.array_equal(details["hit_count"], [len(item["hits"]) for item in reference])
    assert not np.any(details["hit_truncated"])
    for sample, report in enumerate(reference):
        hits = report["hits"]
        assert np.array_equal(
            details["hit_kind"][sample, : len(hits)],
            [{"point": 1, "segment": 2}[hit["kind"]] for hit in hits],
        )
        assert np.array_equal(
            details["hit_index"][sample, : len(hits)],
            [hit.get("point_index", hit.get("segment_index")) for hit in hits],
        )
        assert np.array_equal(
            details["hit_obstacle_index"][sample, : len(hits)],
            [hit["obstacle_index"] for hit in hits],
        )
        assert np.allclose(
            details["hit_clearance"][sample, : len(hits)],
            [hit["clearance"] for hit in hits],
        )

    capacity = 1
    output = {
        "minimum_clearance": np.empty(points.shape[0]),
        "collision": np.empty(points.shape[0], dtype=bool),
        "closest_kind": np.empty(points.shape[0], dtype=np.int8),
        "closest_index": np.empty(points.shape[0], dtype=np.int64),
        "closest_obstacle_index": np.empty(points.shape[0], dtype=np.int64),
        "hit_count": np.empty(points.shape[0], dtype=np.int64),
        "hit_truncated": np.empty(points.shape[0], dtype=bool),
        "hit_kind": np.empty((points.shape[0], capacity), dtype=np.int8),
        "hit_index": np.empty((points.shape[0], capacity), dtype=np.int64),
        "hit_obstacle_index": np.empty((points.shape[0], capacity), dtype=np.int64),
        "hit_clearance": np.empty((points.shape[0], capacity)),
    }
    reused = chain_collision_details_batch(points, obstacles, max_hits=capacity, output=output)
    assert reused["hit_kind"] is output["hit_kind"]
    assert np.all(reused["hit_truncated"] == (reused["hit_count"] > capacity))
