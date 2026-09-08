import numpy as np

from mastermlx import rrt, rrt_star, smooth
from mastermlx.planning.core import _clear, _grow_node_storage, _reparent


def test_rrt_finds_a_free_path():
    path = rrt(
        [0.1, 0.1],
        [0.9, 0.9],
        bounds=[[0.0, 1.0], [0.0, 1.0]],
        step=0.1,
        goal_rate=0.2,
        max_iter=2000,
        random_state=0,
    )

    assert path is not None
    assert np.allclose(path[0], [0.1, 0.1])
    assert np.allclose(path[-1], [0.9, 0.9])
    assert np.all(np.linalg.norm(np.diff(path, axis=0), axis=1) <= 0.1 + 1e-12)


def test_planner_node_storage_grows_geometrically_and_preserves_nodes():
    nodes = np.array([[0.0, 1.0], [2.0, 3.0]])

    expanded = _grow_node_storage(nodes, count=2, maximum=5)
    capped = _grow_node_storage(expanded, count=4, maximum=5)

    assert expanded.shape == (4, 2)
    assert capped.shape == (5, 2)
    assert np.array_equal(expanded[:2], nodes)


def test_clear_without_collision_callback_skips_edge_sampling(monkeypatch):
    def fail_sampling(*args, **kwargs):
        raise AssertionError("collision-free edges should not be sampled")

    monkeypatch.setattr(np, "linspace", fail_sampling)

    assert _clear(np.array([0.0]), np.array([1.0]), None, 0.01)


def test_reparent_propagates_cost_change_to_descendants():
    parents = [-1, 0, 1, 0]
    costs = [0.0, 5.0, 9.0, 1.0]
    children = [{1, 3}, {2}, set(), set()]

    _reparent(1, 3, 3.0, parents, costs, children)

    assert parents == [-1, 3, 1, 0]
    assert costs == [0.0, 3.0, 7.0, 1.0]
    assert children == [{3}, {2}, set(), {1}]


def test_rrt_avoids_obstacle():
    def hit(p):
        return 0.4 < p[0] < 0.6 and p[1] < 0.8

    path = rrt(
        [0.1, 0.1],
        [0.9, 0.1],
        bounds=[[0.0, 1.0], [0.0, 1.0]],
        hit=hit,
        step=0.08,
        goal_rate=0.2,
        max_iter=10000,
        random_state=0,
    )

    assert path is not None
    assert np.all([not hit(point) for point in path])
    assert np.max(path[:, 1]) > 0.8
    assert np.all(np.linalg.norm(np.diff(path, axis=0), axis=1) <= 0.08 + 1e-12)


def test_rrt_star_finds_and_rewires_a_free_path():
    def hit(p):
        return 0.4 < p[0] < 0.6 and p[1] < 0.8

    path = rrt_star(
        [0.1, 0.1],
        [0.9, 0.1],
        bounds=[[0.0, 1.0], [0.0, 1.0]],
        hit=hit,
        step=0.08,
        goal_rate=0.25,
        search_radius=0.25,
        max_iter=1200,
        random_state=2,
        stop_on_first_path=True,
    )

    assert path is not None
    assert np.allclose(path[0], [0.1, 0.1])
    assert np.allclose(path[-1], [0.9, 0.1])
    assert np.all([not hit(point) for point in path])
    assert path.shape[0] < 60


def test_rrt_star_worker_pool_preserves_seeded_path_order():
    def hit(point):
        return 0.4 < point[0] < 0.6 and point[1] < 0.8

    kwargs = dict(
        start=[0.1, 0.1],
        goal=[0.9, 0.1],
        bounds=[[0.0, 1.0], [0.0, 1.0]],
        hit=hit,
        step=0.08,
        goal_rate=0.25,
        search_radius=0.25,
        max_iter=1200,
        random_state=2,
        stop_on_first_path=True,
    )
    serial = rrt_star(workers=1, **kwargs)
    parallel = rrt_star(workers=3, **kwargs)
    assert np.array_equal(parallel, serial)


def test_rrt_worker_validation_is_bounded():
    with np.testing.assert_raises(ValueError):
        rrt([0.0], [1.0], [[0.0, 1.0]], workers=0)


def test_smooth_keeps_endpoints():
    path = np.array([[0.0, 0.0], [0.2, 0.5], [0.5, 0.2], [1.0, 1.0]])

    out = smooth(path, n=50, random_state=0)

    assert np.allclose(out[0], path[0])
    assert np.allclose(out[-1], path[-1])


def test_rrt_supports_arbitrary_dimensions():
    path = rrt(
        [0.1, 0.1, 0.1],
        [0.9, 0.9, 0.9],
        bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        step=0.15,
        goal_rate=0.2,
        max_iter=2000,
        random_state=0,
    )

    assert path is not None
    assert path.shape[1] == 3
    assert np.allclose(path[0], [0.1, 0.1, 0.1])
    assert np.allclose(path[-1], [0.9, 0.9, 0.9])


def test_smooth_supports_arbitrary_dimensions():
    path = np.array([
        [0.0, 0.0, 0.0],
        [0.2, 0.5, 0.1],
        [0.5, 0.2, 0.4],
        [1.0, 1.0, 1.0],
    ])

    out = smooth(path, n=50, random_state=0)

    assert out.shape[1] == 3
    assert np.allclose(out[0], path[0])
    assert np.allclose(out[-1], path[-1])
