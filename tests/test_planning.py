import numpy as np

from mastermlx import rrt, rrt_star, smooth


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
