import numpy as np

from mastermlx import rrt, smooth


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
