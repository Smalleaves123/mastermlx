import numpy as np

from mastermlx import get_backend, set_backend
from mastermlx.graphs import astar
from mastermlx.robotics import smooth_joint_path
from mastermlx.utils.backend import use_cpp_backend, use_cython_backend


def test_smooth_joint_path_numpy_and_compiled_backends_match_boundary_system():
    path = np.array([[0.0], [10.0], [-10.0], [0.0]])
    previous = get_backend()
    try:
        set_backend("numpy")
        numpy_result = smooth_joint_path(path, smoothness=1.0)
        set_backend("auto")
        compiled_result = smooth_joint_path(path, smoothness=1.0)
    finally:
        set_backend(previous)

    expected = np.array([[0.0], [2.5], [-2.5], [0.0]])
    assert np.allclose(numpy_result, expected)
    assert np.allclose(compiled_result, expected)
    assert np.allclose(numpy_result, compiled_result)


def test_compiled_backend_predicates_keep_cython_and_cpp_modes_distinct():
    assert use_cpp_backend("auto")
    assert not use_cpp_backend("cython")
    assert not use_cpp_backend("numpy")
    assert use_cython_backend("auto")
    assert use_cython_backend("cython")
    assert not use_cython_backend("numpy")


def test_cython_mode_does_not_select_cpp_only_graph_kernel(monkeypatch):
    from mastermlx.graphs import core

    def fail_if_called(*args, **kwargs):
        raise AssertionError("C++ kernel was selected in cython mode")

    monkeypatch.setattr(core, "_astar_cpp", fail_if_called)
    previous = get_backend()
    try:
        set_backend("cython")
        path, cost = astar(np.zeros((2, 2), dtype=int), (0, 0), (1, 1))
    finally:
        set_backend(previous)

    assert path[0] == (0, 0)
    assert path[-1] == (1, 1)
    assert cost == 2.0
