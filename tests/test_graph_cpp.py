import numpy as np
import pytest

from mastermlx import get_backend, set_backend
from mastermlx.graphs import (
    CSRGraph,
    bfs_csr,
    connected_components_csr,
    graph_backend_report,
    topological_sort_csr,
)


def _graph():
    return {
        "a": ["b", "c"],
        "b": ["d"],
        "c": ["d"],
        "d": [],
        "isolated": [],
    }


def test_csr_graph_round_trip_and_python_fallback():
    old = get_backend()
    try:
        graph = CSRGraph.from_graph(_graph())
        set_backend("numpy")
        assert graph.to_adjacency() == _graph()
        assert bfs_csr(graph, "a") == ["a", "b", "c", "d"]
        assert [set(component) for component in connected_components_csr(graph)] == [
            {"a", "b", "c", "d"},
            {"isolated"},
        ]
        assert topological_sort_csr(graph) == ["a", "b", "c", "d", "isolated"]
    finally:
        set_backend(old)


def test_cpp_csr_kernels_match_python_when_available():
    if not graph_backend_report()["cpp"]:
        pytest.skip("C++ graph extension is unavailable")
    graph = CSRGraph.from_graph(_graph())
    old = get_backend()
    try:
        set_backend("numpy")
        bfs_ref = bfs_csr(graph, "a")
        components_ref = [set(component) for component in connected_components_csr(graph)]
        topo_ref = topological_sort_csr(graph)
        set_backend("auto")
        assert bfs_csr(graph, "a") == bfs_ref
        assert [set(component) for component in connected_components_csr(graph)] == components_ref
        assert topological_sort_csr(graph) == topo_ref
    finally:
        set_backend(old)


def test_csr_validation_rejects_invalid_arrays():
    with pytest.raises(ValueError, match="indptr"):
        CSRGraph((0, 1), np.array([1, 0]), np.array([1]))


def test_undirected_csr_builds_reverse_edges_but_is_not_a_dag():
    graph = CSRGraph.from_graph({"a": ["b"], "b": []}, directed=False)

    assert graph.to_adjacency() == {"a": ["b"], "b": ["a"]}
    with pytest.raises(ValueError, match="directed"):
        graph.topological_sort()
