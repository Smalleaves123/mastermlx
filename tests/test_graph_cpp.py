import numpy as np
import pytest

from mastermlx import get_backend, set_backend
from mastermlx.graphs import (
    CSRGraph,
    WeightedCSRGraph,
    bfs_csr,
    connected_components_csr,
    dijkstra_csr,
    graph_backend_report,
    multi_source_bfs_csr,
    strongly_connected_components_csr,
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


def _weighted_graph():
    return {
        "a": [("b", 2.0), ("c", 5.0)],
        "b": [("c", 1.0), ("d", 5.0)],
        "c": [("d", 1.0)],
        "d": [],
        "isolated": [],
    }


def test_weighted_csr_round_trip_and_validation():
    graph = WeightedCSRGraph.from_graph(_weighted_graph())

    assert graph.to_weighted_adjacency() == _weighted_graph()
    with pytest.raises(ValueError, match="non-negative"):
        WeightedCSRGraph.from_graph({"a": [("b", -1.0)], "b": []})


def test_weighted_dijkstra_and_multisource_bfs_python_fallback():
    old = get_backend()
    try:
        set_backend("numpy")
        weighted = WeightedCSRGraph.from_graph(_weighted_graph())
        path, cost = dijkstra_csr(weighted, "a", "d")
        assert path == ["a", "b", "c", "d"]
        assert np.isclose(cost, 4.0)
        assert dijkstra_csr(weighted, "a")["d"] == 4.0
        assert dijkstra_csr(weighted, "a", "isolated") == (None, np.inf)

        graph = CSRGraph.from_graph(_graph())
        assert multi_source_bfs_csr(graph, ["a", "isolated"]) == {
            "a": 0,
            "b": 1,
            "c": 1,
            "d": 2,
            "isolated": 0,
        }
    finally:
        set_backend(old)


def test_scc_python_fallback():
    graph = CSRGraph.from_graph(
        {
            "a": ["b"],
            "b": ["a", "c"],
            "c": ["d"],
            "d": ["c"],
            "isolated": [],
        }
    )
    old = get_backend()
    try:
        set_backend("numpy")
        components = strongly_connected_components_csr(graph)
        assert {frozenset(component) for component in components} == {
            frozenset({"a", "b"}),
            frozenset({"c", "d"}),
            frozenset({"isolated"}),
        }
    finally:
        set_backend(old)


def test_cpp_graph_kernels_match_python_for_new_algorithms():
    if not graph_backend_report()["cpp"]:
        pytest.skip("C++ graph extension is unavailable")
    old = get_backend()
    try:
        weighted = WeightedCSRGraph.from_graph(_weighted_graph())
        graph = CSRGraph.from_graph(
            {
                "a": ["b"],
                "b": ["a", "c"],
                "c": ["d"],
                "d": ["c"],
                "isolated": [],
            }
        )
        set_backend("numpy")
        py_path, py_cost = dijkstra_csr(weighted, "a", "d")
        py_distances = dijkstra_csr(weighted, "a")
        py_bfs = multi_source_bfs_csr(graph, ["a", "isolated"])
        py_scc = {frozenset(component) for component in strongly_connected_components_csr(graph)}
        set_backend("auto")
        cpp_path, cpp_cost = dijkstra_csr(weighted, "a", "d")
        cpp_distances = dijkstra_csr(weighted, "a")
        cpp_bfs = multi_source_bfs_csr(graph, ["a", "isolated"])
        cpp_scc = {frozenset(component) for component in strongly_connected_components_csr(graph)}

        assert cpp_path == py_path
        assert np.isclose(cpp_cost, py_cost)
        assert cpp_distances == py_distances
        assert cpp_bfs == py_bfs
        assert cpp_scc == py_scc
    finally:
        set_backend(old)
