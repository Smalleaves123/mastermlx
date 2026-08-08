"""CSR graph representation with optional C++ traversal kernels."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from functools import lru_cache
import heapq
import importlib
import math

import numpy as np

from ..config import get_backend
from ..utils.backend import use_cpp_backend
from .algorithms import (
    Graph,
    _adjacency,
    connected_components,
    strongly_connected_components,
    topological_sort,
)
from .core import bfs


def _validate_arrays(indptr: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indptr = np.asarray(indptr, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    if indptr.ndim != 1 or indptr.size < 1:
        raise ValueError("indptr must be a non-empty 1D array")
    if indices.ndim != 1:
        raise ValueError("indices must be a 1D array")
    if indptr[0] != 0 or indptr[-1] != indices.size:
        raise ValueError("indptr must start at zero and end at len(indices)")
    if np.any(indptr[1:] < indptr[:-1]) or np.any(indptr < 0):
        raise ValueError("indptr must be non-decreasing and non-negative")
    nodes = indptr.size - 1
    if np.any(indices < 0) or np.any(indices >= nodes):
        raise ValueError("indices contain a node outside the graph")
    return np.ascontiguousarray(indptr), np.ascontiguousarray(indices)


def _weighted_adjacency(
    graph: Mapping[Hashable, Iterable[Hashable]],
) -> dict[Hashable, list[tuple[Hashable, float]]]:
    if not isinstance(graph, Mapping):
        raise TypeError("graph must be a mapping of nodes to weighted neighbors")
    adjacency: dict[Hashable, dict[Hashable, float]] = {
        node: {} for node in graph
    }
    for source, neighbors in graph.items():
        for item in neighbors:
            if (
                isinstance(item, (tuple, list))
                and len(item) == 2
                and isinstance(item[1], (int, float, np.number))
            ):
                target, weight = item
            else:
                target, weight = item, 1.0
            value = float(weight)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("weights must be finite and non-negative")
            adjacency.setdefault(target, {})
            previous = adjacency[source].get(target)
            adjacency[source][target] = value if previous is None else min(previous, value)
    return {
        source: list(neighbors.items())
        for source, neighbors in adjacency.items()
    }


@lru_cache(maxsize=2)
def _load_cpp(backend: str | None = None):
    if backend is None:
        backend = get_backend()
    if not use_cpp_backend(backend):
        return None
    try:
        return importlib.import_module("mastermlx.graphs._graph_cpp")
    except ImportError:
        return None


def graph_backend_report() -> dict[str, str | bool]:
    """Report the requested graph backend and whether the C++ module is usable."""

    cpp = _load_cpp(get_backend())
    return {"requested": get_backend(), "cpp": cpp is not None, "active": "cpp" if cpp else "numpy"}


class CSRGraph:
    """Compact adjacency representation that preserves arbitrary node labels."""

    def __init__(
        self,
        nodes: Iterable[Hashable],
        indptr: np.ndarray,
        indices: np.ndarray,
        *,
        directed: bool = True,
    ) -> None:
        self.nodes = tuple(nodes)
        self.indptr, self.indices = _validate_arrays(indptr, indices)
        if len(self.nodes) != self.indptr.size - 1:
            raise ValueError("nodes must match the number of CSR rows")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("nodes must be unique")
        self._node_ids = {node: index for index, node in enumerate(self.nodes)}
        self.directed = bool(directed)
        self.indptr.setflags(write=False)
        self.indices.setflags(write=False)

    def node_id(self, node: Hashable) -> int:
        try:
            return self._node_ids[node]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"unknown graph node: {node!r}") from exc

    @classmethod
    def from_graph(
        cls,
        graph: Graph,
        *,
        node_order: Iterable[Hashable] | None = None,
        directed: bool = True,
    ) -> "CSRGraph":
        adjacency = _adjacency(graph)
        nodes = list(node_order) if node_order is not None else list(adjacency)
        if set(nodes) != set(adjacency):
            raise ValueError("node_order must contain exactly the graph nodes")
        node_ids = {node: index for index, node in enumerate(nodes)}
        reverse: dict[Hashable, list[Hashable]] = {node: [] for node in nodes}
        if not directed:
            for source, targets in adjacency.items():
                for target in targets:
                    reverse[target].append(source)
        rows: list[int] = [0]
        values: list[int] = []
        for node in nodes:
            neighbors = list(adjacency[node])
            if not directed:
                neighbors += reverse[node]
            values.extend(dict.fromkeys(node_ids[neighbor] for neighbor in neighbors))
            rows.append(len(values))
        return cls(nodes, np.asarray(rows), np.asarray(values), directed=directed)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return int(self.indices.size)

    def to_adjacency(self) -> dict[Hashable, list[Hashable]]:
        return {
            node: [
                self.nodes[index] for index in self.indices[self.indptr[row] : self.indptr[row + 1]]
            ]
            for row, node in enumerate(self.nodes)
        }

    def bfs(self, start: Hashable) -> list[Hashable]:
        return bfs_csr(self, start)

    def connected_components(self) -> list[list[Hashable]]:
        return connected_components_csr(self)

    def topological_sort(self) -> list[Hashable]:
        return topological_sort_csr(self)


class WeightedCSRGraph(CSRGraph):
    """CSR graph with non-negative edge weights for weighted algorithms."""

    def __init__(
        self,
        nodes: Iterable[Hashable],
        indptr: np.ndarray,
        indices: np.ndarray,
        weights: np.ndarray,
        *,
        directed: bool = True,
    ) -> None:
        super().__init__(nodes, indptr, indices, directed=directed)
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 1 or weights.size != self.indices.size:
            raise ValueError("weights must be a 1D array matching indices")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("weights must be finite and non-negative")
        self.weights = np.ascontiguousarray(weights)
        self.weights.setflags(write=False)

    @classmethod
    def from_graph(
        cls,
        graph: Mapping[Hashable, Iterable[Hashable]],
        *,
        node_order: Iterable[Hashable] | None = None,
        directed: bool = True,
    ) -> "WeightedCSRGraph":
        adjacency = _weighted_adjacency(graph)
        nodes = list(node_order) if node_order is not None else list(adjacency)
        if set(nodes) != set(adjacency):
            raise ValueError("node_order must contain exactly the graph nodes")
        node_ids = {node: index for index, node in enumerate(nodes)}
        rows: list[int] = [0]
        values: list[int] = []
        weights: list[float] = []

        reverse: dict[Hashable, dict[Hashable, float]] = {node: {} for node in nodes}
        if not directed:
            for source, targets in adjacency.items():
                for target, weight in targets:
                    previous = reverse[target].get(source)
                    reverse[target][source] = weight if previous is None else min(previous, weight)

        for node in nodes:
            neighbors = dict(adjacency[node])
            if not directed:
                for neighbor, weight in reverse[node].items():
                    previous = neighbors.get(neighbor)
                    neighbors[neighbor] = weight if previous is None else min(previous, weight)
            for neighbor, weight in neighbors.items():
                values.append(node_ids[neighbor])
                weights.append(weight)
            rows.append(len(values))

        return cls(
            nodes,
            np.asarray(rows, dtype=np.int64),
            np.asarray(values, dtype=np.int64),
            np.asarray(weights, dtype=np.float64),
            directed=directed,
        )

    def to_weighted_adjacency(self) -> dict[Hashable, list[tuple[Hashable, float]]]:
        return {
            node: [
                (self.nodes[index], float(weight))
                for index, weight in zip(
                    self.indices[self.indptr[row] : self.indptr[row + 1]],
                    self.weights[self.indptr[row] : self.indptr[row + 1]],
                )
            ]
            for row, node in enumerate(self.nodes)
        }

    def dijkstra(self, start: Hashable, goal: Hashable | None = None):
        return dijkstra_csr(self, start, goal)

    def strongly_connected_components(self) -> list[list[Hashable]]:
        return strongly_connected_components_csr(self)


def _as_csr(graph: CSRGraph | Graph) -> CSRGraph:
    return graph if isinstance(graph, CSRGraph) else CSRGraph.from_graph(graph)


def bfs_csr(graph: CSRGraph | Graph, start: Hashable) -> list[Hashable]:
    """Run BFS on a CSR graph, using C++ when the compiled backend is available."""

    csr = _as_csr(graph)
    start_id = csr.node_id(start)
    cpp = _load_cpp(get_backend())
    if cpp is not None:
        order = cpp.bfs_order(csr.indptr, csr.indices, start_id)
        return [csr.nodes[int(index)] for index in order]
    return bfs(csr.to_adjacency(), start)


def _components_from_labels(csr: CSRGraph, labels: np.ndarray) -> list[list[Hashable]]:
    groups: dict[int, list[Hashable]] = {}
    for node, label in zip(csr.nodes, labels):
        groups.setdefault(int(label), []).append(node)
    return list(groups.values())


def connected_components_csr(graph: CSRGraph | Graph) -> list[list[Hashable]]:
    """Find undirected connected components on CSR storage."""

    csr = _as_csr(graph)
    cpp = _load_cpp(get_backend())
    if cpp is not None:
        return _components_from_labels(csr, cpp.connected_components(csr.indptr, csr.indices))
    return connected_components(csr.to_adjacency())


def topological_sort_csr(graph: CSRGraph | Graph) -> list[Hashable]:
    """Topologically sort CSR storage, using C++ when available."""

    csr = _as_csr(graph)
    if not csr.directed:
        raise ValueError("topological sort requires a directed CSR graph")
    cpp = _load_cpp(get_backend())
    if cpp is not None:
        order = cpp.topological_order(csr.indptr, csr.indices)
        return [csr.nodes[int(index)] for index in order]
    return topological_sort(csr.to_adjacency())


def _path_from_predecessors(
    predecessors: np.ndarray,
    start_id: int,
    goal_id: int,
) -> list[int] | None:
    path: list[int] = []
    current = goal_id
    seen: set[int] = set()
    while current != -1:
        if current in seen:
            return None
        seen.add(current)
        path.append(current)
        if current == start_id:
            path.reverse()
            return path
        current = int(predecessors[current])
    return None


def _dijkstra_numpy(
    csr: WeightedCSRGraph,
    start_id: int,
    goal_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.full(csr.n_nodes, np.inf, dtype=np.float64)
    predecessors = np.full(csr.n_nodes, -1, dtype=np.int64)
    distances[start_id] = 0.0
    queue: list[tuple[float, int]] = [(0.0, start_id)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost > distances[node]:
            continue
        if node == goal_id:
            break
        for edge in range(int(csr.indptr[node]), int(csr.indptr[node + 1])):
            neighbor = int(csr.indices[edge])
            next_cost = cost + float(csr.weights[edge])
            if next_cost < distances[neighbor]:
                distances[neighbor] = next_cost
                predecessors[neighbor] = node
                heapq.heappush(queue, (next_cost, neighbor))
    return distances, predecessors


def dijkstra_csr(
    graph: WeightedCSRGraph | Mapping[Hashable, Iterable[Hashable]],
    start: Hashable,
    goal: Hashable | None = None,
):
    """Find weighted shortest paths on CSR storage.

    With ``goal`` set, return ``(path, cost)``. Otherwise return distances for
    reachable nodes. The compiled backend is used for weighted CSR graphs when
    available, with a numerically equivalent Python fallback.
    """

    csr = graph if isinstance(graph, WeightedCSRGraph) else WeightedCSRGraph.from_graph(graph)
    start_id = csr.node_id(start)
    goal_id = -1 if goal is None else csr.node_id(goal)
    cpp = _load_cpp(get_backend())
    if cpp is not None:
        distances, predecessors = cpp.dijkstra_weighted(
            csr.indptr,
            csr.indices,
            csr.weights,
            start_id,
            goal_id,
        )
        distances = np.asarray(distances)
        predecessors = np.asarray(predecessors)
    else:
        distances, predecessors = _dijkstra_numpy(csr, start_id, goal_id)

    if goal is None:
        return {
            node: float(distance)
            for node, distance in zip(csr.nodes, distances)
            if np.isfinite(distance)
        }
    cost = float(distances[goal_id])
    if not np.isfinite(cost):
        return None, math.inf
    path_ids = _path_from_predecessors(predecessors, start_id, goal_id)
    if path_ids is None:
        return None, math.inf
    return [csr.nodes[index] for index in path_ids], cost


def multi_source_bfs_csr(
    graph: CSRGraph | Graph,
    starts: Iterable[Hashable],
) -> dict[Hashable, int]:
    """Return shortest unweighted distances from multiple source nodes."""

    csr = _as_csr(graph)
    start_ids = np.asarray([csr.node_id(node) for node in starts], dtype=np.int64)
    cpp = _load_cpp(get_backend())
    if cpp is not None:
        distances = cpp.multi_source_bfs_distances(csr.indptr, csr.indices, start_ids)
    else:
        distances = np.full(csr.n_nodes, -1, dtype=np.int64)
        queue: list[int] = []
        for start_id in start_ids:
            start_id = int(start_id)
            if distances[start_id] == -1:
                distances[start_id] = 0
                queue.append(start_id)
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            for edge in range(int(csr.indptr[node]), int(csr.indptr[node + 1])):
                neighbor = int(csr.indices[edge])
                if distances[neighbor] == -1:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
    return {node: int(distance) for node, distance in zip(csr.nodes, distances)}


def strongly_connected_components_csr(
    graph: CSRGraph | Graph,
) -> list[list[Hashable]]:
    """Find strongly connected components on CSR storage."""

    csr = _as_csr(graph)
    cpp = _load_cpp(get_backend())
    if cpp is not None:
        labels = cpp.strongly_connected_components(csr.indptr, csr.indices)
        return _components_from_labels(csr, labels)
    return strongly_connected_components(csr.to_adjacency())


__all__ = [
    "CSRGraph",
    "WeightedCSRGraph",
    "bfs_csr",
    "connected_components_csr",
    "dijkstra_csr",
    "graph_backend_report",
    "multi_source_bfs_csr",
    "strongly_connected_components_csr",
    "topological_sort_csr",
]
