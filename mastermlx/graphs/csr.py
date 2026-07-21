"""CSR graph representation with optional C++ traversal kernels."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from functools import lru_cache
import importlib

import numpy as np

from ..config import get_backend
from .algorithms import Graph, _adjacency, connected_components, topological_sort
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


@lru_cache(maxsize=2)
def _load_cpp(backend: str | None = None):
    if backend is None:
        backend = get_backend()
    if backend != "auto":
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
        self.directed = bool(directed)
        self.indptr.setflags(write=False)
        self.indices.setflags(write=False)

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


def _as_csr(graph: CSRGraph | Graph) -> CSRGraph:
    return graph if isinstance(graph, CSRGraph) else CSRGraph.from_graph(graph)


def bfs_csr(graph: CSRGraph | Graph, start: Hashable) -> list[Hashable]:
    """Run BFS on a CSR graph, using C++ when the compiled backend is available."""

    csr = _as_csr(graph)
    start_id = csr.nodes.index(start)
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


__all__ = [
    "CSRGraph",
    "bfs_csr",
    "connected_components_csr",
    "graph_backend_report",
    "topological_sort_csr",
]
