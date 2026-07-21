"""Benchmark Python and optional C++ kernels on CSR graph storage."""

from __future__ import annotations

import time

import numpy as np

from mastermlx import get_backend, set_backend
from mastermlx.graphs import (
    CSRGraph,
    bfs_csr,
    connected_components_csr,
    graph_backend_report,
    topological_sort_csr,
)


def bench(fn, n_runs=3):
    fn()
    timings = []
    result = None
    for _ in range(n_runs):
        start = time.perf_counter()
        result = fn()
        timings.append(time.perf_counter() - start)
    return float(np.mean(timings)), result


def make_chain(n_nodes=50_000):
    nodes = tuple(range(n_nodes))
    indptr = np.arange(n_nodes + 1, dtype=np.int64)
    indptr[-1] = n_nodes - 1
    indices = np.arange(1, n_nodes, dtype=np.int64)
    return CSRGraph(nodes, indptr, indices)


def main():
    graph = make_chain()
    print(f"C++ graph backend: {graph_backend_report()}")
    old = get_backend()
    try:
        for backend in ("numpy", "auto"):
            set_backend(backend)
            bfs_time, bfs_result = bench(lambda: bfs_csr(graph, 0))
            component_time, components = bench(lambda: connected_components_csr(graph))
            topo_time, topo_result = bench(lambda: topological_sort_csr(graph))
            print(
                f"{backend:>5}  bfs={bfs_time:.5f}s ({len(bfs_result)} nodes)  "
                f"components={component_time:.5f}s ({len(components)} groups)  "
                f"topological={topo_time:.5f}s ({len(topo_result)} nodes)"
            )
    finally:
        set_backend(old)


if __name__ == "__main__":
    main()
