"""Benchmark Python and optional C++ kernels on CSR graph storage."""

from __future__ import annotations

import time

import numpy as np

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


def make_weighted_chain(n_nodes=50_000):
    graph = make_chain(n_nodes)
    return WeightedCSRGraph(graph.nodes, graph.indptr, graph.indices, np.ones(graph.n_edges))


def main():
    graph = make_chain()
    weighted_graph = make_weighted_chain()
    starts = [0, weighted_graph.n_nodes // 2]
    print(f"C++ graph backend: {graph_backend_report()}")
    old = get_backend()
    try:
        for backend in ("numpy", "auto"):
            set_backend(backend)
            bfs_time, bfs_result = bench(lambda: bfs_csr(graph, 0))
            component_time, components = bench(lambda: connected_components_csr(graph))
            topo_time, topo_result = bench(lambda: topological_sort_csr(graph))
            multisource_time, distances = bench(lambda: multi_source_bfs_csr(graph, starts))
            scc_time, scc = bench(lambda: strongly_connected_components_csr(graph))
            dijkstra_time, shortest = bench(
                lambda: dijkstra_csr(weighted_graph, 0, weighted_graph.n_nodes - 1)
            )
            print(
                f"{backend:>5}  bfs={bfs_time:.5f}s ({len(bfs_result)} nodes)  "
                f"components={component_time:.5f}s ({len(components)} groups)  "
                f"topological={topo_time:.5f}s ({len(topo_result)} nodes)  "
                f"multi_bfs={multisource_time:.5f}s ({len(distances)} nodes)  "
                f"scc={scc_time:.5f}s ({len(scc)} groups)  "
                f"dijkstra={dijkstra_time:.5f}s (cost={shortest[1]:.1f})"
            )
    finally:
        set_backend(old)


if __name__ == "__main__":
    main()
