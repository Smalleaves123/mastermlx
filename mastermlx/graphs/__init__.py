"""Graph algorithms, DAG utilities, and lightweight knowledge engineering."""

from .core import astar, bfs, dfs, dijkstra
from .algorithms import (
    TaskSchedule,
    connected_components,
    find_subgraph_matches,
    schedule_dag,
    strongly_connected_components,
    to_dot,
    topological_levels,
    topological_sort,
)
from .knowledge import KnowledgeGraph, Rule, Triple
from .csr import (
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

__all__ = [
    "KnowledgeGraph",
    "CSRGraph",
    "Rule",
    "TaskSchedule",
    "Triple",
    "WeightedCSRGraph",
    "astar",
    "bfs",
    "bfs_csr",
    "connected_components",
    "connected_components_csr",
    "dijkstra_csr",
    "dfs",
    "dijkstra",
    "find_subgraph_matches",
    "graph_backend_report",
    "multi_source_bfs_csr",
    "schedule_dag",
    "strongly_connected_components",
    "strongly_connected_components_csr",
    "to_dot",
    "topological_levels",
    "topological_sort",
    "topological_sort_csr",
]
