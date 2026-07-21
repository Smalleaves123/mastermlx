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
    bfs_csr,
    connected_components_csr,
    graph_backend_report,
    topological_sort_csr,
)

__all__ = [
    "KnowledgeGraph",
    "CSRGraph",
    "Rule",
    "TaskSchedule",
    "Triple",
    "astar",
    "bfs",
    "bfs_csr",
    "connected_components",
    "connected_components_csr",
    "dfs",
    "dijkstra",
    "find_subgraph_matches",
    "graph_backend_report",
    "schedule_dag",
    "strongly_connected_components",
    "to_dot",
    "topological_levels",
    "topological_sort",
    "topological_sort_csr",
]
