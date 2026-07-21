"""General graph algorithms and lightweight graph export helpers."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
import heapq
from typing import Any


Graph = Mapping[Hashable, Iterable[Hashable]]


def _adjacency(graph: Graph) -> dict[Hashable, list[Hashable]]:
    if not isinstance(graph, Mapping):
        raise TypeError("graph must be a mapping of nodes to neighbors")
    adjacency: dict[Hashable, list[Hashable]] = {
        node: list(neighbors) for node, neighbors in graph.items()
    }
    for neighbors in adjacency.values():
        for node in neighbors:
            adjacency.setdefault(node, [])
    return adjacency


def connected_components(graph: Graph) -> list[list[Hashable]]:
    """Return connected components of an undirected adjacency mapping."""

    adjacency = _adjacency(graph)
    reverse: dict[Hashable, list[Hashable]] = {node: [] for node in adjacency}
    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            reverse[neighbor].append(node)

    components: list[list[Hashable]] = []
    unseen = set(adjacency)
    while unseen:
        start = next(node for node in adjacency if node in unseen)
        unseen.remove(start)
        queue = deque([start])
        component: list[Hashable] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in (*adjacency[node], *reverse[node]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def strongly_connected_components(graph: Graph) -> list[list[Hashable]]:
    """Return strongly connected components using Tarjan's algorithm."""

    adjacency = _adjacency(graph)
    index = 0
    indices: dict[Hashable, int] = {}
    lowlink: dict[Hashable, int] = {}
    stack: list[Hashable] = []
    on_stack: set[Hashable] = set()
    result: list[list[Hashable]] = []

    def visit(node: Hashable) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in adjacency[node]:
            if neighbor not in indices:
                visit(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif neighbor in on_stack:
                lowlink[node] = min(lowlink[node], indices[neighbor])

        if lowlink[node] != indices[node]:
            return
        component: list[Hashable] = []
        while True:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node:
                break
        result.append(component)

    for node in adjacency:
        if node not in indices:
            visit(node)
    return result


def topological_sort(graph: Graph) -> list[Hashable]:
    """Return a deterministic topological order or raise on a directed cycle."""

    adjacency = _adjacency(graph)
    order = {node: position for position, node in enumerate(adjacency)}
    indegree = {node: 0 for node in adjacency}
    for neighbors in adjacency.values():
        for neighbor in neighbors:
            indegree[neighbor] += 1

    ready = [(order[node], node) for node, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    result: list[Hashable] = []
    while ready:
        _, node = heapq.heappop(ready)
        result.append(node)
        for neighbor in adjacency[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                heapq.heappush(ready, (order[neighbor], neighbor))

    if len(result) != len(adjacency):
        raise ValueError("graph contains a directed cycle")
    return result


def topological_levels(graph: Graph) -> list[list[Hashable]]:
    """Group DAG nodes by dependency depth, starting with source nodes."""

    adjacency = _adjacency(graph)
    order = topological_sort(adjacency)
    levels: dict[Hashable, int] = {}
    for node in order:
        level = levels.get(node, 0)
        for neighbor in adjacency[node]:
            levels[neighbor] = max(levels.get(neighbor, 0), level + 1)

    grouped: list[list[Hashable]] = []
    for node in order:
        level = levels.get(node, 0)
        while len(grouped) <= level:
            grouped.append([])
        grouped[level].append(node)
    return grouped


@dataclass(frozen=True)
class TaskSchedule:
    """Scheduled interval for one DAG task."""

    start: float
    finish: float
    worker: int = 0


def schedule_dag(
    graph: Graph,
    durations: Mapping[Hashable, float] | None = None,
    max_workers: int | None = None,
) -> dict[Hashable, TaskSchedule]:
    """Schedule DAG tasks with dependency constraints.

    ``durations`` defaults to one time unit per task. With no worker limit,
    independent tasks start as soon as their dependencies finish. A positive
    ``max_workers`` enables deterministic list scheduling.
    """

    adjacency = _adjacency(graph)
    order = topological_sort(adjacency)
    position = {node: index for index, node in enumerate(order)}
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be positive")

    task_durations = {node: 1.0 for node in adjacency}
    if durations is not None:
        for node, duration in durations.items():
            if node not in adjacency:
                raise KeyError(f"duration specified for unknown node: {node!r}")
            value = float(duration)
            if value < 0 or value != value or value == float("inf"):
                raise ValueError("task durations must be finite and non-negative")
            task_durations[node] = value

    predecessors: dict[Hashable, list[Hashable]] = {node: [] for node in adjacency}
    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            predecessors[neighbor].append(node)

    if max_workers is None:
        result: dict[Hashable, TaskSchedule] = {}
        for node in order:
            start = max((result[parent].finish for parent in predecessors[node]), default=0.0)
            result[node] = TaskSchedule(start, start + task_durations[node])
        return result

    indegree = {node: len(predecessors[node]) for node in adjacency}
    ready = [(position[node], node) for node, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    running: list[tuple[float, int, Hashable, int]] = []
    available_workers = list(range(max_workers))
    heapq.heapify(available_workers)
    result = {}
    clock = 0.0
    while ready or running:
        while ready and len(running) < max_workers:
            _, node = heapq.heappop(ready)
            start = max((result[parent].finish for parent in predecessors[node]), default=clock)
            worker = heapq.heappop(available_workers)
            schedule = TaskSchedule(start, start + task_durations[node], worker)
            result[node] = schedule
            heapq.heappush(running, (schedule.finish, position[node], node, worker))

        clock = running[0][0]
        finished: list[tuple[float, int, Hashable, int]] = []
        while running and running[0][0] <= clock:
            finished.append(heapq.heappop(running))
        for _, _, node, worker in finished:
            heapq.heappush(available_workers, worker)
            for neighbor in adjacency[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    heapq.heappush(ready, (position[neighbor], neighbor))
    return result


def find_subgraph_matches(
    graph: Graph,
    pattern: Graph,
    *,
    induced: bool = False,
) -> list[dict[Hashable, Hashable]]:
    """Find injective directed subgraph matches.

    The returned mappings use pattern nodes as keys and graph nodes as values.
    Set ``induced=True`` to require that mapped node pairs have exactly the
    same directed edges in both graphs.
    """

    target = _adjacency(graph)
    query = _adjacency(pattern)
    target_edges = {node: set(neighbors) for node, neighbors in target.items()}
    query_edges = {node: set(neighbors) for node, neighbors in query.items()}
    pattern_nodes = sorted(query, key=lambda node: (-len(query_edges[node]), repr(node)))
    candidates = list(target)
    matches: list[dict[Hashable, Hashable]] = []
    mapping: dict[Hashable, Hashable] = {}
    used: set[Hashable] = set()

    def compatible(pattern_node: Hashable, target_node: Hashable) -> bool:
        for other_pattern, other_target in mapping.items():
            if (
                other_pattern in query_edges[pattern_node]
                and other_target not in target_edges[target_node]
            ):
                return False
            if (
                pattern_node in query_edges[other_pattern]
                and target_node not in target_edges[other_target]
            ):
                return False
            if induced:
                if (
                    other_pattern not in query_edges[pattern_node]
                    and other_target in target_edges[target_node]
                ):
                    return False
                if (
                    pattern_node not in query_edges[other_pattern]
                    and target_node in target_edges[other_target]
                ):
                    return False
        return True

    def search(index: int) -> None:
        if index == len(pattern_nodes):
            matches.append(dict(mapping))
            return
        pattern_node = pattern_nodes[index]
        for target_node in candidates:
            if target_node in used or not compatible(pattern_node, target_node):
                continue
            mapping[pattern_node] = target_node
            used.add(target_node)
            search(index + 1)
            used.remove(target_node)
            del mapping[pattern_node]

    search(0)
    return matches


def to_dot(
    graph: Graph,
    *,
    directed: bool = True,
    node_labels: Mapping[Hashable, str] | None = None,
) -> str:
    """Export an adjacency mapping as Graphviz DOT without extra dependencies."""

    adjacency = _adjacency(graph)
    nodes = list(adjacency)
    identifiers = {node: f"n{index}" for index, node in enumerate(nodes)}
    edge = "->" if directed else "--"
    header = "digraph G" if directed else "graph G"

    def quote(value: Any) -> str:
        return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'

    lines = [f"{header} {{"]
    for node in nodes:
        label = node_labels.get(node, str(node)) if node_labels is not None else str(node)
        lines.append(f"  {identifiers[node]} [label={quote(label)}];")
    seen: set[tuple[Hashable, Hashable]] = set()
    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            if directed or repr(node) <= repr(neighbor):
                pair = (node, neighbor)
            else:
                pair = (neighbor, node)
            if not directed and pair in seen:
                continue
            seen.add(pair)
            lines.append(f"  {identifiers[node]} {edge} {identifiers[neighbor]};")
    lines.append("}")
    return "\n".join(lines)


__all__ = [
    "TaskSchedule",
    "connected_components",
    "find_subgraph_matches",
    "schedule_dag",
    "strongly_connected_components",
    "to_dot",
    "topological_levels",
    "topological_sort",
]
