# Graph computing and knowledge engineering

`mastermlx.graphs` provides dependency-free graph utilities while keeping the
existing mapping-based API. A graph is represented as `{node: neighbors}`;
nodes only appearing as neighbors are accepted automatically.

## Algorithms

```python
from mastermlx.graphs import (
    bfs,
    connected_components,
    dijkstra,
    find_subgraph_matches,
    schedule_dag,
    to_dot,
    topological_sort,
)

graph = {
    "extract": ["clean"],
    "clean": ["train"],
    "train": ["report"],
    "report": [],
}

order = topological_sort(graph)
schedule = schedule_dag(graph, {"extract": 2, "clean": 1, "train": 4, "report": 1})
dot = to_dot(graph)
```

`topological_sort` raises `ValueError` for cyclic dependencies. `schedule_dag`
returns `TaskSchedule(start, finish, worker)` entries and supports a bounded
`max_workers` value. `find_subgraph_matches` returns injective mappings from
pattern nodes to graph nodes; `induced=True` also checks non-edges.

## Knowledge graph

Facts are ordinary triples. Variables in rule patterns use a `?name` prefix.

```python
from mastermlx.graphs import KnowledgeGraph, Rule, Triple

knowledge = KnowledgeGraph([
    Triple("alice", "parent", "bob"),
    Triple("bob", "parent", "carol"),
])
knowledge.infer([
    Rule(
        (Triple("?x", "parent", "?y"), Triple("?y", "parent", "?z")),
        Triple("?x", "grandparent", "?z"),
    )
])

assert knowledge.query("alice", "grandparent", "carol")
assert knowledge.path("alice", "carol", predicate="parent") == [
    "alice", "bob", "carol"
]
```

The core package has no NetworkX or Graphviz runtime dependency. Use the DOT
text returned by `to_dot` or `KnowledgeGraph.to_dot` with a local Graphviz
installation when a visual artifact is needed.

## CSR and C++ acceleration

For large integer-indexed graphs, convert once to CSR storage. The original
node labels are preserved by `CSRGraph`, while the optional C++ extension works
on contiguous integer arrays internally.

```python
from mastermlx.graphs import CSRGraph, bfs_csr, topological_sort_csr

csr = CSRGraph.from_graph(graph)
visit_order = bfs_csr(csr, "extract")
task_order = topological_sort_csr(csr)
```

The accelerated functions are `bfs_csr`, `connected_components_csr`, and
`topological_sort_csr`. They fall back to the Python implementations when the
extension is unavailable or the backend is set to `numpy`. Run
`python benchmarks/bench_graphs.py` to compare both paths locally.
