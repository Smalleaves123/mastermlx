import pytest

from mastermlx import (
    KnowledgeGraph,
    Rule,
    Triple,
    connected_components,
    find_subgraph_matches,
    schedule_dag,
    strongly_connected_components,
    to_dot,
    topological_levels,
    topological_sort,
)


def test_connected_components_and_strong_components():
    graph = {"a": ["b"], "b": ["a", "c"], "c": [], "d": []}

    components = [set(item) for item in connected_components(graph)]
    assert {"a", "b", "c"} in components
    assert {"d"} in components

    strong = [set(item) for item in strongly_connected_components(graph)]
    assert {"a", "b"} in strong
    assert {"c"} in strong


def test_topological_sort_levels_and_cycle_detection():
    graph = {"fetch": ["clean"], "clean": ["train"], "train": ["report"], "report": []}

    assert topological_sort(graph) == ["fetch", "clean", "train", "report"]
    assert topological_levels(graph) == [["fetch"], ["clean"], ["train"], ["report"]]
    with pytest.raises(ValueError, match="cycle"):
        topological_sort({"a": ["b"], "b": ["a"]})


def test_schedule_dag_respects_dependencies_and_workers():
    graph = {"a": ["c"], "b": ["c"], "c": []}
    schedule = schedule_dag(graph, {"a": 2, "b": 1, "c": 3}, max_workers=2)

    assert schedule["a"].finish <= schedule["c"].start
    assert schedule["b"].finish <= schedule["c"].start
    assert schedule["c"].finish == 5


def test_subgraph_matching_and_dot_export():
    graph = {"a": ["b", "c"], "b": ["c"], "c": []}
    pattern = {"x": ["y"], "y": []}

    matches = find_subgraph_matches(graph, pattern)
    assert {"x": "a", "y": "b"} in matches
    assert {"x": "a", "y": "c"} in matches
    dot = to_dot(graph)
    assert "digraph G" in dot
    assert "n0 -> n1" in dot


def test_knowledge_graph_query_path_and_rule_inference():
    graph = KnowledgeGraph(
        [
            Triple("alice", "parent", "bob"),
            Triple("bob", "parent", "carol"),
        ]
    )
    rule = Rule(
        (Triple("?x", "parent", "?y"), Triple("?y", "parent", "?z")),
        Triple("?x", "grandparent", "?z"),
    )

    assert graph.query(predicate="parent") == [
        Triple("alice", "parent", "bob"),
        Triple("bob", "parent", "carol"),
    ]
    assert graph.path("alice", "carol", predicate="parent") == ["alice", "bob", "carol"]
    assert graph.infer([rule]) == 1
    assert graph.query("alice", "grandparent", "carol")
    assert 'label="parent"' in graph.to_dot()


def test_rule_variables_can_bind_none():
    graph = KnowledgeGraph([Triple("alice", "value", None)])
    rule = Rule((Triple("?x", "value", None),), Triple("?x", "known", True))

    assert graph.infer([rule]) == 1
    assert graph.query("alice", "known", True)
