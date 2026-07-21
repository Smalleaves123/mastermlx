"""Small dependency-free knowledge graph and forward-chaining rule engine."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Any


Term = Hashable


@dataclass(frozen=True)
class Triple:
    """A subject-predicate-object fact."""

    subject: Term
    predicate: Term
    object: Term


@dataclass(frozen=True)
class Rule:
    """A Horn-style rule expressed as premise triples and one conclusion."""

    premises: tuple[Triple, ...]
    conclusion: Triple

    def __post_init__(self) -> None:
        if not self.premises:
            raise ValueError("a rule must contain at least one premise")
        object.__setattr__(self, "premises", tuple(self.premises))


def _is_variable(term: Term) -> bool:
    return isinstance(term, str) and term.startswith("?")


def _unify(pattern: Term, value: Term, bindings: dict[str, Term]) -> bool:
    if _is_variable(pattern):
        name = str(pattern)
        if name not in bindings:
            bindings[name] = value
            return True
        return bindings[name] == value
    return pattern == value


def _match(pattern: Triple, fact: Triple, bindings: dict[str, Term]) -> bool:
    snapshot = dict(bindings)
    if all(
        _unify(expected, actual, bindings)
        for expected, actual in zip(
            (pattern.subject, pattern.predicate, pattern.object),
            (fact.subject, fact.predicate, fact.object),
        )
    ):
        return True
    bindings.clear()
    bindings.update(snapshot)
    return False


def _instantiate(triple: Triple, bindings: dict[str, Term]) -> Triple:
    values: list[Term] = []
    for term in (triple.subject, triple.predicate, triple.object):
        if _is_variable(term):
            name = str(term)
            if name not in bindings:
                raise ValueError(f"unbound rule variable: {name}")
            values.append(bindings[name])
        else:
            values.append(term)
    return Triple(*values)


class KnowledgeGraph:
    """Ordered fact store with wildcard queries and forward rule inference."""

    def __init__(self, triples: Iterable[Triple] | None = None) -> None:
        self._facts: list[Triple] = []
        self._fact_set: set[Triple] = set()
        if triples is not None:
            for triple in triples:
                self.add_triple(triple)

    def __len__(self) -> int:
        return len(self._facts)

    def __contains__(self, triple: object) -> bool:
        return triple in self._fact_set

    @property
    def triples(self) -> tuple[Triple, ...]:
        return tuple(self._facts)

    def add(self, subject: Term, predicate: Term, object: Term) -> bool:
        """Add a fact and return whether it was new."""

        return self.add_triple(Triple(subject, predicate, object))

    def add_triple(self, triple: Triple) -> bool:
        if not isinstance(triple, Triple):
            raise TypeError("triple must be a Triple instance")
        if triple in self._fact_set:
            return False
        self._fact_set.add(triple)
        self._facts.append(triple)
        return True

    def query(
        self,
        subject: Term | None = None,
        predicate: Term | None = None,
        object: Term | None = None,
    ) -> list[Triple]:
        """Return facts matching the supplied terms; ``None`` is a wildcard."""

        return [
            triple
            for triple in self._facts
            if (subject is None or triple.subject == subject)
            and (predicate is None or triple.predicate == predicate)
            and (object is None or triple.object == object)
        ]

    def match(self, pattern: Triple) -> list[dict[str, Term]]:
        """Match a triple pattern using variables named like ``"?x"``."""

        matches: list[dict[str, Term]] = []
        for fact in self._facts:
            bindings: dict[str, Term] = {}
            if _match(pattern, fact, bindings):
                matches.append(bindings)
        return matches

    def infer(self, rules: Iterable[Rule], max_rounds: int = 100) -> int:
        """Apply rules until a fixed point and return the number of new facts."""

        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        rules = tuple(rules)
        added = 0
        for _ in range(max_rounds):
            new_facts: list[Triple] = []
            for rule in rules:
                for bindings in self._rule_bindings(rule):
                    fact = _instantiate(rule.conclusion, bindings)
                    if fact not in self._fact_set and fact not in new_facts:
                        new_facts.append(fact)
            if not new_facts:
                return added
            for fact in new_facts:
                self.add_triple(fact)
            added += len(new_facts)
        return added

    def _rule_bindings(self, rule: Rule) -> list[dict[str, Term]]:
        bindings_list: list[dict[str, Term]] = [{}]
        for premise in rule.premises:
            next_bindings: list[dict[str, Term]] = []
            for bindings in bindings_list:
                for fact in self._facts:
                    candidate = dict(bindings)
                    if _match(premise, fact, candidate):
                        next_bindings.append(candidate)
            bindings_list = next_bindings
            if not bindings_list:
                break
        return bindings_list

    def path(
        self,
        start: Term,
        goal: Term,
        predicate: Term | None = None,
    ) -> list[Term] | None:
        """Find a shortest subject-object path, optionally constrained by predicate."""

        adjacency: dict[Term, list[Term]] = {}
        for triple in self._facts:
            if predicate is None or triple.predicate == predicate:
                adjacency.setdefault(triple.subject, []).append(triple.object)
        queue = deque([start])
        previous: dict[Term, Term | None] = {start: None}
        while queue:
            node = queue.popleft()
            if node == goal:
                result: list[Term] = []
                while node is not None:
                    result.append(node)
                    node = previous[node]
                return result[::-1]
            for neighbor in adjacency.get(node, []):
                if neighbor not in previous:
                    previous[neighbor] = node
                    queue.append(neighbor)
        return None

    def to_dot(self, predicate: Term | None = None) -> str:
        """Export facts as a labeled Graphviz DOT graph."""

        def quote(value: Any) -> str:
            return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'

        facts = [fact for fact in self._facts if predicate is None or fact.predicate == predicate]
        nodes: list[Term] = []
        for fact in facts:
            for node in (fact.subject, fact.object):
                if node not in nodes:
                    nodes.append(node)
        identifiers = {node: f"n{index}" for index, node in enumerate(nodes)}
        lines = ["digraph KnowledgeGraph {"]
        for node in nodes:
            lines.append(f"  {identifiers[node]} [label={quote(node)}];")
        for fact in facts:
            label = quote(fact.predicate)
            lines.append(
                f"  {identifiers[fact.subject]} -> {identifiers[fact.object]} [label={label}];"
            )
        lines.append("}")
        return "\n".join(lines)


__all__ = ["KnowledgeGraph", "Rule", "Triple"]
