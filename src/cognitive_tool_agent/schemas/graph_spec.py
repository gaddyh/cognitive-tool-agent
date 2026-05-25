from __future__ import annotations

from collections import defaultdict, deque
from typing import Literal

from pydantic import BaseModel


NodeRole = Literal[
    "perceive",
    "reason",
    "readiness",
    "plan",
    "act",
    "learn",
    "monolithic",
]


class NodeSpec(BaseModel):
    id: str
    role: NodeRole
    model_hint: str = "stub"
    prompt_template: str | None = None
    input_schema: str | None = None
    output_schema: str | None = None


class EdgeSpec(BaseModel):
    from_node: str
    to_node: str
    condition: str | None = None


class GraphSpec(BaseModel):
    id: str
    nodes: list[NodeSpec]
    edges: list[EdgeSpec] = []
    latency_estimate: float = 1.0
    cost_estimate: float = 1.0

    def topological_order(self) -> list[NodeSpec]:
        node_map = {n.id: n for n in self.nodes}
        in_degree: dict[str, int] = defaultdict(int)
        adjacency: dict[str, list[str]] = defaultdict(list)

        for node in self.nodes:
            in_degree.setdefault(node.id, 0)

        for edge in self.edges:
            adjacency[edge.from_node].append(edge.to_node)
            in_degree[edge.to_node] += 1

        queue: deque[str] = deque(n.id for n in self.nodes if in_degree[n.id] == 0)
        order: list[NodeSpec] = []

        while queue:
            nid = queue.popleft()
            order.append(node_map[nid])
            for successor in adjacency[nid]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        if len(order) != len(self.nodes):
            raise ValueError(f"Cycle detected in GraphSpec '{self.id}'")

        return order
