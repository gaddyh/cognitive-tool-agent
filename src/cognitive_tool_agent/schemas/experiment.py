"""Experiment configuration schemas.

Separates topology (GraphSpec — what the recommender finds) from execution
apparatus (NodeRuntimeConfig — the experiment knob).

Design decisions:
- GraphRecommender emits topology only; mode is always an experiment choice,
  not a recommender finding.
- adapter is a string hint (e.g. "openai:gpt-4.1") resolved to a ModelAdapter
  object by the executor at runtime.
- node_id (not role) is the join key so graphs with multiple nodes sharing a
  role remain unambiguous.
"""
from __future__ import annotations

from pydantic import BaseModel

from ..adapters.base import AgentMode
from .graph_spec import GraphSpec


class NodeRuntimeConfig(BaseModel):
    """Per-node execution configuration for a single experiment run."""

    node_id: str
    mode: AgentMode = "stub"
    adapter: str | None = None


class ExperimentSpec(BaseModel):
    """A single serializable experiment: topology + per-node execution config.

    This is the reproducible record of a run.  Two runs differing only in
    grounding mode are two ExperimentSpecs that share the same GraphSpec.
    """

    graph: GraphSpec
    runtime: list[NodeRuntimeConfig] = []

    def runtime_map(self) -> dict[str, NodeRuntimeConfig]:
        """Return a node_id → NodeRuntimeConfig lookup."""
        return {r.node_id: r for r in self.runtime}
