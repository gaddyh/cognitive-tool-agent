from __future__ import annotations

from dataclasses import dataclass

from typing import Any

from ..adapters.base import ModelAdapter
from ..agents.act_agent import ActAgent
from ..agents.grounding_agent import GroundingAgent
from ..agents.learn_agent import LearnAgent
from ..agents.perceive_agent import PerceiveAgent
from ..agents.plan_agent import PlanAgent
from ..agents.readiness_agent import ReadinessAgent
from ..agents.reason_agent import ReasonAgent
from .node_input import NodeInput
from .wiring_validator import WiringError, validate_wiring
from ..schemas.common import UserInput
from ..schemas.dataset import DatasetRow
from ..schemas.experiment import ExperimentSpec, NodeRuntimeConfig
from ..schemas.graph_spec import EdgeSpec, GraphSpec, NodeSpec, NodeRole  # noqa: F401
from ..schemas.grounding import GroundingResult
from ..schemas.node_io import ROLE_INPUTS, ROLE_OUTPUT
from ..schemas.perceive import PerceptionResult
from ..schemas.reason import ReasoningResult
from ..schemas.readiness import ReadinessResult
from ..schemas.plan import PlanResult
from ..schemas.act import ActionResult
from ..schemas.learn import LearningResult
from ..schemas.trace import CognitiveTrace
from ..tools.registry import ToolRegistry

_ROLE_TO_AGENT: dict[str, type] = {
    "perceive":  PerceiveAgent,
    "reason":    ReasonAgent,
    "readiness": ReadinessAgent,
    "grounding": GroundingAgent,
    "plan":      PlanAgent,
    "act":       ActAgent,
    "learn":     LearnAgent,
}


@dataclass
class RunContext:
    row: DatasetRow
    registry: ToolRegistry
    user_input: UserInput
    perception: PerceptionResult | None = None
    reasoning: ReasoningResult | None = None
    grounding: GroundingResult | None = None
    readiness: ReadinessResult | None = None
    plan: PlanResult | None = None
    action: ActionResult | None = None
    learning: LearningResult | None = None

    def to_trace(self) -> CognitiveTrace:
        return CognitiveTrace(
            input=self.user_input,
            perception=self.perception,
            reasoning=self.reasoning,
            grounding=self.grounding,
            readiness=self.readiness,
            plan=self.plan,
            action=self.action,
            learning=self.learning,
        )


def _row_to_user_input(row: DatasetRow, registry: ToolRegistry) -> UserInput:
    available_tools = [
        schema
        for name in row.tools
        if (schema := registry.lookup(name)) is not None
    ]
    return UserInput(
        message=row.user_message,
        conversation_context=row.context,
        available_tools=available_tools,
        world_state=row.world_state,
    )


class GraphExecutor:
    """
    Node-driven executor.  Accepts an ExperimentSpec (topology + per-node
    execution config) and iterates nodes in topological order.

    Nodes absent from ExperimentSpec.runtime default to mode='stub' with no
    adapter.  The runtime config is keyed by node_id (not role) so graphs
    with multiple nodes sharing a role remain unambiguous.
    """

    def __init__(self, model_adapters: dict[str, ModelAdapter] | None = None) -> None:
        # Maps adapter hint string (e.g. "openai:gpt-4.1") to a resolved
        # ModelAdapter object.  Empty until real adapters are wired.
        self._model_adapters: dict[str, ModelAdapter] = model_adapters or {}

    def run(
        self,
        experiment: ExperimentSpec,
        row: DatasetRow,
        registry: ToolRegistry,
    ) -> CognitiveTrace:
        # Guard: typo'd node_id in runtime config silently falls to stub mode,
        # making an oracle experiment look like a stub run (oracle gap = zero).
        graph_node_ids = {n.id for n in experiment.graph.nodes}
        runtime_map = experiment.runtime_map()
        unknown_ids = set(runtime_map) - graph_node_ids
        if unknown_ids:
            raise ValueError(
                f"ExperimentSpec.runtime references node_ids not present in graph "
                f"'{experiment.graph.id}': {sorted(unknown_ids)}"
            )

        user_input = _row_to_user_input(row, registry)

        report = validate_wiring(experiment.graph)
        if not report.ok:
            raise WiringError(
                f"Graph '{experiment.graph.id}' has wiring errors:\n"
                + "\n".join(f"  - {e}" for e in report.errors)
            )

        ctx = RunContext(row=row, registry=registry, user_input=user_input)
        ordered_nodes = experiment.graph.topological_order()

        for node in ordered_nodes:
            node_cfg = runtime_map.get(node.id, NodeRuntimeConfig(node_id=node.id))
            self._dispatch(node, ctx, node_cfg, experiment.graph)

        return ctx.to_trace()

    def _make_agent(
        self, role: str, cfg: NodeRuntimeConfig
    ) -> PerceiveAgent | ReasonAgent | ReadinessAgent | GroundingAgent | PlanAgent | ActAgent | LearnAgent:
        agent_cls = _ROLE_TO_AGENT.get(role)
        if agent_cls is None:
            if role == "monolithic":
                raise NotImplementedError(
                    "Node role 'monolithic' is deprecated. "
                    "Express the baseline as a two-node graph: plan → act with no upstream edges. "
                    "See graph_candidate_generator.make_monolithic_baseline() for the canonical form."
                )
            if role == "memory":
                raise NotImplementedError(
                    "Node role 'memory' is recommender-only in v1. "
                    "GraphRecommender maps memory capability to 'learn' node, not 'memory'. "
                    "If you see this error, a graph was hand-crafted with a 'memory' node."
                )
            raise ValueError(f"Unknown node role: {role!r}")
        model_adapter = self._model_adapters.get(cfg.adapter) if cfg.adapter else None
        return agent_cls(mode=cfg.mode, model_adapter=model_adapter)

    def _build_node_input(
        self, node: NodeSpec, ctx: RunContext, graph: GraphSpec
    ) -> NodeInput:
        """Gather edge-supplied slots for this node from RunContext.

        Only slots listed in ROLE_INPUTS[node.role] are included — ordering-only
        edges (e.g. act→learn) are skipped silently.  The validator at run()
        entry is the first line of defense against mis-wired ordering edges;
        this skip is never the only check.

        learn invariant: trace_so_far is never edge-supplied.  It is always
        computed from RunContext.to_trace() after all prior nodes have written
        their results.  ROLE_INPUTS['learn'] is empty, so no incoming edge
        can pass the slot guard below.
        """
        supplied: dict[str, Any] = {}
        role_inputs = ROLE_INPUTS.get(node.role, {})
        for edge in graph.edges:
            if edge.to_node == node.id:
                slot = edge.provides or ROLE_OUTPUT.get(
                    graph.node_map[edge.from_node].role
                )
                if slot and slot in role_inputs:
                    supplied[slot] = getattr(ctx, slot, None)

        trace_so_far = ctx.to_trace() if node.role == "learn" else None

        return NodeInput(
            user_input=ctx.user_input,
            registry=ctx.registry,
            row=ctx.row,
            trace_so_far=trace_so_far,
            **supplied,
        )

    def _dispatch(
        self, node: NodeSpec, ctx: RunContext, cfg: NodeRuntimeConfig, graph: GraphSpec
    ) -> None:
        node_input = self._build_node_input(node, ctx, graph)
        agent = self._make_agent(node.role, cfg)
        result = agent.run(node_input)
        output_slot = ROLE_OUTPUT.get(node.role)
        if output_slot:
            setattr(ctx, output_slot, result)
