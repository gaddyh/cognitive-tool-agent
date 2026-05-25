from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..schemas.common import UserInput
from ..schemas.dataset import DatasetRow
from ..schemas.graph_spec import EdgeSpec, GraphSpec, NodeSpec, NodeRole  # noqa: F401
from ..schemas.perceive import PerceptionResult
from ..schemas.reason import ReasoningResult
from ..schemas.readiness import ReadinessResult
from ..schemas.plan import PlanResult
from ..schemas.act import ActionResult
from ..schemas.learn import LearningResult
from ..schemas.trace import CognitiveTrace
from ..tools.registry import ToolRegistry


@dataclass
class RunContext:
    row: DatasetRow
    registry: ToolRegistry
    perception: PerceptionResult | None = None
    reasoning: ReasoningResult | None = None
    readiness: ReadinessResult | None = None
    plan: PlanResult | None = None
    action: ActionResult | None = None
    learning: LearningResult | None = None

    def to_trace(self) -> CognitiveTrace:
        user_input = _row_to_user_input(self.row, self.registry)
        return CognitiveTrace(
            input=user_input,
            perception=self.perception,
            reasoning=self.reasoning,
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
    Node-driven executor.  Iterates GraphSpec.nodes in topological order and
    dispatches each node to the agent registered for its role.  No cognitive
    stages are hardcoded; the graph definition fully controls execution.
    """

    def __init__(self) -> None:
        from ..agents.perceive_agent import PerceiveAgent
        from ..agents.reason_agent import ReasonAgent
        from ..agents.readiness_agent import ReadinessAgent
        from ..agents.plan_agent import PlanAgent
        from ..agents.act_agent import ActAgent
        from ..agents.learn_agent import LearnAgent

        self._perceive = PerceiveAgent()
        self._reason = ReasonAgent()
        self._readiness = ReadinessAgent()
        self._plan = PlanAgent()
        self._act = ActAgent()
        self._learn = LearnAgent()

    def run(self, graph_spec: GraphSpec, row: DatasetRow, registry: ToolRegistry) -> CognitiveTrace:
        ctx = RunContext(row=row, registry=registry)
        ordered_nodes = graph_spec.topological_order()

        for node in ordered_nodes:
            self._dispatch(node, ctx)

        return ctx.to_trace()

    def _dispatch(self, node: NodeSpec, ctx: RunContext) -> None:
        role = node.role
        user_input = _row_to_user_input(ctx.row, ctx.registry)

        if role == "perceive":
            ctx.perception = self._perceive.run(user_input)

        elif role == "reason":
            ctx.reasoning = self._reason.run(user_input, ctx.perception)

        elif role == "readiness":
            ctx.readiness = self._readiness.run(user_input, ctx.reasoning, ctx.registry)

        elif role == "plan":
            ctx.plan = self._plan.run(user_input, ctx.reasoning, ctx.readiness)

        elif role == "act":
            ctx.action = self._act.run(ctx.plan, ctx.registry)

        elif role == "learn":
            ctx.learning = self._learn.run(user_input, ctx.to_trace())

        elif role == "monolithic":
            ctx.plan, ctx.action = _run_monolithic(user_input, ctx.registry)

        else:
            raise ValueError(f"Unknown node role: {role!r}")


def _run_monolithic(
    user_input: UserInput, registry: ToolRegistry
) -> tuple[PlanResult, ActionResult]:
    """
    Monolithic baseline: direct keyword → tool mapping, no cognitive stages.
    """
    from ..agents.plan_agent import PlanAgent
    from ..agents.act_agent import ActAgent

    plan = PlanAgent().run(user_input, reasoning=None, readiness=None)
    action = ActAgent().run(plan, registry)
    return plan, action
