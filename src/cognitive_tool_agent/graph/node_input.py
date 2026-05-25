"""NodeInput — the uniform per-node execution context.

Every agent receives a NodeInput instead of positional arguments.  This
removes heterogeneous arities from the executor and is the enabler for
edge-driven wiring (Phase 2+).

Ambient fields (always present, not determined by graph edges):
    user_input, registry, row   — run-level constants
    trace_so_far                — computed from RunContext.to_trace() at the
                                  moment the learn node dispatches.  INVARIANT:
                                  (a) this field is NEVER populated from an
                                  incoming edge, (b) it is NEVER narrowed in
                                  Phase 5, and (c) learn must be topologically
                                  last so that to_trace() includes all prior
                                  node outputs.

Edge-supplied fields (None if no incoming edge provides the slot):
    perception, reasoning, grounding, readiness, plan, action

Phase 1: every node receives a fully-populated NodeInput (all slots filled
from RunContext).  Narrowing to only edge-supplied slots happens in Phase 5.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas.act import ActionResult
from ..schemas.common import UserInput
from ..schemas.dataset import DatasetRow
from ..schemas.grounding import GroundingResult
from ..schemas.learn import LearningResult
from ..schemas.perceive import PerceptionResult
from ..schemas.plan import PlanResult
from ..schemas.readiness import ReadinessResult
from ..schemas.reason import ReasoningResult
from ..schemas.trace import CognitiveTrace
from ..tools.registry import ToolRegistry


@dataclass
class NodeInput:
    # --- ambient (always present, not edge-driven) ---
    user_input: UserInput
    registry: ToolRegistry
    row: DatasetRow
    # trace_so_far: ALWAYS computed from RunContext.to_trace() at dispatch time
    # for the learn node.  NEVER populated from an incoming edge.  NEVER
    # narrowed in Phase 5.  Invariant: learn is topologically last.
    trace_so_far: CognitiveTrace | None = field(default=None)
    # --- edge-supplied (None if no edge provides the slot) ---
    perception: PerceptionResult | None = field(default=None)
    reasoning: ReasoningResult | None = field(default=None)
    grounding: GroundingResult | None = field(default=None)
    readiness: ReadinessResult | None = field(default=None)
    plan: PlanResult | None = field(default=None)
    action: ActionResult | None = field(default=None)
    learning: LearningResult | None = field(default=None)
