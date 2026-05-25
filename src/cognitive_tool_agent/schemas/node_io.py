"""Role → I/O maps for load-time wiring validation and edge-driven dispatch.

These maps are the single source of truth for what each node role produces
and what it can consume from upstream edges.

ROLE_OUTPUT
-----------
Maps each role to the *one* RunContext/NodeInput slot it writes.

ONE CANONICAL OUTPUT SLOT PER ROLE — invariant.
If a role ever gains two outputs, provides=None becomes ambiguous for that
role.  Explicit provides will then be required for all edges from that role.
Update this comment when the invariant breaks.

ROLE_INPUTS
-----------
Maps each role to the slots it can consume from incoming edges, along with
whether the slot is "required" or "optional".

    "required"  → validator raises WiringError if no incoming edge provides it
    "optional"  → agent handles None gracefully; validator only warns

LIMITATION: only act's 'plan' dependency is "required"; everything else is
"optional" because agents degrade gracefully when upstream is absent.  The
validator therefore enforces only broken act nodes, not misconfigured full
pipelines.  The full-pipeline equivalence test is the primary safety net for
full-pipeline wiring correctness.

Note on 'learn': trace_so_far is ambient (computed from RunContext at dispatch
time) — it is never edge-supplied and never appears in ROLE_INPUTS.
"""
from __future__ import annotations

ROLE_OUTPUT: dict[str, str] = {
    "perceive":  "perception",
    "reason":    "reasoning",
    "readiness": "readiness",
    "grounding": "grounding",
    "plan":      "plan",
    "act":       "action",
    "learn":     "learning",
}

ROLE_INPUTS: dict[str, dict[str, str]] = {
    "perceive":  {},
    "reason":    {"perception": "optional"},
    "readiness": {"reasoning":  "optional"},
    "grounding": {"reasoning":  "optional"},
    "plan":      {"reasoning":  "optional", "readiness": "optional"},
    "act":       {"plan":       "required"},
    "learn":     {},
}
