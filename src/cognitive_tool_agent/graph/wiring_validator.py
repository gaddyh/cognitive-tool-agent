"""Load-time wiring validator for GraphSpec.

validate_wiring(graph) is called at GraphExecutor.run() entry, before the
node loop.  It is the first and authoritative line of defense against
misconfigured graphs.  _build_node_input's silent skip of ordering-only
edges is never the first check.

Algorithm:
1. For each edge: resolve provides (infer from ROLE_OUTPUT[from_node.role]
   if None; reject explicit value that contradicts ROLE_OUTPUT).
2. Warn if an edge supplies a slot not in ROLE_INPUTS[to_node.role]
   (ordering-only edge — e.g. act→learn).
3. Error if two incoming edges to the same node supply the same slot.
4. For each node: if any ROLE_INPUTS[role] slot is "required" and no
   incoming edge provides it → WiringError.
5. Warn if any node appears after a learn node in topological order
   (learn invariant guard: to_trace() would miss that node's output).

Returns WiringReport(errors, warnings).  Raise on errors is caller's
responsibility (GraphExecutor.run() raises; tests may inspect directly).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas.graph_spec import GraphSpec
from ..schemas.node_io import ROLE_INPUTS, ROLE_OUTPUT


class WiringError(ValueError):
    """Raised when a graph has a hard wiring violation."""


@dataclass
class WiringReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def validate_wiring(graph: GraphSpec) -> WiringReport:
    """Validate edge wiring for a GraphSpec.  Returns a WiringReport.

    Callers that want fail-fast behaviour should call::

        report = validate_wiring(graph)
        if not report.ok:
            raise WiringError(report.errors[0])
    """
    report = WiringReport()
    nmap = graph.node_map

    # slot_providers[to_node_id][slot] = from_node_id
    # Used to detect duplicate providers.
    slot_providers: dict[str, dict[str, str]] = {n.id: {} for n in graph.nodes}
    # incoming_slots[to_node_id] = set of slots provided by incoming edges
    incoming_slots: dict[str, set[str]] = {n.id: set() for n in graph.nodes}

    for edge in graph.edges:
        from_node = nmap.get(edge.from_node)
        to_node = nmap.get(edge.to_node)

        if from_node is None:
            report.errors.append(
                f"Edge references unknown from_node '{edge.from_node}'"
            )
            continue
        if to_node is None:
            report.errors.append(
                f"Edge references unknown to_node '{edge.to_node}'"
            )
            continue

        # Resolve provides
        canonical = ROLE_OUTPUT.get(from_node.role)
        if edge.provides is None:
            slot = canonical
            if slot is None:
                report.errors.append(
                    f"Edge {edge.from_node}→{edge.to_node}: cannot infer provides — "
                    f"role '{from_node.role}' has no entry in ROLE_OUTPUT"
                )
                continue
        else:
            slot = edge.provides
            if canonical is not None and slot != canonical:
                report.errors.append(
                    f"Edge {edge.from_node}→{edge.to_node}: explicit provides='{slot}' "
                    f"contradicts ROLE_OUTPUT['{from_node.role}'] = '{canonical}'"
                )
                continue

        # Check for duplicate providers of the same slot to the same node
        if slot in slot_providers[to_node.id]:
            existing = slot_providers[to_node.id][slot]
            report.errors.append(
                f"Node '{to_node.id}': slot '{slot}' is provided by both "
                f"'{existing}' and '{from_node.id}'"
            )
        else:
            slot_providers[to_node.id][slot] = from_node.id
            incoming_slots[to_node.id].add(slot)

        # Warn if slot is not in ROLE_INPUTS for to_node (ordering-only edge)
        to_inputs = ROLE_INPUTS.get(to_node.role, {})
        if slot not in to_inputs:
            report.warnings.append(
                f"Edge {edge.from_node}→{edge.to_node}: slot '{slot}' is not in "
                f"ROLE_INPUTS['{to_node.role}'] — treating as ordering-only edge"
            )

    # Check required inputs are satisfied
    for node in graph.nodes:
        role_inputs = ROLE_INPUTS.get(node.role, {})
        for slot, requirement in role_inputs.items():
            if requirement == "required" and slot not in incoming_slots[node.id]:
                report.errors.append(
                    f"Node '{node.id}' (role='{node.role}'): required input slot "
                    f"'{slot}' has no incoming edge providing it"
                )

    # Learn invariant: warn if any node appears after a learn node in topo order
    try:
        topo = graph.topological_order()
    except (ValueError, KeyError):
        report.errors.append(
            f"GraphSpec '{graph.id}' has a cycle or references an unknown node id"
        )
        return report

    seen_learn = False
    for node in topo:
        if seen_learn and node.role not in ("memory",):
            report.warnings.append(
                f"Node '{node.id}' (role='{node.role}') appears after a learn node "
                f"in topological order — learn's to_trace() will not include this "
                f"node's output, violating the learn invariant"
            )
        if node.role == "learn":
            seen_learn = True

    return report
