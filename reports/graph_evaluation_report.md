# Graph Evaluation Report

_Generated: 2026-05-25 17:44 UTC_

**Source dataset**: `data/out/action_sequence.jsonl`  
**Rows evaluated**: 98

## Comparison Table

| Graph | Nodes | E2E Success | Tool Acc | Arg Match | Policy Viol | Failures | Grounding |
|---|---:|---:|---:|---:|---:|---:|---|
| `monolithic` | 2 | 0% | 0% | 0% | 0% | 98 | n/a |
| `minimal` | 3 | 0% | 0% | 0% | 0% | 98 | n/a |
| `recommended_stub` | 7 | 0% | 0% | 0% | 0% | 98 | stub |
| `recommended_oracle` | 7 | 0% | 0% | 0% | 0% | 98 | oracle |

## Revision Suggestions

### [HIGH] `grounding` — argument_resolution_failure

**Suggestion**: Split grounding node: separate 'item_ids' resolution from general entity grounding. High-frequency ID fields need dedicated lookup chains.

**Rationale**: argument_exact_match=0.00 (below 0.50 threshold). Peak grounding arg: 'item_ids' with strength=0.85.

**Evidence**:
- `argument_exact_match=0.000`
- `peak_grounding_arg=item_ids`
- `peak_grounding_instances=88`

### [MEDIUM] `graph_topology` — recommended_not_beating_minimal

**Suggestion**: Recommended graph is not outperforming the minimal graph. This is likely the stub-grounding ceiling. Resolve grounding before concluding that the topology is wrong.

**Rationale**: recommended_stub E2E=0.00 ≤ minimal E2E=0.00. Stub grounding cannot resolve IDs, making extra nodes a drag.

**Evidence**:
- `recommended_stub_e2e=0.000`
- `minimal_e2e=0.000`
- `grounding_mode=stub`
