# Cognitive Graph Recommendation

_Generated: 2026-05-25 19:04 UTC — Source: `reports/cognitive_dataset_report.json`_

## Capability Inference

| Capability | Required | Strength | Evidence |
|---|:---:|---:|---|
| `memory` | **YES** | 0.54 | 89.9% of 'order_id' values are tool-chained |
| `grounding` | **YES** | 0.85 | 85.2% of 'item_ids' values require grounding (peak arg, 88 instances) |
| `readiness` | **YES** | 0.43 | write actions comprise 26 of 60 failures |
| `deep_planning` | **YES** | 0.61 | avg chain depth is 3.04 across tools |

## Raw Signals

| Signal | Value |
|---|---:|
| `chaining_strength` | 0.5407 |
| `grounding_strength` | 0.1513 |
| `peak_grounding_strength` | 0.8520 |
| `peak_grounding_instances` | 88.0000 |
| `write_fraction` | 0.1943 |
| `avg_chain_depth` | 3.0375 |
| `write_failure_fraction` | 0.4333 |

## Recommended Graph

> `perceive → reason → grounding → readiness → plan → act → learn`

| Property | Value |
|---|---|
| Graph ID | `recommended_memory_grounding_readiness_deep_planning` |
| Nodes | 7 |
| Confidence | **0.61** |
| memory_required | True |
| readiness_required | True |
| parallel_lookup_nodes | False |

### Node sequence

| # | Node | Role |
|---:|---|---|
| 1 | `perceive` | perceive |
| 2 | `reason` | reason |
| 3 | `grounding` | grounding |
| 4 | `readiness` | readiness |
| 5 | `plan` | plan |
| 6 | `act` | act |
| 7 | `learn` | learn |

## Rationale

- perceive always included (signal extraction baseline)
- plan + act always included (core action execution)
- reason included: avg_chain_depth=3.04 exceeds reasoning threshold
- grounding included: grounding_strength=0.85 — NL→structured mapping required
- readiness included: readiness_strength=0.43 — write-risk gate required
- learn included: chaining_strength=0.54 — persistent state/memory required

## Capability Detail

### `memory` — Required (strength 0.54)

- 89.9% of 'order_id' values are tool-chained

### `grounding` — Required (strength 0.85)

- 85.2% of 'item_ids' values require grounding (peak arg, 88 instances)
- global grounding_strength=0.151 (diluted by high-volume zero-grounding args)

### `readiness` — Required (strength 0.43)

- write actions comprise 26 of 60 failures
- write_fraction=0.19

### `deep_planning` — Required (strength 0.61)

- avg chain depth is 3.04 across tools
