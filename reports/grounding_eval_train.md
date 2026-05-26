# Grounding Evaluation — train

_Generated: 2026-05-26 01:32 UTC_

**Total rows**: 1

## Grounding Quality

| variant | req_arg_match | field_prec | field_rec | field_f1 | missing | halluc | schema_ok | confidence | cost_usd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `deterministic` | 0% | 0% | 0% | 0% | 100% | 0% | 100% | — | — |
| `grounding_llm_v1` | 100% | 100% | 100% | 100% | 0% | 0% | 100% | 1.00 | 0.0026 |

## Grounding Node Latency

| variant | avg_latency_ms |
|---|---:|
| `deterministic` | 0.00 |
| `grounding_llm_v1` | 1437.66 |

## Per-Simulation Grounding Latency

| sim_key | turns | det_total_ms | llm_total_ms | llm_avg_ms_per_turn |
|---|---:|---:|---:|---:|
| `78:04d0d690` | 1 | 0.000 | 1437.7 | 1437.7 |

## Baseline Tau-Bench Timing

_simulation_count_: 62

| metric | value |
|---|---:|
| tau_duration_seconds_total | 1502.216 |
| tau_duration_seconds_avg | 24.229 |
| tau_agent_generation_time_seconds_total | 1031.429 |
| tau_agent_generation_time_seconds_avg_per_assistant_turn | 1.413 |
| tau_agent_generation_turns_total | 730 |
| message_span_seconds_total | 1502.192 |
| message_span_seconds_avg | 24.229 |

_Note: `message_span_seconds` = timestamp span from first to last message (includes user/tool/framework overhead)._