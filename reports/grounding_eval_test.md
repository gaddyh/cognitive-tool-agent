# Grounding Evaluation — test

_Generated: 2026-05-26 01:32 UTC_

**Total rows**: 1

## Grounding Quality

| variant | req_arg_match | field_prec | field_rec | field_f1 | missing | halluc | schema_ok | confidence | cost_usd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `deterministic` | 0% | 0% | 0% | 0% | 100% | 0% | 100% | — | — |
| `grounding_llm_v1` | 100% | 100% | 100% | 100% | 0% | 0% | 100% | 1.00 | 0.0031 |

## Grounding Node Latency

| variant | avg_latency_ms |
|---|---:|
| `deterministic` | 0.00 |
| `grounding_llm_v1` | 1672.50 |

## Per-Simulation Grounding Latency

| sim_key | turns | det_total_ms | llm_total_ms | llm_avg_ms_per_turn |
|---|---:|---:|---:|---:|
| `16:075ee962` | 1 | 0.001 | 1672.5 | 1672.5 |

## Baseline Tau-Bench Timing

_simulation_count_: 18

| metric | value |
|---|---:|
| tau_duration_seconds_total | 339.928 |
| tau_duration_seconds_avg | 18.885 |
| tau_agent_generation_time_seconds_total | 213.724 |
| tau_agent_generation_time_seconds_avg_per_assistant_turn | 1.149 |
| tau_agent_generation_turns_total | 186 |
| message_span_seconds_total | 339.91 |
| message_span_seconds_avg | 18.884 |

_Note: `message_span_seconds` = timestamp span from first to last message (includes user/tool/framework overhead)._