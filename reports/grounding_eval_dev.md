# Grounding Evaluation — dev

_Generated: 2026-05-26 01:37 UTC_

**Total rows**: 10

## Grounding Quality

| variant | req_arg_match | field_prec | field_rec | field_f1 | missing | halluc | schema_ok | confidence | cost_usd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `deterministic` | 30% | 58% | 32% | 42% | 59% | 14% | 100% | — | — |
| `grounding_llm_v1` | 60% | 75% | 60% | 67% | 12% | 7% | 100% | 0.95 | 0.1445 |

## Grounding Node Latency

| variant | avg_latency_s |
|---|---:|
| `deterministic` | 0.00 |
| `grounding_llm_v1` | 2.16 |

## Per-Simulation Grounding Latency

| sim_key | turns | det_total_s | llm_total_s | llm_avg_s_per_turn |
|---|---:|---:|---:|---:|
| `23:28cfedcd` | 6 | 0.000 | 15.634 | 2.606 |
| `98:4378b93a` | 4 | 0.000 | 5.946 | 1.487 |

## Baseline Tau-Bench Timing

_simulation_count_: 20

| metric | value |
|---|---:|
| tau_duration_seconds_total | 393.827 |
| tau_duration_seconds_avg | 19.691 |
| tau_agent_generation_time_seconds_total | 263.629 |
| tau_agent_generation_time_seconds_avg_per_assistant_turn | 1.221 |
| tau_agent_generation_turns_total | 216 |
| message_span_seconds_total | 393.82 |
| message_span_seconds_avg | 19.691 |

_Note: `message_span_seconds` = timestamp span from first to last message (includes user/tool/framework overhead)._