# Turn-Level Graph Evaluation Report

_Generated: 2026-05-25 19:55 UTC_

**Source dataset**: `data/out/turn_supervision.jsonl`  
**Rows evaluated**: 547

## Comparison Table

| Graph | Nodes | E2E Success | Tool Acc | Arg Match | Policy Viol | Failures | Grounding |
|---|---:|---:|---:|---:|---:|---:|---|
| `monolithic` | 2 | 0% | 0% | 0% | 0% | 547 | n/a |
| `minimal` | 3 | 0% | 0% | 0% | 0% | 547 | n/a |
| `recommended_stub` | 7 | 2% | 2% | 0% | 0% | 547 | stub |
| `recommended_deterministic` | 7 | 39% | 39% | 23% | 0% | 547 | deterministic |
| `recommended_oracle` | 7 | 100% | 100% | 99% | 0% | 547 | oracle |

## Field-Level Grounding Summary (recommended_deterministic)

| arg_field | rows_with_field | det_resolved | exact_match | resolve_rate | match_rate |
|---|---:|---:|---:|---:|---:|
| `order_id` | 234 | 141 | 78 | 60% | 33% |
| `product_id` | 64 | 62 | 18 | 97% | 28% |
| `user_id` | 108 | 108 | 108 | 100% | 100% |
| `payment_method_id` | 86 | 86 | 71 | 100% | 83% |
| `item_ids` | 85 | 0 | 0 | 0% | 0% |
| `new_item_ids` | 55 | 0 | 0 | 0% | 0% |