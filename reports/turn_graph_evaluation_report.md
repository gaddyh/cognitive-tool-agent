# Turn-Level Graph Evaluation Report

_Generated: 2026-05-25 23:42 UTC_

**Source dataset**: `data/out/splits/test_supervision.jsonl`  
**Rows evaluated**: 92

## Comparison Table

| Graph | Nodes | E2E Success | Tool Acc | Arg Match | Policy Viol | Failures | Grounding |
|---|---:|---:|---:|---:|---:|---:|---|
| `monolithic` | 2 | 0% | 0% | 0% | 0% | 92 | n/a |
| `minimal` | 3 | 0% | 0% | 0% | 0% | 92 | n/a |
| `recommended_stub` | 7 | 2% | 2% | 0% | 0% | 92 | stub |
| `recommended_deterministic` | 7 | 37% | 37% | 24% | 0% | 92 | deterministic |
| `recommended_oracle` | 7 | 100% | 100% | 100% | 0% | 92 | oracle |

## Field-Level Grounding Summary (recommended_deterministic)

| arg_field | rows_with_field | det_resolved | exact_match | resolve_rate | match_rate |
|---|---:|---:|---:|---:|---:|
| `order_id` | 39 | 22 | 14 | 56% | 36% |
| `product_id` | 12 | 11 | 4 | 92% | 33% |
| `user_id` | 17 | 17 | 17 | 100% | 100% |
| `payment_method_id` | 15 | 15 | 13 | 100% | 87% |
| `item_ids` | 15 | 0 | 0 | 0% | 0% |
| `new_item_ids` | 8 | 0 | 0 | 0% | 0% |