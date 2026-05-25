# Turn-Level Graph Evaluation Report

_Generated: 2026-05-25 19:31 UTC_

**Source dataset**: `data/out/turn_supervision.jsonl`  
**Rows evaluated**: 30

## Comparison Table

| Graph | Nodes | E2E Success | Tool Acc | Arg Match | Policy Viol | Failures | Grounding |
|---|---:|---:|---:|---:|---:|---:|---|
| `monolithic` | 2 | 0% | 0% | 0% | 0% | 30 | n/a |
| `minimal` | 3 | 0% | 0% | 0% | 0% | 30 | n/a |
| `recommended_stub` | 7 | 13% | 13% | 0% | 0% | 30 | stub |
| `recommended_oracle` | 7 | 100% | 100% | 96% | 0% | 30 | oracle |