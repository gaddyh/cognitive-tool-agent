# Cognitive Dataset Report

_Generated: 2026-05-25 19:04 UTC — Source: `baseline_retail_100`_

## Dataset Summary

| Metric | Value |
|---|---:|
| Tasks | 100 |
| Simulations | 100 |
| Messages | 2714 |
| Expected actions | 514 |
| Actual tool calls | 797 |
| Matched actions | 427 |
| Failed actions | 60 |
| **Tool entropy** | **3.063 bits** |
| **Avg tools / simulation** | **7.97** |
| **Avg turns before write** | **21.1** |
| Read tool calls | 626 |
| Write tool calls | 151 |
| **Read / write ratio** | **4.15** |

## Cognitive Burden Breakdown

Tools ranked by complexity score (descending).

| Tool | Type | Args | Extract | Memory | Readiness | Reasoning | Grounding | Score |
|---|---|---:|---|---|---|---|---|---:|
| `modify_user_address` | write | 7 | very_high | very_high | high | high | high | **21** |
| `modify_pending_order_address` | write | 7 | very_high | very_high | high | medium | high | **20** |
| `exchange_delivered_order_items` | write | 4 | very_high | very_high | high | medium | very_high | **17** |
| `modify_pending_order_items` | write | 4 | very_high | very_high | high | high | very_high | **17** |
| `return_delivered_order_items` | write | 3 | high | very_high | high | high | very_high | **14** |
| `modify_pending_order_payment` | write | 2 | medium | very_high | high | medium | very_high | **13** |
| `cancel_pending_order` | write | 2 | medium | very_high | high | high | high | **12** |
| `get_order_details` | read | 1 | low | very_high | low | high | very_high | **8** |
| `get_product_details` | read | 1 | low | very_high | low | high | very_high | **8** |
| `transfer_to_human_agents` | generic | 1 | low | very_high | low | high | very_high | **8** |
| `calculate` | generic | 1 | low | very_high | low | high | very_high | **8** |
| `get_user_details` | read | 1 | low | very_high | low | medium | very_high | **6** |
| `get_item_details` | read | 1 | low | medium | low | very_high | low | **5** |
| `find_user_id_by_name_zip` | read | 3 | high | low | low | low | low | **3** |
| `find_user_id_by_email` | read | 1 | low | low | low | low | low | **1** |
| `list_all_product_types` | unknown | 0 | low | low | low | low | low | **0** |

### Burden signal legend

| Signal | Meaning |
|---|---|
| extraction burden | number of required arguments |
| memory burden | avg turns between user hint and tool call |
| readiness burden | write tools require explicit confirmation |
| reasoning burden | depth of preceding tool calls (chain length) |
| grounding burden | fraction of args not explicitly provided by user |

## Argument Emergence Matrix

How each required argument reaches the agent — as a percentage of all instances.

| Argument | Instances | Explicit % | Tool-Chained % | Grounding % | Inference % |
|---|---:|---:|---:|---:|---:|
| `address1` | 20 | 25.0% | 60.0% | 10.0% | 5.0% |
| `address2` | 20 | 30.0% | 60.0% | 10.0% | 0.0% |
| `city` | 20 | 30.0% | 65.0% | 5.0% | 0.0% |
| `country` | 20 | 30.0% | 70.0% | 0.0% | 0.0% |
| `email` | 14 | 85.7% | 0.0% | 14.3% | 0.0% |
| `expression` | 3 | 0.0% | 0.0% | 0.0% | 100.0% |
| `first_name` | 60 | 100.0% | 0.0% | 0.0% | 0.0% |
| `item_id` | 2 | 100.0% | 0.0% | 0.0% | 0.0% |
| `item_ids` | 88 | 0.0% | 0.0% | 85.2% | 14.8% |
| `last_name` | 60 | 100.0% | 0.0% | 0.0% | 0.0% |
| `new_item_ids` | 55 | 0.0% | 0.0% | 83.6% | 16.4% |
| `order_id` | 277 | 9.7% | 89.9% | 0.0% | 0.4% |
| `payment_method_id` | 89 | 0.0% | 100.0% | 0.0% | 0.0% |
| `product_id` | 42 | 0.0% | 100.0% | 0.0% | 0.0% |
| `reason` | 23 | 34.8% | 4.3% | 47.8% | 13.0% |
| `state` | 20 | 15.0% | 75.0% | 5.0% | 5.0% |
| `summary` | 2 | 0.0% | 0.0% | 100.0% | 0.0% |
| `user_id` | 63 | 0.0% | 100.0% | 0.0% | 0.0% |
| `zip` | 80 | 86.2% | 10.0% | 3.8% | 0.0% |

> **Explicit** — value appeared verbatim in user message  
> **Tool-Chained** — value came from a preceding tool result  
> **Grounding** — action matched but not directly extractable (NL→structured)  
> **Inference** — not resolved (failed actions or ambiguous)

## Failure Heatmap

### By Tool

| Value | Failures |
|---|---:|
| `calculate` | 13 |
| `get_product_details` | 9 |
| `get_order_details` | 9 |
| `modify_pending_order_address` | 6 |
| `return_delivered_order_items` | 5 |
| `exchange_delivered_order_items` | 5 |
| `modify_pending_order_items` | 5 |
| `cancel_pending_order` | 3 |
| `transfer_to_human_agents` | 2 |
| `modify_user_address` | 2 |
| `find_user_id_by_name_zip` | 1 |

### By Cognitive Stage

| Value | Failures |
|---|---:|
| `action` | 24 |
| `lookup` | 18 |
| `reasoning` | 13 |
| `escalation` | 2 |
| `unknown` | 2 |
| `auth` | 1 |

### By Read / Write

| Value | Failures |
|---|---:|
| `write` | 26 |
| `read` | 19 |
| `generic` | 15 |

### By Argument

| Value | Failures |
|---|---:|
| `order_id` | 20 |
| `expression` | 13 |
| `item_ids` | 11 |
| `product_id` | 9 |
| `new_item_ids` | 7 |
| `state` | 7 |
| `zip` | 5 |
| `payment_method_id` | 4 |
| `address1` | 4 |
| `address2` | 4 |
| `city` | 4 |
| `country` | 4 |
| `summary` | 2 |
| `reason` | 2 |
| `user_id` | 1 |

## Cognitive Complexity Score Ranking

```
score = required_args_count
      + memory_burden_score  (low=0 medium=1 high=2 very_high=3)
      + write_penalty        (write=2, else 0)
      + grounding_penalty    (grounding_fraction × n_args, rounded)
      + confirmation_penalty (write=1, else 0)
      + chain_depth_score    (min(avg_chain_depth, 3))
```

| Rank | Tool | Score |
|---:|---|---:|
| 1 | `modify_user_address` | 21 |
| 2 | `modify_pending_order_address` | 20 |
| 3 | `exchange_delivered_order_items` | 17 |
| 4 | `modify_pending_order_items` | 17 |
| 5 | `return_delivered_order_items` | 14 |
| 6 | `modify_pending_order_payment` | 13 |
| 7 | `cancel_pending_order` | 12 |
| 8 | `get_order_details` | 8 |
| 9 | `get_product_details` | 8 |
| 10 | `transfer_to_human_agents` | 8 |
| 11 | `calculate` | 8 |
| 12 | `get_user_details` | 6 |
| 13 | `get_item_details` | 5 |
| 14 | `find_user_id_by_name_zip` | 3 |
| 15 | `find_user_id_by_email` | 1 |
| 16 | `list_all_product_types` | 0 |
