# Cognitive Dataset Report

_Generated: 2026-05-26 00:00 UTC — Source: `data/out`_

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
| **Avg turns before write** | **21.4** |
| Read tool calls | 626 |
| Write tool calls | 151 |
| **Read / write ratio** | **4.15** |

## Cognitive Burden Breakdown

Tools ranked by complexity score (descending).

| Tool | Type | Args | Extract | Memory | Readiness | Reasoning | Grounding | Score |
|---|---|---:|---|---|---|---|---|---:|
| `modify_user_address` | write | 7 | very_high | very_high | high | high | high | **21** |
| `modify_pending_order_address` | write | 7 | very_high | very_high | high | medium | high | **20** |
| `exchange_delivered_order_items` | write | 4 | very_high | very_high | high | high | very_high | **17** |
| `modify_pending_order_items` | write | 4 | very_high | very_high | high | high | very_high | **17** |
| `return_delivered_order_items` | write | 3 | high | very_high | high | high | very_high | **14** |
| `cancel_pending_order` | write | 2 | medium | very_high | high | medium | very_high | **13** |
| `modify_pending_order_payment` | write | 2 | medium | very_high | high | medium | very_high | **13** |
| `get_order_details` | read | 1 | low | very_high | low | high | very_high | **8** |
| `get_product_details` | read | 1 | low | very_high | low | high | very_high | **8** |
| `calculate` | generic | 1 | low | very_high | low | high | very_high | **8** |
| `get_user_details` | read | 1 | low | very_high | low | medium | very_high | **6** |
| `get_item_details` | read | 1 | low | medium | low | very_high | low | **5** |
| `find_user_id_by_name_zip` | read | 3 | high | low | low | low | low | **3** |
| `find_user_id_by_email` | read | 1 | low | low | low | low | low | **1** |
| `transfer_to_human_agents` | generic | 1 | low | low | low | low | low | **1** |
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
| `address1` | 17 | 29.4% | 64.7% | 5.9% | 0.0% |
| `address2` | 17 | 35.3% | 52.9% | 11.8% | 0.0% |
| `city` | 17 | 29.4% | 64.7% | 5.9% | 0.0% |
| `country` | 17 | 35.3% | 64.7% | 0.0% | 0.0% |
| `email` | 4 | 100.0% | 0.0% | 0.0% | 0.0% |
| `expression` | 3 | 0.0% | 0.0% | 0.0% | 100.0% |
| `first_name` | 39 | 100.0% | 0.0% | 0.0% | 0.0% |
| `item_id` | 2 | 100.0% | 0.0% | 0.0% | 0.0% |
| `item_ids` | 48 | 0.0% | 0.0% | 83.3% | 16.7% |
| `last_name` | 39 | 100.0% | 0.0% | 0.0% | 0.0% |
| `new_item_ids` | 30 | 0.0% | 0.0% | 86.7% | 13.3% |
| `order_id` | 156 | 7.7% | 91.7% | 0.0% | 0.6% |
| `payment_method_id` | 49 | 0.0% | 100.0% | 0.0% | 0.0% |
| `product_id` | 24 | 0.0% | 100.0% | 0.0% | 0.0% |
| `reason` | 12 | 25.0% | 8.3% | 41.7% | 25.0% |
| `state` | 17 | 11.8% | 76.5% | 5.9% | 5.9% |
| `user_id` | 38 | 0.0% | 100.0% | 0.0% | 0.0% |
| `zip` | 56 | 82.1% | 14.3% | 3.6% | 0.0% |

> **Explicit** — value appeared verbatim in user message  
> **Tool-Chained** — value came from a preceding tool result  
> **Grounding** — action matched but not directly extractable (NL→structured)  
> **Inference** — not resolved (failed actions or ambiguous)

## Failure Heatmap

### By Tool

| Value | Failures |
|---|---:|
| `calculate` | 10 |
| `get_order_details` | 9 |
| `get_product_details` | 6 |
| `return_delivered_order_items` | 5 |
| `modify_pending_order_address` | 5 |
| `modify_pending_order_items` | 3 |
| `cancel_pending_order` | 3 |
| `modify_user_address` | 2 |
| `exchange_delivered_order_items` | 2 |
| `transfer_to_human_agents` | 1 |
| `find_user_id_by_name_zip` | 1 |

### By Cognitive Stage

| Value | Failures |
|---|---:|
| `action` | 18 |
| `lookup` | 15 |
| `reasoning` | 10 |
| `unknown` | 2 |
| `escalation` | 1 |
| `auth` | 1 |

### By Read / Write

| Value | Failures |
|---|---:|
| `write` | 20 |
| `read` | 16 |
| `generic` | 11 |

### By Argument

| Value | Failures |
|---|---:|
| `order_id` | 17 |
| `expression` | 10 |
| `item_ids` | 9 |
| `state` | 7 |
| `product_id` | 6 |
| `zip` | 5 |
| `new_item_ids` | 4 |
| `address1` | 4 |
| `address2` | 4 |
| `city` | 4 |
| `country` | 4 |
| `payment_method_id` | 3 |
| `reason` | 2 |
| `summary` | 1 |
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
| 6 | `cancel_pending_order` | 13 |
| 7 | `modify_pending_order_payment` | 13 |
| 8 | `get_order_details` | 8 |
| 9 | `get_product_details` | 8 |
| 10 | `calculate` | 8 |
| 11 | `get_user_details` | 6 |
| 12 | `get_item_details` | 5 |
| 13 | `find_user_id_by_name_zip` | 3 |
| 14 | `find_user_id_by_email` | 1 |
| 15 | `transfer_to_human_agents` | 1 |
| 16 | `list_all_product_types` | 0 |
