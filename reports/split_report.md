# Split Report

> **Descriptive split validation artifact.**  
> Not an input to graph recommendation.  
> `experimental_boundary.artifact_type = descriptive` | `data_scope = all_splits` | `allowed_to_influence_graph = false`

**Version:** `scenario_stratified_grounding_v1`  
**Seed:** 42  
**Strategy:** primary = ['scenario_type', 'requires_grounding']  
**Ratio:** train 0.6 / dev 0.2 / test 0.2

## Split Sizes

| split | simulations | turns |
| ----- | ----------- | ----- |
| train | 62          | 62    |
| dev   | 20          | 20    |
| test  | 18          | 18    |
| total | 100         | 100   |

## Scenario Type

| scenario_type                              | overall  | train | dev | test |
| ------------------------------------------ | -------- | ----- | --- | ---- |
| calculate|multi_action                     | 1 (1%)   | 1     | 0   | 0    |
| cancel|multi_action                        | 4 (4%)   | 2     | 1   | 1    |
| cancel|single_action                       | 6 (6%)   | 4     | 1   | 1    |
| exchange|multi_action                      | 3 (3%)   | 2     | 1   | 0    |
| exchange|single_action                     | 19 (19%) | 11    | 4   | 4    |
| lookup_only|single_action                  | 7 (7%)   | 5     | 1   | 1    |
| modify_address|multi_action                | 2 (2%)   | 2     | 0   | 0    |
| modify_address|single_action               | 2 (2%)   | 2     | 0   | 0    |
| modify_order_items|multi_action            | 14 (14%) | 8     | 3   | 3    |
| modify_order_items|single_action           | 10 (10%) | 6     | 2   | 2    |
| modify_pending_order_payment|single_action | 1 (1%)   | 1     | 0   | 0    |
| return|multi_action                        | 7 (7%)   | 4     | 1   | 2    |
| return|single_action                       | 15 (15%) | 9     | 3   | 3    |
| transfer|multi_action                      | 1 (1%)   | 1     | 0   | 0    |
| transfer|single_action                     | 3 (3%)   | 0     | 2   | 1    |
| update_address|multi_action                | 2 (2%)   | 2     | 0   | 0    |
| update_address|single_action               | 3 (3%)   | 2     | 1   | 0    |

## Primary Scenario

| primary_scenario             | overall  | train | dev | test |
| ---------------------------- | -------- | ----- | --- | ---- |
| calculate                    | 1 (1%)   | 1     | 0   | 0    |
| cancel                       | 10 (10%) | 6     | 2   | 2    |
| exchange                     | 22 (22%) | 13    | 5   | 4    |
| lookup_only                  | 7 (7%)   | 5     | 1   | 1    |
| modify_address               | 4 (4%)   | 4     | 0   | 0    |
| modify_order_items           | 24 (24%) | 14    | 5   | 5    |
| modify_pending_order_payment | 1 (1%)   | 1     | 0   | 0    |
| return                       | 22 (22%) | 13    | 4   | 5    |
| transfer                     | 4 (4%)   | 1     | 2   | 1    |
| update_address               | 5 (5%)   | 4     | 1   | 0    |

## Difficulty Bucket

| difficulty_bucket | overall  | train | dev | test |
| ----------------- | -------- | ----- | --- | ---- |
| easy              | 3 (3%)   | 2     | 0   | 1    |
| hard              | 74 (74%) | 44    | 15  | 15   |
| medium            | 23 (23%) | 16    | 5   | 2    |

## Requires Grounding

| requires_grounding | overall  | train | dev | test |
| ------------------ | -------- | ----- | --- | ---- |
| False              | 3 (3%)   | 2     | 0   | 1    |
| True               | 97 (97%) | 60    | 20  | 17   |

## Has item_ids / new_item_ids

| has_item_ids | overall  | train | dev | test |
| ------------ | -------- | ----- | --- | ---- |
| False        | 26 (26%) | 18    | 5   | 3    |
| True         | 74 (74%) | 44    | 15  | 15   |

## Has order_id

| has_order_id | overall  | train | dev | test |
| ------------ | -------- | ----- | --- | ---- |
| False        | 3 (3%)   | 2     | 0   | 1    |
| True         | 97 (97%) | 60    | 20  | 17   |

## Has product_id

| has_product_id | overall  | train | dev | test |
| -------------- | -------- | ----- | --- | ---- |
| False          | 66 (66%) | 42    | 14  | 10   |
| True           | 34 (34%) | 20    | 6   | 8    |

## Requires Tool Chaining

| requires_tool_chaining | overall  | train | dev | test |
| ---------------------- | -------- | ----- | --- | ---- |
| False                  | 31 (31%) | 21    | 5   | 5    |
| True                   | 69 (69%) | 41    | 15  | 13   |

## Is Multi-Action

| is_multi_action | overall  | train | dev | test |
| --------------- | -------- | ----- | --- | ---- |
| False           | 66 (66%) | 40    | 14  | 12   |
| True            | 34 (34%) | 22    | 6   | 6    |

## Terminal Tool Fingerprint

| terminal_tool_fingerprint                                                    | overall  | train | dev | test |
| ---------------------------------------------------------------------------- | -------- | ----- | --- | ---- |
| calculate+cancel_pending_order                                               | 1 (1%)   | 1     | 0   | 0    |
| calculate+cancel_pending_order+return_delivered_order_items                  | 1 (1%)   | 0     | 0   | 1    |
| calculate+exchange_delivered_order_items                                     | 2 (2%)   | 2     | 0   | 0    |
| calculate+get_item_details+modify_pending_order_items                        | 1 (1%)   | 1     | 0   | 0    |
| calculate+modify_pending_order_items                                         | 3 (3%)   | 1     | 1   | 1    |
| calculate+return_delivered_order_items                                       | 3 (3%)   | 3     | 0   | 0    |
| cancel_pending_order                                                         | 6 (6%)   | 4     | 1   | 1    |
| cancel_pending_order+exchange_delivered_order_items                          | 1 (1%)   | 0     | 1   | 0    |
| cancel_pending_order+modify_pending_order_address                            | 1 (1%)   | 1     | 0   | 0    |
| cancel_pending_order+modify_pending_order_address+modify_pending_order_items | 1 (1%)   | 1     | 0   | 0    |
| cancel_pending_order+modify_pending_order_items                              | 1 (1%)   | 0     | 1   | 0    |
| cancel_pending_order+return_delivered_order_items                            | 5 (5%)   | 2     | 1   | 2    |
| exchange_delivered_order_items                                               | 19 (19%) | 11    | 4   | 4    |
| exchange_delivered_order_items+modify_pending_order_items                    | 2 (2%)   | 1     | 1   | 0    |
| exchange_delivered_order_items+return_delivered_order_items                  | 1 (1%)   | 0     | 1   | 0    |
| modify_pending_order_address                                                 | 2 (2%)   | 2     | 0   | 0    |
| modify_pending_order_address+modify_pending_order_items                      | 4 (4%)   | 4     | 0   | 0    |
| modify_pending_order_address+modify_pending_order_items+modify_user_address  | 2 (2%)   | 1     | 0   | 1    |
| modify_pending_order_address+modify_user_address                             | 2 (2%)   | 2     | 0   | 0    |
| modify_pending_order_items                                                   | 10 (10%) | 6     | 2   | 2    |
| modify_pending_order_items+modify_user_address                               | 1 (1%)   | 1     | 0   | 0    |
| modify_pending_order_items+return_delivered_order_items                      | 1 (1%)   | 0     | 0   | 1    |
| modify_pending_order_payment                                                 | 1 (1%)   | 1     | 0   | 0    |
| modify_user_address                                                          | 3 (3%)   | 2     | 1   | 0    |
| none                                                                         | 7 (7%)   | 5     | 1   | 1    |
| return_delivered_order_items                                                 | 15 (15%) | 9     | 3   | 3    |
| return_delivered_order_items+transfer_to_human_agents                        | 1 (1%)   | 1     | 0   | 0    |
| transfer_to_human_agents                                                     | 3 (3%)   | 0     | 2   | 1    |

## Warnings

- WARNING: `train` split has zero simulations for scenario_type `transfer|single_action` (present in overall)
- WARNING: `dev` split has zero simulations for scenario_type `calculate|multi_action` (present in overall)
- WARNING: `dev` split has zero simulations for scenario_type `modify_address|multi_action` (present in overall)
- WARNING: `dev` split has zero simulations for scenario_type `modify_address|single_action` (present in overall)
- WARNING: `dev` split has zero simulations for scenario_type `modify_pending_order_payment|single_action` (present in overall)
- WARNING: `dev` split has zero simulations for scenario_type `transfer|multi_action` (present in overall)
- WARNING: `dev` split has zero simulations for scenario_type `update_address|multi_action` (present in overall)
- WARNING: `test` split has zero simulations for scenario_type `calculate|multi_action` (present in overall)
- WARNING: `test` split has zero simulations for scenario_type `exchange|multi_action` (present in overall)
- WARNING: `test` split has zero simulations for scenario_type `modify_address|multi_action` (present in overall)
- WARNING: `test` split has zero simulations for scenario_type `modify_address|single_action` (present in overall)
- WARNING: `test` split has zero simulations for scenario_type `modify_pending_order_payment|single_action` (present in overall)
- WARNING: `test` split has zero simulations for scenario_type `transfer|multi_action` (present in overall)
- WARNING: `test` split has zero simulations for scenario_type `update_address|multi_action` (present in overall)
- WARNING: `test` split has zero simulations for scenario_type `update_address|single_action` (present in overall)
- WARNING: terminal_tool_fingerprint `calculate+cancel_pending_order` appears only in train (not in dev or test)
- WARNING: terminal_tool_fingerprint `calculate+exchange_delivered_order_items` appears only in train (not in dev or test)
- WARNING: terminal_tool_fingerprint `calculate+get_item_details+modify_pending_order_items` appears only in train (not in dev or test)
- WARNING: terminal_tool_fingerprint `calculate+return_delivered_order_items` appears only in train (not in dev or test)
- WARNING: terminal_tool_fingerprint `cancel_pending_order+modify_pending_order_address` appears only in train (not in dev or test)
- WARNING: terminal_tool_fingerprint `cancel_pending_order+modify_pending_order_address+modify_pending_order_items` appears only in train (not in dev or test)
- WARNING: terminal_tool_fingerprint `modify_pending_order_address` appears only in train (not in dev or test)
- WARNING: terminal_tool_fingerprint `modify_pending_order_address+modify_pending_order_items` appears only in train (not in dev or test)
- WARNING: terminal_tool_fingerprint `modify_pending_order_address+modify_user_address` appears only in train (not in dev or test)
- WARNING: terminal_tool_fingerprint `modify_pending_order_items+modify_user_address` appears only in train (not in dev or test)
- WARNING: terminal_tool_fingerprint `modify_pending_order_payment` appears only in train (not in dev or test)
- WARNING: terminal_tool_fingerprint `return_delivered_order_items+transfer_to_human_agents` appears only in train (not in dev or test)
