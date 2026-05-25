# Cognitive Tool Agent

A dataset-to-agent-architecture lab for evaluation-driven tool-calling systems.

This repo explores a concrete thesis:

> Tool-agent failures are not one blob.  
> They can be decomposed into measurable cognitive spaces, and the right graph should be justified by dataset evidence, not architectural taste.

The current focus is tau2-style retail tool-agent traces. The repo converts raw simulations into cognitive artifacts, infers capability pressure, recommends a graph, and evaluates graph variants against turn-level tool-call decisions.

---

## Current status

The repo now has a working evaluation loop:

```text
tau2 results.json
   ↓
trace conversion
   ↓
cognitive dataset artifacts
   ↓
dataset reports + capability inference
   ↓
recommended cognitive graph
   ↓
turn-level graph evaluation
   ↓
stub vs deterministic vs oracle comparison
```

The most important current result:

```text
recommended_stub           2% E2E /  0% argument match
recommended_deterministic 39% E2E / 23% argument match
recommended_oracle       100% E2E / 99% argument match
```

This means the system now has a measurable grounding gap:

```text
stub < deterministic < oracle
```

The deterministic grounding node closes a real part of the oracle gap without using labels.

---

## Why this repo exists

Most agent development looks like this:

```text
prompt → run → fail → tweak prompt → run again
```

This repo is moving toward:

```text
dataset → metrics → baseline → failure analysis → capability isolation → graph revision → reevaluation
```

The goal is not to build another generic agent framework.

The goal is to make agent architecture measurable.

---

## Core thesis

A tool-calling assistant often collapses many different cognitive responsibilities into one prompt:

```text
perceive user intent
select tool
ground entities
fill arguments
check readiness
plan next action
execute
learn/update state
```

When the agent fails, a single E2E score does not explain why.

This repo decomposes the problem into stages:

```text
perceive → reason → grounding → readiness → plan → act → learn
```

Each stage has typed inputs and outputs. Each stage can be stubbed, implemented deterministically, backed by an LLM, or treated as an oracle ceiling.

---

## What has been proven so far

### 1. Trace conversion works

The tau2 retail run is converted into cognitive artifacts.

Current conversion summary:

| Metric | Value |
|---|---:|
| Tasks | 100 |
| Simulations | 100 |
| Messages | 2,714 |
| Expected actions | 514 |
| Actual tool calls | 797 |
| Matched actions | 427 |
| Failed actions | 60 |

Generated artifacts:

```text
data/out/tool_registry.json
data/out/action_sequence.jsonl
data/out/turn_supervision.jsonl
data/out/failure_rows.jsonl
data/out/conversion_summary.json
```

This proves the raw tau2 traces contain enough structure to extract action-level, turn-level, and failure-level supervision.

---

### 2. The dataset exposes capability pressure

The report builder computes cognitive pressure signals from the converted artifacts.

Current extended dataset summary:

| Metric | Value |
|---|---:|
| Tool entropy | 3.063 bits |
| Avg tools / simulation | 7.97 |
| Avg turns before write | 21.1 |
| Read / write ratio | 4.15 |

Top complexity tools include:

| Tool | Type | Complexity |
|---|---|---:|
| modify_user_address | write | 21 |
| modify_pending_order_address | write | 20 |
| exchange_delivered_order_items | write | 17 |
| modify_pending_order_items | write | 17 |
| return_delivered_order_items | write | 14 |
| modify_pending_order_payment | write | 13 |
| cancel_pending_order | write | 12 |
| get_order_details | read | 8 |

Top argument failure fields:

| Argument | Failure count |
|---|---:|
| order_id | 20 |
| expression | 13 |
| item_ids | 11 |
| product_id | 9 |
| new_item_ids | 7 |
| state | 7 |
| zip | 5 |
| payment_method_id | 4 |

---

### 3. Capability inference recommends the full cognitive graph

The recommender infers these capability requirements:

| Capability | Required | Strength | Evidence |
|---|---:|---:|---|
| memory | yes | 0.54 | `order_id` values are heavily tool-chained |
| grounding | yes | 0.85 | `item_ids` has very high peak grounding pressure |
| readiness | yes | 0.43 | many failures involve write actions |
| deep_planning | yes | 0.61 | average chain depth is above threshold |

Recommended graph:

```text
perceive → reason → grounding → readiness → plan → act → learn
```

This graph is not chosen by taste. It is justified by dataset signals.

---

## Why the original graph evaluation was all zero

The first graph evaluation compressed each full simulation into one row:

```text
first user message → selected primary / final-ish action
```

That was misaligned.

A typical simulation has many turns and many tool calls. The report showed an average of 21.1 turns before write actions. Asking weak stubs to infer a late write action from the first user message produced all zeros.

This was not a graph failure. It was an evaluation-unit mismatch.

---

## Turn-level evaluation

The fix was to evaluate the graph at the correct unit:

```text
local turn context → next assistant tool call
```

The new turn-level adapter builds one row per `call_tool` turn from `turn_supervision.jsonl`.

Each row contains:

```text
user_message
expected_tool
expected_arguments
prior_tool_calls
prior_tool_results
conversation_context
tool registry
```

This creates 547 turn-level tool-call decision rows.

Example rows:

| Row | Expected tool | Expected args |
|---|---|---|
| turn 4 | find_user_id_by_email | email |
| turn 6 | get_user_details | user_id |
| turn 10 | get_order_details | order_id |
| turn 14 | list_all_product_types | `{}` |
| turn 16 | get_product_details | product_id |

---

## Current graph evaluation results

Full 547-row turn-level evaluation:

| Graph | Nodes | E2E Success | Tool Acc | Arg Match | Grounding |
|---|---:|---:|---:|---:|---|
| monolithic | 2 | 0% | 0% | 0% | n/a |
| minimal | 3 | 0% | 0% | 0% | n/a |
| recommended_stub | 7 | 2% | 2% | 0% | stub |
| recommended_deterministic | 7 | 39% | 39% | 23% | deterministic |
| recommended_oracle | 7 | 100% | 100% | 99% | oracle |

This is the first key proof result:

```text
A dedicated deterministic grounding node improves the graph from 2% to 39% E2E success without using labels.
```

---

## Field-level grounding results

Field-level summary for `recommended_deterministic`:

| Argument field | Rows with field | Deterministic resolved | Exact match | Resolve rate | Match rate |
|---|---:|---:|---:|---:|---:|
| order_id | 234 | 141 | 78 | 60% | 33% |
| product_id | 64 | 62 | 18 | 97% | 28% |
| user_id | 108 | 108 | 108 | 100% | 100% |
| payment_method_id | 86 | 86 | 71 | 100% | 83% |
| item_ids | 85 | 0 | 0 | 0% | 0% |
| new_item_ids | 55 | 0 | 0 | 0% | 0% |

Interpretation:

```text
Solved:
- user_id grounding
- much of payment_method_id grounding

Partially solved:
- order_id grounding
- product_id grounding

Not solved yet:
- item_ids
- new_item_ids
```

This is the point of the architecture: failures are no longer opaque. They are localized by field and capability.

---

## Grounding modes

The grounding node currently supports several modes:

| Mode | Meaning |
|---|---|
| stub | Returns little/no resolved grounding. Baseline floor. |
| deterministic | Uses prior tool calls, prior tool results, and explicit text patterns. No labels. |
| oracle | Copies expected arguments from the dataset row. Ceiling only. |
| llm | Reserved for future model-backed grounding. |
| disabled | Used when grounding is intentionally removed. |

The oracle is not a production solution. It is a measurement instrument.

It answers:

```text
If grounding were perfect, could the rest of the graph use it?
```

The answer is now yes.

---

## Deterministic grounding

Deterministic grounding is a conservative schema-aware lookup layer.

It receives:

```text
selected_tool
selected_tool.required_fields
user_message
prior_tool_calls
prior_tool_results
conversation_context
```

It outputs:

```text
resolved_args
unresolved_fields
```

It is forbidden from using:

```text
row.expected.expected_arguments
row.expected.expected_tool
```

Current Pass 1 resolves scalar IDs:

| Field | Strategy |
|---|---|
| order_id | prior tool call args, prior tool result JSON, `#W...` regex |
| user_id | prior tool call args, prior tool result JSON, user-id-like result strings |
| product_id | prior tool call args, prior tool result JSON, conservative product type matching |
| payment_method_id | prior tool call args, prior tool result JSON |
| generic `*_id` | same-key lookup in prior calls/results |

Deferred to Pass 2:

```text
item_ids
new_item_ids
```

These require item-list and product-variant grounding, not simple scalar lookup.

---

## Architecture

Current package layout:

```text
src/cognitive_tool_agent/
  adapters/
  agents/
  datasets/
  evals/
  graph/
  graph_builder/
  graph_runner/
  recommender/
  reports/
  schemas/
  tools/
  trace_converter/
```

Important modules:

| Module | Responsibility |
|---|---|
| `trace_converter` | Converts raw tau2 traces into cognitive artifacts |
| `reports` | Builds dataset reports, topology reports, failure heatmaps |
| `recommender` | Infers required capabilities and recommends graph topology |
| `graph` | Executes cognitive graphs with edge-driven dispatch |
| `agents` | Implements cognitive stage stubs, deterministic logic, and oracle modes |
| `graph_runner` | Evaluates graph variants over adapted datasets |
| `evals` | Scores traces with strict metrics |

---

## Graph execution model

The executor is edge-driven.

Each role writes one canonical output slot:

| Role | Output slot |
|---|---|
| perceive | perception |
| reason | reasoning |
| grounding | grounding |
| readiness | readiness |
| plan | plan |
| act | action |
| learn | learning |

Each node receives a `NodeInput` containing only the upstream slots allowed by `ROLE_INPUTS`.

This makes graph wiring testable. A node cannot silently consume arbitrary context unless the graph declares that dependency.

---

## Current graph variants

The turn-level evaluator compares five graph configurations:

| Graph | Shape / mode |
|---|---|
| monolithic | `plan → act` |
| minimal | `perceive → plan → act` |
| recommended_stub | full recommended graph with stub grounding |
| recommended_deterministic | full recommended graph with deterministic grounding |
| recommended_oracle | full recommended graph with oracle grounding |

The current evidence shows that the full graph only becomes useful when the grounding node is capable.

---

## Running the full pipeline

From repo root:

```bash
python scripts/convert_traces.py \
  --input data/raw/simulations/baseline_retail_100/results.json \
  --out-dir data/out
```

```bash
python scripts/build_reports.py \
  --out-dir data/out \
  --reports-dir reports \
  --source baseline_retail_100
```

```bash
python scripts/recommend_graph.py \
  --report reports/cognitive_dataset_report.json \
  --out reports/recommended_graph.json
```

```bash
python scripts/run_turn_graph_evaluation.py
```

Optional limited run:

```bash
python scripts/run_turn_graph_evaluation.py --limit 20
```

Explain one row:

```bash
python scripts/explain_turn_graph_row.py \
  --row-index 0 \
  --graph recommended_deterministic
```

Other graph options:

```text
monolithic
minimal
recommended_stub
recommended_deterministic
recommended_oracle
```

---

## Legacy action-sequence evaluation

There is also an action-sequence graph evaluation path:

```bash
python scripts/run_graph_evaluation.py
```

This path evaluates:

```text
first user message → selected primary action
```

It is useful as a stress test, but it is currently too misaligned for the stub/deterministic graph and tends to produce all-zero results.

The turn-level evaluator is the current primary evaluation path.

---

## Development workflow

Recommended workflow:

```text
1. Convert traces
2. Build cognitive reports
3. Recommend graph
4. Run turn-level graph evaluation
5. Inspect field-level failures
6. Add deterministic or model-backed capability
7. Reevaluate
```

Do not optimize prompts before the dataset, metrics, and baseline are stable.

---

## Tests

The repo includes tests for:

```text
schema validation
trace conversion
dataset profiling
graph execution
edge-driven wiring
graph recommender
turn-level adapter
deterministic grounding
graph evaluation
revision advice
```

Current milestone acceptance criteria:

```text
all existing tests pass
turn-level graph evaluation emits five graph rows
recommended_deterministic beats recommended_stub
field-level grounding report is produced
deterministic grounding does not use expected arguments
```

---

## Current limitations

The current result is strong but scoped.

Known limitations:

```text
- deterministic grounding is Pass 1 only
- item_ids and new_item_ids are unresolved
- product_id grounding is over-eager and often wrong
- order_id grounding needs safer disambiguation
- graph evaluation is turn-level, not full tau2 interactive replay
- no LLM-backed node is implemented yet
- no DSPy optimization loop yet
```

---

## Next milestones

### Milestone 2.1 — Stabilize scalar grounding

Improve fields already attempted:

```text
order_id
product_id
payment_method_id
```

Targets:

```text
order_id exact match > 33%
product_id exact match > 28%
deterministic E2E >= 39%
wrong-resolution rate decreases
```

Focus:

```text
prefer unresolved over wrong
handle multiple candidate IDs conservatively
use immediate/local context before broad history
```

### Milestone 3 — Item-list grounding

Add a dedicated item grounding capability:

```text
item_ids
new_item_ids
```

This requires resolving user references against order details and product details.

Expected inputs:

```text
user message
prior order details
prior product details
item names
colors/sizes
quantities
replacement intent
```

### Milestone 4 — LLM grounding

Only after deterministic baselines and oracle gaps are clear:

```text
stub → deterministic → llm → oracle
```

The goal is not to replace measurement with an LLM. The goal is to test whether an LLM can close more of the already-measured grounding gap.

### Milestone 5 — Full trajectory evaluation

Eventually, evaluate full tau2-style interactive trajectories instead of local turn-level rows.

This should come after the turn-level cognitive spaces are measurable and optimized.

---

## Research claim so far

The current honest claim:

```text
On 547 tau2 retail turn-level tool-call decisions, the system exposes a large grounding bottleneck.
A full cognitive graph with stub grounding reaches 2% E2E success.
The same graph with deterministic grounding reaches 39%.
The same graph with oracle grounding reaches 100%.

This shows that the dataset contains measurable grounding pressure,
that the graph can consume improved grounding,
and that deterministic non-label logic can close part of the oracle gap.
```

That is the first real proof of the repo’s direction.

---

## Design philosophy

This project follows evaluation-driven development:

```text
dataset → metrics → baseline → failure analysis → capability isolation → optimization → reevaluation
```

Prompts are not the source of truth.

The source of truth is:

```text
dataset + metrics + controlled comparisons
```

A graph is not believed because it looks elegant.  
It earns its place only if it moves the right metric on the right behavior slice.
