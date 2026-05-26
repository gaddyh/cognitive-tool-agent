# Cognitive Tool Agent

A dataset-to-agent-architecture lab for evaluation-driven tool-calling systems.

This repo explores a concrete thesis:

> Tool-agent failures are not one blob.  
> They can be decomposed into measurable cognitive spaces, and the right graph should be justified by dataset evidence, not architectural taste.

The current focus is tau2-style retail tool-agent traces. The repo converts raw simulations into cognitive artifacts, creates a stratified train/dev/test split, derives a graph from train-only behavioral analysis, freezes that graph, and evaluates it independently across all/train/dev/test.

---

## Current status

The repo now has a working end-to-end experimental harness:

```text
tau2 results.json
   ↓
trace conversion
   ↓
simulation profiling + stratified train/dev/test split
   ↓
TRAIN ONLY:
  cognitive dataset report
  capability inference
  graph recommendation
   ↓
frozen recommended_graph.json
   ↓
EVALUATION:
  all / train / dev / test
   ↓
proof table
```

Latest full-pipeline run:

```bash
python scripts/run_pipeline.py
```

Input:

```text
data/raw/simulations/baseline_retail_100/results.json
```

Output:

```text
data/out
reports
```

Latest split:

| Split | Turn-level rows |
|---|---:|
| all | 547 |
| train | 349 |
| dev | 106 |
| test | 92 |

The graph was derived from train-only behavioral analysis, frozen, then evaluated independently across train/dev/test.

---

## Latest proof table

Same frozen `recommended_graph.json`, evaluated across splits.

### E2E Success

| Variant | all | train | dev | test |
|---|---:|---:|---:|---:|
| `recommended_stub` | 2% | 2% | 1% | 2% |
| `recommended_deterministic` | 39% | 39% | 40% | 37% |
| `recommended_oracle` | 100% | 100% | 100% | 100% |

### Tool Accuracy

| Variant | all | train | dev | test |
|---|---:|---:|---:|---:|
| `recommended_stub` | 2% | 2% | 1% | 2% |
| `recommended_deterministic` | 39% | 39% | 40% | 37% |
| `recommended_oracle` | 100% | 100% | 100% | 100% |

### Argument Match

| Variant | all | train | dev | test |
|---|---:|---:|---:|---:|
| `recommended_stub` | 0% | 0% | 0% | 0% |
| `recommended_deterministic` | 23% | 22% | 23% | 24% |
| `recommended_oracle` | 99% | 98% | 98% | 100% |

Interpretation:

```text
stub < deterministic < oracle
```

The deterministic grounding node closes a real part of the oracle gap without using expected labels. The dev/test scores are close to train, which is the important result: the train-derived graph is not only fitting the train split.

---

## Experimental design boundary

This repo now enforces a train-only EDD boundary.

### Rule

```text
Anything that learns, summarizes, infers, recommends, or shapes the graph uses train only.

Anything that measures performance may run on all/train/dev/test.

tool_registry.json is the only global artifact because it defines the environment/API surface,
not observed behavior.
```

### Artifact scopes

| Artifact | Scope | Allowed to influence graph design? |
|---|---|---:|
| `tool_registry.json` | global | yes, environment only |
| `conversion_summary.json` | global descriptive | no |
| `split_manifest.json` | global descriptive | no |
| `split_report.md` | all splits, descriptive | no |
| `cognitive_dataset_report.json` | train only | yes |
| capability inference | train only | yes |
| `recommended_graph.json` | derived from train only | yes, frozen before eval |
| evaluation outputs | all/train/dev/test | no, measurement only |

Verification from the latest run:

```text
source = "train split (62 sims)" | boundary = train_only
```

This confirms that `cognitive_dataset_report.json` is not using dev/test simulations.

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
| Simulations | 100 |
| Turn-level tool-call rows | 547 |
| Train rows | 349 |
| Dev rows | 106 |
| Test rows | 92 |

Generated artifacts include:

```text
data/out/tool_registry.json
data/out/action_sequence.jsonl
data/out/turn_supervision.jsonl
data/out/failure_rows.jsonl
data/out/conversion_summary.json
data/out/splits/train_simulation_ids.json
data/out/splits/dev_simulation_ids.json
data/out/splits/test_simulation_ids.json
data/out/splits/train_supervision.jsonl
data/out/splits/dev_supervision.jsonl
data/out/splits/test_supervision.jsonl
```

This proves the raw tau2 traces contain enough structure to extract action-level, turn-level, split-level, and failure-level supervision.

---

### 2. The split is scenario-stratified

The pipeline creates a simulation-level train/dev/test split.

The split is stratified by scenario profile, including signals such as:

```text
scenario family
single_action vs multi_action
grounding vs no_grounding
difficulty bucket
```

This matters because a random split could hide important behavior classes in only one split. The goal is not just to split rows. The goal is to preserve behavior-space coverage across train/dev/test.

---

### 3. Analysis artifacts are train-only

The cognitive report and graph recommendation are generated from train simulations only.

Latest run:

```text
2/5  Build cognitive dataset report  [TRAIN ONLY]
  ✓ source = "train split (62 sims)" | boundary = train_only

3/5  Recommend graph  [TRAIN ONLY → frozen]
  ✓ Graph: perceive → reason → grounding → readiness → plan → act → learn
  ✓ confidence = 0.59
```

This is the key experimental discipline:

```text
Design on train.
Freeze the graph.
Measure on train/dev/test.
```

---

### 4. Capability inference recommends the full cognitive graph

Recommended graph:

```text
perceive → reason → grounding → readiness → plan → act → learn
```

The recommendation is not just a hand-drawn architecture. It is derived from train-only cognitive pressure signals, then frozen before evaluation.

This is the first important repo-level claim:

```text
Behavioral traces can be used to infer a candidate cognitive topology.
```

---

### 5. Deterministic grounding improves the graph

The same recommended topology is evaluated with three grounding modes:

| Variant | Meaning |
|---|---|
| `recommended_stub` | Full graph with weak/stub grounding |
| `recommended_deterministic` | Full graph with deterministic non-label grounding |
| `recommended_oracle` | Full graph with oracle grounding ceiling |

Latest all-split result:

| Variant | E2E Success | Tool Acc | Arg Match |
|---|---:|---:|---:|
| `recommended_stub` | 2% | 2% | 0% |
| `recommended_deterministic` | 39% | 39% | 23% |
| `recommended_oracle` | 100% | 100% | 99% |

This isolates grounding as a real bottleneck:

```text
The graph structure can use better grounding.
The deterministic grounding node closes part of the oracle gap.
The remaining gap is measurable.
```

---

## The full pipeline

Run everything:

```bash
python scripts/run_pipeline.py \
  --input data/raw/simulations/baseline_retail_100/results.json
```

Default phases:

```text
DESIGN PHASE (train only)
  1/5  Convert traces + stratified split
  2/5  Build cognitive dataset report [TRAIN ONLY]
  3/5  Recommend graph [TRAIN ONLY → frozen]
  4/5  Build split report [descriptive]

EVALUATION PHASE
  5/5  Evaluate on all / train / dev / test
```

Incremental flags:

```bash
python scripts/run_pipeline.py --skip-convert
python scripts/run_pipeline.py --skip-reports
python scripts/run_pipeline.py --skip-recommend
python scripts/run_pipeline.py --limit 20
```

The `--limit` flag is useful for smoke tests. It should not be used for claims.

---

## Running individual steps

Convert traces and build splits:

```bash
python scripts/convert_traces.py \
  --input data/raw/simulations/baseline_retail_100/results.json \
  --out-dir data/out
```

Build cognitive report from train only:

```bash
python scripts/build_reports.py \
  --out-dir data/out \
  --reports-dir reports \
  --source baseline_retail_100
```

By default, reports are train-only.

Full-dataset inspection is possible but should not be used for graph design:

```bash
python scripts/build_reports.py --no-train-only
```

Recommend graph from the train-only report:

```bash
python scripts/recommend_graph.py \
  --report reports/cognitive_dataset_report.json \
  --out reports/recommended_graph.json
```

Run turn-level graph evaluation:

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

---

## Current graph variants

The turn-level evaluator compares graph configurations such as:

| Graph | Shape / mode |
|---|---|
| `recommended_stub` | full recommended graph with stub grounding |
| `recommended_deterministic` | full recommended graph with deterministic grounding |
| `recommended_oracle` | full recommended graph with oracle grounding |

The important comparison is controlled:

```text
same graph topology
same evaluation rows
only grounding mode changes
```

That makes the stub/deterministic/oracle gap interpretable.

---

## Grounding modes

The grounding node currently supports several modes:

| Mode | Meaning |
|---|---|
| `stub` | Returns little/no resolved grounding. Baseline floor. |
| `deterministic` | Uses prior tool calls, prior tool results, and explicit text patterns. No labels. |
| `oracle` | Copies expected arguments from the dataset row. Ceiling only. |
| `llm` | Reserved for future model-backed grounding. |
| `disabled` | Used when grounding is intentionally removed. |

The oracle is not a production solution. It is a measurement instrument.

It answers:

```text
If grounding were perfect, could the rest of the graph use it?
```

The current answer is yes.

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

Current Pass 1 resolves scalar IDs such as:

| Field | Strategy |
|---|---|
| `order_id` | prior tool call args, prior tool result JSON, order-id regex |
| `user_id` | prior tool call args, prior tool result JSON, user-id-like result strings |
| `product_id` | prior tool call args, prior tool result JSON, conservative product type matching |
| `payment_method_id` | prior tool call args, prior tool result JSON |
| generic `*_id` | same-key lookup in prior calls/results |

Deferred to later passes:

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
| `perceive` | `perception` |
| `reason` | `reasoning` |
| `grounding` | `grounding` |
| `readiness` | `readiness` |
| `plan` | `plan` |
| `act` | `action` |
| `learn` | `learning` |

Each node receives a `NodeInput` containing only the upstream slots allowed by the graph wiring.

This makes graph wiring testable. A node cannot silently consume arbitrary context unless the graph declares that dependency.

---

## Why the original graph evaluation was all zero

The first graph evaluation compressed each full simulation into one row:

```text
first user message → selected primary / final-ish action
```

That was misaligned.

A typical simulation has many turns and many tool calls. Asking weak stubs to infer a late write action from the first user message produced all zeros.

This was not necessarily a graph failure. It was an evaluation-unit mismatch.

The current primary evaluation unit is:

```text
local turn context → next assistant tool call
```

This creates 547 turn-level tool-call decision rows.

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
2. Build stratified train/dev/test split
3. Build train-only cognitive reports
4. Recommend graph from train-only evidence
5. Freeze recommended_graph.json
6. Evaluate frozen graph on train/dev/test
7. Inspect field-level failures
8. Add deterministic or model-backed capability
9. Reevaluate
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
cognitive_dataset_report.json is train-only
recommended_graph.json is derived from train-only report
evaluation outputs exist for all/train/dev/test
same frozen graph is evaluated across splits
recommended_deterministic beats recommended_stub
oracle remains the measured ceiling
```

---

## Current limitations

The current result is strong but scoped.

Known limitations:

```text
- deterministic grounding is Pass 1 only
- item_ids and new_item_ids are unresolved
- product_id grounding can be over-eager
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
deterministic E2E >= 39%
wrong-resolution rate decreases
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
On 547 tau2 retail turn-level tool-call decisions,
a graph derived from train-only behavioral analysis was frozen
and evaluated independently across all/train/dev/test.

The full cognitive graph with stub grounding reaches:
  2% all / 2% train / 1% dev / 2% test E2E success.

The same frozen graph with deterministic grounding reaches:
  39% all / 39% train / 40% dev / 37% test E2E success.

The same frozen graph with oracle grounding reaches:
  100% all / 100% train / 100% dev / 100% test E2E success.

This shows that the dataset contains measurable grounding pressure,
that the graph can consume improved grounding,
and that deterministic non-label logic can close part of the oracle gap
without obvious train-only overfitting in the current split.
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
