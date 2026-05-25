# Cognitive Graph Lab

**Dataset-to-Agent-Architecture Lab.** Given a behavioral dataset of tool-calling
traces, infer the *cognitive topology* the task actually requires, recommend an
executable agent graph that covers those requirements, run it against baselines,
and surface where the graph still falls short.

The repository is not just a runtime agent. It is a lab for discovering the
*right* runtime graph from measurable evidence.

---

## Core Thesis

Most tool-calling agents fail because their cognitive responsibilities are
entangled inside a single prompt — perception, extraction, lookup, grounding,
memory, policy checking, planning, execution, and learning all mixed together.
That entanglement makes failures hard to debug, evaluate, and optimize.

This project takes a different path, built on one hypothesis:

> Behavioral datasets contain recoverable cognitive topology.

Instead of only asking *"did the agent succeed?"*, the lab asks:

- What kind of cognition does this dataset require?
- Where do tool arguments actually come from?
- Which capabilities are structurally necessary?
- Which graph should exist for this task — and does building it actually help?

The guiding principle throughout is **no optimization before measurement.**

---

## What Currently Works

Everything in this repository is **deterministic and inspectable today.** No LLM
calls are made. Each cognitive agent runs a documented stub heuristic and exposes
a `model_adapter` slot where a real model backend will later be injected.

The end-to-end loop you can run right now:

1. **Convert** raw tau-bench-style simulation traces into deterministic cognitive
   dataset artifacts.
2. **Report** on the cognitive topology of those artifacts (burden per tool,
   argument origins, failure heatmap).
3. **Infer** which capabilities the dataset requires (memory, grounding,
   readiness, deep planning) via threshold rules.
4. **Recommend** an executable graph that covers the inferred capabilities.
5. **Evaluate** that graph against baselines, including an *oracle* variant.
6. **Advise** on revisions by comparing stub vs oracle performance to locate the
   real bottleneck.

---

## Pipeline

```
Raw behavioral traces (results.json)
  │
  ▼  trace_converter/
Cognitive dataset artifacts
  (tool_registry · action_sequence · turn_supervision · failure_rows)
  │
  ▼  reports/
Cognitive topology reports
  (burden · argument emergence · failure heatmap)
  │
  ▼  recommender/
Capability inference  →  Recommended GraphSpec
  │
  ▼  graph_runner/ + graph/
Multi-graph evaluation
  (monolithic · minimal · recommended_stub · recommended_oracle)
  │
  ▼  recommender/revision_advisor.py
Graph revision suggestions
```

---

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 1. Convert tau-style simulation traces into cognitive dataset artifacts
python scripts/convert_traces.py \
  --input data/raw/results.json \
  --out-dir data/out/

# 2. Build cognitive topology reports
python scripts/build_reports.py \
  --input data/out/ \
  --out-dir data/out/

# 3. Recommend a cognitive graph from the report
python scripts/recommend_graph.py \
  --report data/out/cognitive_dataset_report.json \
  --out reports/recommended_graph.json

# 4. Evaluate graphs against each other (monolithic / minimal / stub / oracle)
python scripts/run_graph_evaluation.py \
  --recommended reports/recommended_graph.json \
  --report      reports/cognitive_dataset_report.json \
  --out-dir     data/out \
  --reports-dir reports

# Other entry points
python scripts/run_baseline.py --dataset data/dev/tool_calling_micro.jsonl  # monolithic only
python scripts/run_lab.py      --dataset data/dev/tool_calling_micro.jsonl  # original lab loop
python scripts/evaluate_trace.py                                            # score a single trace

# Run tests
pytest
```

Requires Python ≥ 3.11. Runtime dependencies are `pydantic>=2.0` and `rich>=13.0`.

---

## Scripts

| Script                     | Purpose                                                                 |
| -------------------------- | ----------------------------------------------------------------------- |
| `convert_traces.py`        | Raw tau-style traces → deterministic cognitive dataset artifacts        |
| `build_reports.py`         | Converted artifacts → cognitive topology reports                        |
| `recommend_graph.py`       | Report → capability inference → recommended `GraphSpec`                 |
| `run_graph_evaluation.py`  | Run + score 4 graph variants and emit revision advice                   |
| `run_baseline.py`          | Run only the monolithic baseline, per-row results                       |
| `run_lab.py`               | Original architecture-search lab loop (profiler → candidates → report)  |
| `evaluate_trace.py`        | Score a single cognitive trace against expected behavior                |

---

## The Cognitive Graph

A `GraphSpec` is a set of typed nodes executed in topological order by
`GraphExecutor`. No cognitive stages are hardcoded in the executor; the graph
definition fully controls which nodes run.

```
perceive → reason → grounding → readiness → plan → act → learn
```

Each node maps to an agent with a deterministic stub implementation today:

| Node        | Agent             | Stub responsibility                                              |
| ----------- | ----------------- | ---------------------------------------------------------------- |
| `perceive`  | `PerceiveAgent`   | Intent candidates, entity mentions, ambiguity detection          |
| `reason`    | `ReasonAgent`     | Tool selection, entity resolution, missing-requirement detection |
| `grounding` | `GroundingAgent`  | Resolve references to concrete IDs/values                        |
| `readiness` | `ReadinessAgent`  | Policy enforcement, confirmation gating, required fields         |
| `plan`      | `PlanAgent`       | Next action: `execute_tool` / `ask_followup` / `reject`          |
| `act`       | `ActAgent`        | Tool execution result or user-facing response                    |
| `learn`     | `LearnAgent`      | Runtime memory update, trace summary                             |

Not every task needs every node. The goal is to infer the **minimal sufficient
graph** from the dataset.

> Note: the `memory` capability is recommender-only in v1 — it is mapped onto the
> `learn` node rather than executed as a standalone node.

### Grounding modes

`GroundingAgent` is the most developed seam and demonstrates the pattern the rest
of the agents will follow. It runs in one of three modes:

- **`stub`** — deterministic heuristic resolution (default).
- **`oracle`** — resolves arguments directly from expected values, establishing a
  ceiling for "what if grounding were perfect."
- **`disabled`** — no grounding performed.

The oracle mode is what makes the evaluation harness able to quantify how much
grounding quality actually matters (see below).

---

## Graph Evaluation Harness

`run_graph_evaluation.py` runs four graph variants over the dataset:

| Variant              | Graph                                                | Role                              |
| -------------------- | ---------------------------------------------------- | --------------------------------- |
| `monolithic`         | single keyword→tool mapping                          | floor baseline                    |
| `minimal`            | `perceive → plan → act`                              | minimal decomposition             |
| `recommended_stub`   | full recommended graph, stub grounding               | what we'd ship today              |
| `recommended_oracle` | full recommended graph, oracle grounding             | ceiling if grounding were perfect |

The **stub-vs-oracle gap** is the key measurement: it isolates how much of the
remaining failure is attributable to grounding quality specifically, rather than
to graph structure.

### Revision Advisor

`GraphRevisionAdvisor` reads the evaluation report plus capability inference and
emits targeted suggestions. For example, when oracle grounding beats stub
grounding by more than a threshold on end-to-end success, it flags grounding as
the primary bottleneck and recommends investing in a real grounding agent with
entity lookup and fuzzy ID resolution — rather than blaming graph topology.

---

## Reports

| Artifact                        | Purpose                                            |
| ------------------------------- | -------------------------------------------------- |
| `cognitive_dataset_report.json` | Machine-readable full report                       |
| `cognitive_dataset_report.md`   | Human-readable report                              |
| `cognitive_action_topology.csv` | Per-tool cognitive burden table                    |
| `argument_emergence.csv`        | Where each argument comes from                     |
| `failure_heatmap.csv`           | Failures by tool, stage, read/write, and argument  |

### Argument Emergence

A core insight of the project: "argument extraction" is not one capability.
Different arguments require different cognitive machinery.

| Argument type                              | Typical origin                                |
| ------------------------------------------ | --------------------------------------------- |
| `first_name`, `last_name`, `zip`           | explicit user text                            |
| `order_id`, `user_id`, `payment_method_id` | tool-chained from prior results               |
| `item_ids`, `new_item_ids`                 | grounded from natural language + tool results |

---

## Capability Inference

The inference engine reads the cognitive report and decides which capabilities a
dataset structurally requires, using threshold rules in `recommender/thresholds.py`.

| Capability      | Signal                                       |
| --------------- | -------------------------------------------- |
| `memory`        | high tool-chaining across arguments          |
| `grounding`     | high global *or* peak grounding pressure     |
| `readiness`     | write/action risk and write failure fraction |
| `deep_planning` | high average tool-chain depth                |

Grounding uses both a global average and a **peak** signal, so a flood of trivial
zero-grounding arguments cannot statistically hide a rare but structurally
critical argument like `item_ids`.

---

## Evaluation Metrics

| Metric                  | Stage     | Description                                  |
| ----------------------- | --------- | -------------------------------------------- |
| `end_to_end_success`    | —         | Final action matches expected                |
| `tool_name_accuracy`    | plan      | Correct tool selected                        |
| `argument_exact_match`  | act       | Arguments exactly match expected             |
| `policy_violation_rate` | readiness | Fraction of rows with policy violations      |
| `stage_failure_rate`    | —         | Fraction with any stage failure              |

---

## Repository Structure

```
src/cognitive_tool_agent/
├── schemas/            Pydantic contracts for every stage, artifact, and graph
├── trace_converter/    Deterministic tau-style trace → cognitive artifacts
├── reports/            Cognitive topology report builders
├── recommender/        Signal extraction, capability inference, graph recommendation,
│                       and the revision advisor
├── agents/             Stub cognitive agents (perceive … learn), each with a
│                       model_adapter slot for future LLM backends
├── graph/              GraphExecutor + RunContext (node-driven execution)
├── graph_runner/       Multi-graph evaluation harness + trace writer
├── graph_builder/      Original lab loop (profiler, candidate generator, optimizer)
├── evals/              Metrics and evaluator
├── tools/              Tool registry + fake tools for offline runs
└── datasets/           JSONL dataset loader

data/
├── raw/                Input simulation traces
├── out/                Converted artifacts, reports, evaluation outputs
├── dev/                Small dev datasets (tool_calling_micro.jsonl, ...)
└── test/               Test fixtures

scripts/                CLI entry points (see Scripts table above)
tests/                  pytest suite (schemas, converter, recommender, graph runner, ...)
```

---

## Backend Design

Current backend: **`stub`**. All inference is deterministic and inspectable:

- trace conversion is deterministic
- report generation is deterministic
- capability inference is threshold-based
- graph recommendation is rule-based
- runtime agents are stub / rule-based

Every agent exposes a future `ModelAdapter` slot:

```python
class PerceiveAgent:
    def __init__(self, model_adapter: ModelAdapter | None = None): ...
```

Planned backends: OpenAI, Anthropic, configurable per-node routing, DSPy
optimization over prompts/programs, and learned graph search. The architecture is
designed so models can replace heuristics **without changing dataset contracts.**

---

## Development Principle

> No optimization before measurement.

```
raw traces → deterministic conversion → topology report → capability inference
→ graph recommendation → baseline execution → failure analysis → optimization
```

Datasets are treated as behavior-space specifications, not example collections.
Metrics act as selection pressure. Graph search is driven by measured failures,
not prompt intuition.

---

## Long-Term Goal

> Build a measurable cognitive operating system for production agents —

where datasets reveal required capabilities, reports expose cognitive topology,
graph recommendations are evidence-based, each node has explicit contracts,
failures map to cognitive stages, and optimizers search over measurable behavior
space.
