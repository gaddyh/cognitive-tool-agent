# Cognitive Graph Lab

**Dataset-to-Agent-Architecture Lab**: given a behavioral dataset, infer the cognitive topology of the task and recommend the graph that best explains and solves it under measurable constraints.

---

## Core Thesis

Most tool-calling agents fail because their cognitive responsibilities are entangled inside a single prompt.

A single prompt often mixes:

- perception
- extraction
- lookup
- grounding
- memory
- readiness / policy checking
- planning
- execution
- learning

This makes failures hard to debug, evaluate, and optimize.

This repository explores a different path:

> Behavioral datasets contain recoverable cognitive topology.

Instead of only asking:

```text
Did the agent succeed?
```

the lab asks:

```text
What kind of cognition does this dataset require?
Where do tool arguments come from?
Which capabilities are structurally necessary?
Which graph should exist for this task?
```

The long-term goal:

> Given a behavioral dataset, learn the cognitive graph that best explains and solves the task under measurable constraints.

---

## New Vision

The project is evolving from a fixed cognitive agent into a **dataset-driven graph recommender**.

The new pipeline:

```text
Raw behavioral traces
  ↓
Trace-to-Cognitive-Dataset Converter
  ↓
Cognitive Dataset Reports
  ↓
Capability Inference Engine
  ↓
Graph Recommender
  ↓
Recommended GraphSpec
  ↓
Evaluation / execution / optimization
```

The key shift:

```text
from: manually designing cognitive graphs
to: inferring cognitive graphs from dataset topology
```

---

## Why Reports Matter

The reports expose hidden structure inside a tool-calling dataset.

Examples of cognitive topology signals:

| Signal | Meaning | Graph implication |
|---|---|---|
| High tool-chaining % | Important values come from prior tools | memory / state node |
| High grounding % | Values are not explicit in user text | grounding node |
| High write failure fraction | State-changing actions are risky | readiness / policy node |
| High chain depth | Tool decisions depend on prior steps | reasoning node |
| High explicit arg % | Values are directly extractable | lightweight perception/extraction |

This lets the system recommend graph structure from measurable evidence.

---

## Current End-to-End Flow

```text
1. Convert raw tau-style traces
   results.json
     ↓
   tool_registry.json
   action_sequence.jsonl
   turn_supervision.jsonl
   failure_rows.jsonl
   conversion_summary.json

2. Build cognitive reports
   converted artifacts
     ↓
   cognitive_dataset_report.json
   cognitive_dataset_report.md
   cognitive_action_topology.csv
   argument_emergence.csv
   failure_heatmap.csv

3. Infer required capabilities
   cognitive_dataset_report.json
     ↓
   memory_required
   grounding_required
   readiness_required
   deep_planning_required

4. Recommend a graph
   capability inference
     ↓
   recommended_graph.json
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
  --out data/out/recommended_graph.json

# 4. Original micro-demo Lab loop
python scripts/run_lab.py --dataset data/dev/tool_calling_micro.jsonl

# Run tests
pytest
```

---

## Trace-to-Cognitive-Dataset Converter

The converter ingests tau-bench-style simulation JSON.

Input structure:

```text
tasks[]
  evaluation_criteria.actions[]     # expected tool calls + arguments

simulations[]
  messages[]                        # user / assistant / tool turns
  reward_info.action_checks[]       # action_match, action_reward
```

It emits:

| Artifact | Purpose |
|---|---|
| `tool_registry.json` | Per-tool required args, seen args, usage counts, read/write type |
| `action_sequence.jsonl` | One row per simulation: expected vs actual tool sequences |
| `turn_supervision.jsonl` | One row per turn: deterministic cognitive labels |
| `failure_rows.jsonl` | Failed expected actions with aligned actual behavior |
| `conversion_summary.json` | Fast sanity-check counts |

This layer is intentionally deterministic. No LLM annotation is used.

---

## Cognitive Dataset Reports

The report builder turns converted traces into cognitive topology.

Main report artifacts:

| Artifact | Purpose |
|---|---|
| `cognitive_dataset_report.json` | Machine-readable full report |
| `cognitive_dataset_report.md` | Human-readable report |
| `cognitive_action_topology.csv` | Tool-level cognitive burden table |
| `argument_emergence.csv` | Argument origin matrix |
| `failure_heatmap.csv` | Failures by tool, stage, read/write, and argument |

### Cognitive Action Topology

Each tool is scored by cognitive burden:

| Signal | Meaning |
|---|---|
| extraction burden | number of required arguments |
| memory burden | average turn distance before action |
| readiness burden | write tools require confirmation / policy gate |
| reasoning burden | preceding tool-chain depth |
| grounding burden | fraction of arguments not directly explicit |

### Argument Emergence Matrix

This is one of the core insights of the project.

It asks:

```text
Where do tool arguments actually come from?
```

Examples:

| Argument type | Typical origin |
|---|---|
| `first_name`, `last_name`, `zip` | explicit user text |
| `order_id`, `user_id`, `payment_method_id` | tool-chained |
| `item_ids`, `new_item_ids` | grounded from natural language + tool results |

This shows that “argument extraction” is not one capability. Different arguments require different cognitive machinery.

---

## Capability Inference Engine

The Capability Inference Engine reads `cognitive_dataset_report.json` and infers which cognitive capabilities are required.

Capabilities:

| Capability | Signal |
|---|---|
| `memory` | high tool-chaining across arguments |
| `grounding` | high global or peak grounding pressure |
| `readiness` | write/action risk and write failure fraction |
| `deep_planning` | high average tool-chain depth |

The grounding signal uses both:

```text
global grounding strength
peak grounding strength
```

This prevents high-volume zero-grounding arguments from hiding structurally important arguments like `item_ids` or `new_item_ids`.

Example:

```text
grounding_strength = weighted global average
peak_grounding_strength = max grounding % among sufficiently common args
effective_grounding = max(global, peak)
```

---

## Graph Recommender

The recommender wraps an executable `GraphSpec` inside a `RecommendedGraph`.

Example output:

```text
perceive → reason → grounding → readiness → plan → act → learn
```

The recommendation includes:

```text
graph_spec
required_capabilities
rationale
confidence
memory_required
readiness_required
parallel_lookup_nodes
```

This keeps the recommendation explainable while remaining compatible with the graph executor.

---

## Original Graph Builder Lab

The original Lab loop remains as the first architecture-search scaffold:

```text
Dataset
  ↓
Dataset Profiler
  ↓
Behavior Decomposer
  ↓
Graph Candidate Generator
  ↓
Baseline Runner
  ↓
Evaluator
  ↓
Failure Analyzer
  ↓
Graph Optimizer
  ↓
LabReport
```

It generates and compares simple graph candidates:

| Candidate | Graph | Notes |
|---|---|---|
| A | `monolithic` | Baseline only |
| B | `perceive → plan → act` | Minimal decomposition |
| C | `perceive → reason → readiness → plan → act → learn` | Full initial cognitive graph |

This layer is still useful for deterministic candidate evaluation, but the newer report/recommender layer is where the graph structure begins to emerge from dataset topology.

---

## Runtime Cognitive Pipeline

The full pipeline may include:

```text
UserInput
  ↓
PerceptionResult      — intent candidates, entity mentions, ambiguity detection
  ↓
ReasoningResult       — tool selection, chain interpretation, missing requirements
  ↓
GroundingResult       — resolve natural language entities to concrete IDs/values
  ↓
ReadinessResult       — policy enforcement, confirmation gate, required fields
  ↓
PlanResult            — next action: execute_tool | ask_followup | reject | abstain
  ↓
ActionResult          — tool execution result or user-facing response
  ↓
LearningResult        — runtime memory update, trace summary, unresolved goals
```

Not every task needs every node.

The goal is to infer the minimal sufficient graph from data.

---

## Repository Structure

```text
cognitive-tool-agent/
│
├── src/cognitive_tool_agent/
│   │
│   ├── schemas/
│   │   ├── common.py              UserInput, ToolSchema, Confidence, Evidence
│   │   ├── perceive.py            PerceptionResult
│   │   ├── reason.py              ReasoningResult
│   │   ├── readiness.py           ReadinessResult
│   │   ├── plan.py                PlanResult, ToolCallPlan
│   │   ├── act.py                 ActionResult
│   │   ├── learn.py               LearningResult, FailureAnalysis
│   │   ├── trace.py               CognitiveTrace
│   │   ├── dataset.py             DatasetRow, ExpectedBehavior
│   │   ├── graph_spec.py          NodeSpec, EdgeSpec, GraphSpec
│   │   ├── graph_builder.py       DatasetProfile, LabReport, FailureMap, ...
│   │   ├── simulation.py          Raw tau-style simulation schemas
│   │   ├── trace_converter.py     Converter artifact schemas
│   │   └── recommender.py         CapabilityRequirement, RecommendedGraph
│   │
│   ├── trace_converter/
│   │   ├── simulation_loader.py
│   │   ├── tool_registry_scanner.py
│   │   ├── action_aligner.py
│   │   ├── turn_supervisor.py
│   │   ├── failure_extractor.py
│   │   └── converter.py
│   │
│   ├── reports/
│   │   ├── report_builder.py
│   │   ├── action_topology.py
│   │   ├── argument_emergence.py
│   │   └── failure_heatmap.py
│   │
│   ├── recommender/
│   │   ├── thresholds.py
│   │   ├── signal_extractor.py
│   │   ├── capability_inference.py
│   │   └── graph_recommender.py
│   │
│   ├── agents/
│   │   ├── perceive_agent.py
│   │   ├── reason_agent.py
│   │   ├── readiness_agent.py
│   │   ├── plan_agent.py
│   │   ├── act_agent.py
│   │   └── learn_agent.py
│   │
│   ├── graph/
│   │   └── cognitive_graph.py      GraphExecutor, RunContext
│   │
│   ├── graph_builder/
│   │   ├── dataset_profiler.py
│   │   ├── behavior_decomposer.py
│   │   ├── graph_candidate_generator.py
│   │   ├── evaluation_designer.py
│   │   ├── baseline_runner.py
│   │   ├── failure_analyzer.py
│   │   ├── graph_optimizer.py
│   │   └── lab.py
│   │
│   ├── evals/
│   │   ├── metrics.py
│   │   └── evaluator.py
│   │
│   ├── tools/
│   │   ├── registry.py
│   │   └── fake_tools.py
│   │
│   └── datasets/
│       └── loader.py
│
├── data/
│   ├── raw/
│   ├── out/
│   └── dev/
│       └── tool_calling_micro.jsonl
│
├── scripts/
│   ├── convert_traces.py
│   ├── build_reports.py
│   ├── recommend_graph.py
│   ├── run_lab.py
│   ├── run_baseline.py
│   └── evaluate_trace.py
│
└── tests/
    ├── test_schemas.py
    ├── test_graph_smoke.py
    ├── test_trace_converter.py
    └── test_recommender.py
```

---

## Backend Design

Current backend: `stub`.

All current inference is deterministic and inspectable:

- trace conversion is deterministic
- report generation is deterministic
- capability inference is threshold-based
- graph recommendation is rule-based
- runtime agents are stub/rule-based

Every agent exposes a future `ModelAdapter` slot:

```python
class PerceiveAgent:
    def __init__(self, model_adapter: ModelAdapter | None = None): ...
```

Future backends:

- OpenAI
- Anthropic
- configurable per-node routing
- DSPy optimization over prompts/programs
- learned graph search

The architecture is designed so models can replace heuristics without changing dataset contracts.

---

## Evaluation Metrics

| Metric | Stage | Description |
|---|---|---|
| `end_to_end_success` | — | Final action matches expected |
| `tool_name_accuracy` | plan | Correct tool selected |
| `argument_exact_match` | act | Arguments exactly match expected |
| `policy_violation_rate` | readiness | Fraction of rows with policy violations |
| `stage_failure_rate` | — | Fraction with any stage failure |
| `capability_coverage` | graph | Recommended graph covers inferred required capabilities |
| `graph_complexity` | graph | Number of nodes / expected cost / expected latency |

---

## Development Principle

No optimization before measurement.

The intended workflow:

```text
raw traces
  ↓
deterministic conversion
  ↓
topology report
  ↓
capability inference
  ↓
graph recommendation
  ↓
baseline execution
  ↓
failure analysis
  ↓
optimization
```

Datasets are treated as behavior-space specifications, not example collections.

Metrics act as selection pressure.

Graph search should be driven by measured failures, not prompt intuition.

---

## Long-Term Goal

> Build a measurable cognitive operating system for production agents.

Where:

- datasets reveal required capabilities
- reports expose cognitive topology
- graph recommendations are evidence-based
- each node has explicit contracts
- failures are mapped to cognitive stages
- optimizers search over measurable behavior space

The repo is not just a runtime agent.

It is a lab for discovering the right runtime graph.
