# Cognitive Graph Lab

**Dataset-to-Agent-Architecture Lab**: given a behavioral dataset, learn the cognitive graph that best explains and solves the task under measurable constraints.

---

## Core Thesis

Most tool-calling agents fail because their cognitive responsibilities are entangled inside a single prompt, making them hard to debug, evaluate, and optimize.

This repository goes further than decomposing a single agent.

It builds a **meta-agent that searches over cognitive decompositions**:

```text
Dataset
  ↓
Dataset Profiler        — What is the input/output space? What task type?
  ↓
Behavior Decomposer     — What cognitive stages does this task require?
  ↓
Graph Candidate Generator  — What graph architectures are plausible?
  ↓
Baseline Runner         — Run the simplest graph first. No optimization before baseline.
  ↓
Evaluator               — Score each candidate per stage and end-to-end.
  ↓
Failure Analyzer        — Map failures to cognitive stages.
  ↓
Graph Optimizer         — Recommend the next graph revision.
  ↓
LabReport               — graph spec + scores + failure map + tradeoff summary
```

---

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run the full Graph Builder loop
python scripts/run_lab.py --dataset data/dev/tool_calling_micro.jsonl

# Run only the monolithic baseline
python scripts/run_baseline.py --dataset data/dev/tool_calling_micro.jsonl

# Run tests
pytest
```

---

## What the Lab Produces

```text
LabReport
├── DatasetProfile       task_type, ambiguity_rate, label_set, tool_count
├── BehaviorDecomposition  stages, rationale
├── GraphCandidate[]     nodes, edges, latency_estimate, cost_estimate
├── EvaluationPlan       per-stage metrics
├── baseline_scores      { candidate_id → { metric → value } }
├── optimized_scores     { candidate_id → { metric → value } }
├── FailureMap           per-row failure_stage + failure_type
├── GraphRevision        from → to, change_type, rationale
└── tradeoff_summary     E2E success vs latency vs cost per candidate
```

---

## Graph Candidates

Three candidates are generated for every task:

| Candidate | Graph | Latency | Notes |
|---|---|---|---|
| A | `monolithic` | 1x | No decomposition. Baseline only. |
| B | `perceive → plan → act` | 2x | Separates signal extraction from execution. |
| C | `perceive → reason → readiness → plan → act → learn` | 4x | Full cognitive pipeline with policy gate. |

The executor is **node-driven**: it iterates `GraphSpec.nodes` in topological order and dispatches each node by role. No stages are hardcoded — the graph definition controls execution.

---

## Runtime Cognitive Pipeline

The full 6-stage pipeline (Candidate C):

```text
UserInput
  ↓
PerceptionResult    — intent candidates, entity mentions, ambiguity detection
  ↓
ReasoningResult     — entity grounding, tool selection, missing requirement analysis
  ↓
ReadinessResult     — policy enforcement, confirmation gate, required field validation
  ↓
PlanResult          — next action: execute_tool | ask_followup | reject | abstain
  ↓
ActionResult        — tool execution result or user-facing response
  ↓
LearningResult      — failure diagnosis, regression tags, optimization target
```

Each stage has a clear responsibility, structured input/output schemas, and can be evaluated independently.

---

## Failure Taxonomy

| Failure | Stage | Type |
|---|---|---|
| Wrong tool selected | planning | `wrong_tool_selected` |
| Executed without confirmation | readiness | `premature_execution` |
| Missing required argument | readiness | `missing_fields_blocked_execution` |
| Unsupported action not rejected | planning | `rejection_missed` |
| Wrong argument values | acting | `wrong_arguments` |
| No tool matched intent | reasoning | `tool_not_selected` |

---

## Repository Structure

```text
cognitive-tool-agent/
│
├── src/cognitive_tool_agent/
│   │
│   ├── schemas/
│   │   ├── common.py          UserInput, ToolSchema, Confidence, Evidence
│   │   ├── perceive.py        PerceptionResult
│   │   ├── reason.py          ReasoningResult
│   │   ├── readiness.py       ReadinessResult
│   │   ├── plan.py            PlanResult, ToolCallPlan
│   │   ├── act.py             ActionResult
│   │   ├── learn.py           LearningResult, FailureAnalysis
│   │   ├── trace.py           CognitiveTrace
│   │   ├── dataset.py         DatasetRow, ExpectedBehavior
│   │   ├── graph_spec.py      NodeSpec, EdgeSpec, GraphSpec
│   │   └── graph_builder.py   DatasetProfile, LabReport, FailureMap, ...
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
│   │   └── cognitive_graph.py  GraphExecutor (node-driven), RunContext
│   │
│   ├── graph_builder/          Meta-agent loop
│   │   ├── dataset_profiler.py
│   │   ├── behavior_decomposer.py
│   │   ├── graph_candidate_generator.py
│   │   ├── evaluation_designer.py
│   │   ├── baseline_runner.py
│   │   ├── failure_analyzer.py
│   │   ├── graph_optimizer.py
│   │   └── lab.py              Orchestrator → LabReport
│   │
│   ├── evals/
│   │   ├── metrics.py          end_to_end_success, tool_name_accuracy, ...
│   │   └── evaluator.py        Evaluator.score(traces, rows)
│   │
│   ├── tools/
│   │   ├── registry.py         ToolRegistry
│   │   └── fake_tools.py       get_order_status, cancel_order, update_address
│   │
│   └── datasets/
│       └── loader.py           load_jsonl(path) → list[DatasetRow]
│
├── data/dev/
│   └── tool_calling_micro.jsonl   5-row tool-calling demo dataset
│
├── scripts/
│   ├── run_lab.py              Full Graph Builder loop → LabReport
│   ├── run_baseline.py         Monolithic baseline only → per-row table
│   └── evaluate_trace.py       Load saved traces → failure analysis
│
└── tests/
    ├── test_schemas.py         Round-trip Pydantic tests for all schemas
    └── test_graph_smoke.py     Full Lab loop integration tests
```

---

## Backend Design

Current backend: `stub` — all agents use deterministic keyword/rule-based logic.

Every agent exposes a `ModelAdapter` slot:

```python
class PerceiveAgent:
    def __init__(self, model_adapter: ModelAdapter | None = None): ...
```

Phase 2 will add:
- `LLMAdapter(OpenAIBackend)`
- `LLMAdapter(AnthropicBackend)`
- Per-node model routing via `NodeSpec.model_hint`

The loop, schemas, executor, and evaluators are model-agnostic.

---

## Evaluation Metrics

| Metric | Stage | Description |
|---|---|---|
| `end_to_end_success` | — | Final action matches expected |
| `tool_name_accuracy` | plan | Correct tool selected |
| `argument_exact_match` | act | Arguments exactly match expected |
| `policy_violation_rate` | readiness | Fraction of rows with policy violations |
| `stage_failure_rate` | — | Fraction with any stage failure |

---

## Long-Term Goal

> Given a behavioral dataset, learn the cognitive graph that best explains and solves the task under measurable constraints.

The repo is not just a runtime agent. It is a lab for discovering the right runtime graph.
