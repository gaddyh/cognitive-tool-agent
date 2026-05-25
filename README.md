# Cognitive Graph Lab

**Dataset-to-Agent-Architecture Lab.** Given a behavioral dataset of tool-calling
traces, infer the *cognitive topology* a task actually requires, recommend an
executable agent graph that covers those requirements, run it against baselines,
and surface where the graph still falls short.

This is not just a runtime agent. It is a lab for **discovering the right runtime
graph from measurable evidence.**

---

## 1. The Core Idea

Most tool-calling agents are built as one giant prompt that does everything at
once - perceive, reason, look things up, check policy, plan, act, learn. When
such an agent fails, you cannot tell *which* of those jobs broke, because they
are all tangled together in a single step.

This project takes the opposite approach. It splits cognition into **separate
single-job stages**, each with a typed input and a typed output, so that:

1. every stage's work is individually inspectable, and
2. a failure can be attributed to a specific stage.

On top of that sits one hypothesis, which is the whole reason the lab exists:

> **Behavioral datasets contain recoverable cognitive topology.**

Meaning: if you look at *how* tasks actually unfold in the data, the data itself
tells you which cognitive stages the task needs - instead of you guessing and
hand-tuning a prompt. The guiding rule throughout is:

> **No optimization before measurement.**

Measure what the task requires first; design (and later, optimize) second.

---

## 2. The Pipeline (end to end)

```
Raw behavioral traces (tau-bench-style results.json)
  |
  v  trace_converter/        deterministic - no LLM
Cognitive dataset artifacts
  (tool registry, action sequences, per-turn supervision, failure rows)
  |
  v  reports/
Cognitive topology reports
  (cognitive burden, argument emergence, failure heatmap)
  |
  v  recommender/
Capability inference  ->  Recommended GraphSpec
  |
  v  graph_runner/ + graph/
Multi-graph evaluation
  (monolithic, minimal, recommended_stub, recommended_oracle)
  |
  v  recommender/revision_advisor.py
Graph revision suggestions
```

Every stage is **deterministic and inspectable today.** No LLM calls are made
anywhere yet - each cognitive stage runs a documented stub heuristic, and exposes
a slot where a real model will later be plugged in.

---

## 3. The Cognitive Graph

A `GraphSpec` is a set of typed **nodes** connected by **edges**, executed in
dependency order. Each node has a `role`; each role maps to an agent.

```
perceive -> reason -> grounding -> readiness -> plan -> act -> learn
```

| Node        | Agent            | Job (current stub behavior)                                      |
| ----------- | ---------------- | ---------------------------------------------------------------- |
| `perceive`  | `PerceiveAgent`  | Intent candidates, entity mentions, ambiguity detection          |
| `reason`    | `ReasonAgent`    | Tool selection, entity resolution, missing-requirement detection |
| `grounding` | `GroundingAgent` | Resolve references to concrete IDs/values                        |
| `readiness` | `ReadinessAgent` | Policy enforcement, confirmation gating, required fields         |
| `plan`      | `PlanAgent`      | Next action: `execute_tool` / `ask_followup` / `reject`          |
| `act`       | `ActAgent`       | Tool execution result or user-facing response                    |
| `learn`     | `LearnAgent`     | Runtime memory update, trace summary                             |

Not every task needs every node. The point of the lab is to infer the **minimal
sufficient graph** for a given dataset.

> **`memory` is recommender-only.** The recommender can decide a task *needs*
> memory, but there is no executable `memory` node - that capability is mapped
> onto the `learn` node. A hand-crafted graph containing a literal `memory` node
> will deliberately raise an error rather than silently do nothing.

---

## 4. How Execution Works **Today** (current code)

This section describes what is actually in the repository right now. Section 7
describes the refactor that changes it.

### Typed contracts between stages

Stages never pass each other free-form text. Each one emits a **typed Pydantic
result** - `PerceptionResult`, `ReasoningResult`, etc. - with named fields and
validated values (e.g. a confidence score is forced into `0.0-1.0`). If a stage
produces the wrong shape, the error fires *at that handoff*, not three stages
later. These contracts are what make "you can tell which stage broke" true in
practice.

### The trace: one evidence file per run

As stages run, each result is attached to a single `CognitiveTrace` with one slot
per stage (`perception`, `reasoning`, ... `learning`), each either filled or
`None`. The trace is your detective's evidence file: after a run you can read the
entire cognitive chain and see exactly where it went wrong. The `None` slots also
show which stages a task didn't even use.

### The executor

`GraphExecutor` reads a `GraphSpec`, computes a valid execution order, and runs
each node. The order is **data-driven**: `GraphSpec.topological_order()` runs
Kahn's algorithm over the edges, which both orders the nodes by dependency *and*
detects cycles for free (a graph that can't be ordered is rejected).

**The current limitation this README is honest about:** while *order* comes from
the edges, the *data passed between stages* is still hardcoded inside the
executor. The dispatch method literally contains:

```python
ctx.reasoning = self._reason.run(user_input, ctx.perception)
ctx.plan      = self._plan.run(user_input, ctx.reasoning, ctx.readiness)
```

So "reason consumes perception" and "plan consumes reasoning + readiness" live in
Python, not in the graph spec. The edges and this hardcoded wiring are *two*
descriptions of the same dependencies - and the hardcoded one wins. Removing that
duplication is the refactor described in Section 7.

### The model seam (not yet live)

Each agent is built to run in a `mode` (`stub` today; `llm` later) and accepts a
`model_adapter` - the standard plug a real model will fit into. **Today this seam
is scaffolding:** the adapter is accepted but not yet consulted, and the LLM code
paths raise "not implemented." Making this seam real, one node at a time, is the
main roadmap (Section 8).

`GroundingAgent` is the most-developed stage and previews the pattern: it already
runs in three modes - `stub` (heuristic), `oracle` (fed the correct answers, to
measure a ceiling), and `disabled`.

---

## 5. How "the data reveals the topology" actually works

This is counting, not magic. The converter reads the example traces and computes
**signals** per tool - most importantly the **argument emergence matrix**, which
asks where each tool argument *comes from*:

| Argument type                              | Typical origin                                |
| ------------------------------------------ | --------------------------------------------- |
| `first_name`, `last_name`, `zip`           | explicit in the user's text                   |
| `order_id`, `user_id`, `payment_method_id` | chained from a previous tool's result         |
| `item_ids`, `new_item_ids`                 | grounded from natural language + tool results |

The key insight: **"argument extraction" is not one capability.** A `zip` you
read off the message; an `order_id` you must remember from an earlier call; an
`item_id` you must *ground* from fuzzy language. Different origins demand different
machinery - which is exactly what tells you which stages the task needs.

`recommender/thresholds.py` then turns those signals into capability flags:

| Capability      | Triggered by                                  |
| --------------- | --------------------------------------------- |
| `memory`        | high tool-chaining across arguments           |
| `grounding`     | high global *or* peak grounding pressure      |
| `readiness`     | write/action risk and write-failure fraction  |
| `deep_planning` | high average tool-chain depth                 |

Grounding uses both an **average** and a **peak** signal, so a flood of trivial
zero-grounding arguments can't statistically bury one rare-but-critical argument
like `item_ids`.

**Honest caveat:** these thresholds are hand-set today. So the "inference" is
currently as good as the cutoffs chosen - the same human intuition the project
critiques, relocated from prompt-writing to threshold-setting. Making the
thresholds *learned from which graphs actually win* is the real long-term step.

---

## 6. The Evaluation Harness - and why the oracle matters

`run_graph_evaluation.py` runs four graph variants over the dataset:

| Variant              | Graph                                        | Role                              |
| -------------------- | -------------------------------------------- | --------------------------------- |
| `monolithic`         | plan->act, starved of upstream cognition     | floor baseline                    |
| `minimal`            | `perceive -> plan -> act`                    | minimal decomposition             |
| `recommended_stub`   | full recommended graph, stub grounding       | what we'd ship today              |
| `recommended_oracle` | full recommended graph, **oracle** grounding | ceiling if grounding were perfect |

The **stub-vs-oracle gap** is the single most important measurement in the repo:
`recommended_stub` and `recommended_oracle` are the *same topology* run with one
knob changed (grounding mode). The gap isolates how much remaining failure is due
to **grounding quality** specifically, versus **graph structure**. The
`GraphRevisionAdvisor` reads that gap and, when oracle beats stub by enough, flags
grounding as the bottleneck and recommends building a real grounding agent -
rather than blaming topology.

This is the empirical loop working as intended: the system measures something it
did *not* presuppose.

---

## 7. The Edge-Driven Wiring Refactor (in flight)

> **Status: planned / in progress - NOT yet merged.** The code today still uses
> the hardcoded wiring described in Section 4. This section explains what is
> changing and why, so the design is understandable end to end.

### The problem being solved

As noted in Section 4, dependencies are described twice: once in the graph's
edges (which only control *order*), and once hardcoded in the executor (which
controls *what data flows*). Two sources of truth for the same fact, and the
hardcoded one silently wins. This means the graph spec does **not** fully describe
a run - the real data-flow is hidden in Python.

The goal of the refactor: **make the spec the single source of truth for both
order *and* the data that flows between nodes.** After it, the executor becomes a
true graph interpreter - the `GraphSpec` is the program, and the executor just
runs whatever graph it's handed.

### Change 1 - every agent gets one uniform shape

Today the agents have mismatched signatures (`perceive` takes 1 argument, `plan`
takes 3, `act` wants the registry, `learn` wants the whole trace). You cannot make
edges drive wiring while every agent expects different positional arguments. So
every agent is collapsed to a single shape:

```python
def run(self, ctx: NodeInput) -> SomeResult: ...
```

`NodeInput` is a small bundle the executor assembles per node. It has two kinds of
fields:

- **Ambient** (always available): `user_input`, `registry`, `row`, and - for the
  `learn` node only - `trace_so_far`.
- **Edge-supplied** (filled *only* if an incoming edge provides them):
  `perception`, `reasoning`, `grounding`, `readiness`, `plan`, `action`.

### Change 2 - `monolithic` stops being a node

`monolithic` was never a real cognitive stage - it's just "plan then act with no
upstream cognition," which is simply a **graph shape**. So it's re-expressed as a
two-node graph (`plan -> act`, no upstream edges) via `make_monolithic_baseline()`,
and the `monolithic` node role becomes a fail-loud tripwire (same pattern as
`memory`). A safety-net equivalence test asserts the old monolithic code path and
the new two-node graph produce *identical* results on every dev row - so the floor
baseline the oracle gap is measured against doesn't silently shift.

### Change 3 - edges carry the payload

`EdgeSpec` gains a `provides` field naming what flows along that edge:

```python
class EdgeSpec(BaseModel):
    from_node: str
    to_node: str
    provides: str | None = None   # None = "whatever this role canonically produces"
    condition: str | None = None
```

`provides=None` is a *deliberate* default meaning "carry the upstream role's
canonical output," resolved through a `ROLE_OUTPUT` map (`perceive -> perception`,
`reason -> reasoning`, ...). This rests on a documented, load-bearing invariant:
**one canonical output slot per role.** The day a role produces two outputs,
`provides=None` becomes ambiguous for that role and an explicit value is required.
An explicit `provides` is validated against `ROLE_OUTPUT` so it can never
contradict the role.

### Change 4 - a load-time wiring validator

Before a single node runs, `validate_wiring()` checks the graph against a
`ROLE_INPUTS` map: every input a role *requires* must be supplied by some incoming
edge, no two edges may supply the same slot to the same node, and a few structural
warnings (e.g. a node placed after `learn`, which would make its output invisible
to `learn`'s `trace_so_far`). A misconfigured graph now **refuses to start**
instead of quietly producing a broken trace.

> **Honest limitation:** in the first version, only `act`'s dependency on `plan`
> is marked *required*; the rest are *optional* because the stub agents tolerate
> `None`. So the validator catches broken `act` nodes but does not yet *enforce*
> full-pipeline correctness. A full-pipeline equivalence test is the primary
> safety net for that until `ROLE_INPUTS` is tightened.

### Change 5 - the dispatch flip

The big `if role == "perceive" ... elif role == "reason" ...` ladder collapses
into one generic step:

```python
def _dispatch(self, node, run_ctx, cfg, graph):
    node_input = self._build_node_input(node, run_ctx, graph)  # gather from incoming edges
    agent      = self._make_agent(node.role, cfg)              # role -> agent (+ mode/adapter)
    result     = agent.run(node_input)
    setattr(run_ctx, ROLE_OUTPUT[node.role], result)
```

`_build_node_input` looks at the node's **incoming edges**, gathers exactly those
payloads, and builds a `NodeInput` containing only them plus the ambient fields.
That last part is the real prize: **a node can only see the upstreams its edges
declare.** No more hidden god-object; the graph defines the data-flow.

### Before / after, in one line

| Concern            | Today (role-ladder)                         | After (edge-driven)                         |
| ------------------ | ------------------------------------------- | ------------------------------------------- |
| Execution order    | from edges (Kahn's algorithm)               | from edges (unchanged)                      |
| Data between nodes | **hardcoded** in `_dispatch`                | **from edges** (`provides` + `ROLE_INPUTS`) |
| Agent signatures   | heterogeneous, positional                   | uniform `run(ctx: NodeInput)`               |
| `monolithic`       | special node branch                         | a plain `plan -> act` graph                 |
| Bad graph          | runs, silently wrong                        | rejected at load by the validator           |

### The `learn` invariant (why it's safe)

`learn`'s `trace_so_far` is **never** edge-supplied - it is always computed from
the accumulated `RunContext` at the moment `learn` dispatches. That is correct
*only because* `learn` is topologically last (every prior result has landed) and
`to_trace()` reads the full run context. This invariant is documented in the code
and guarded by the validator, because the narrowing in this refactor must never
strip it.

---

## 8. Roadmap: introducing LLMs, one node at a time

The model seam (Section 4) becomes real *after* the edge-driven refactor lands,
in measurability order:

1. **Phase 0 - make the seam real:** one shared `ModelAdapter`, an injectable
   per-node config, and a dual-run harness (stub vs LLM scored against existing
   ground truth). A failing test that asserts "an injected adapter is actually
   called" is the green gate.
2. **`perceive`** first - clean entry point, easy to score against extracted
   entity mentions.
3. **`grounding`** next - the node your own analysis flags as structurally hard;
   the first node expected to *beat* the stub, not just match it.
4. **`reason`**, then **`readiness`** (kept hybrid: hard policy stays
   deterministic, LLM only judges fuzzy confirmation), then **`plan`/`act`**.
5. **`learn`** last - least measurable, most experimental.

A node only graduates from stub to LLM-default when it **beats the stub on its own
metric** across the dev set. The stub stays forever as the cheap deterministic
regression baseline.

---

## 9. Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 1. Convert tau-style simulation traces into cognitive dataset artifacts
python scripts/convert_traces.py --input data/raw/results.json --out-dir data/out/

# 2. Build cognitive topology reports
python scripts/build_reports.py --input data/out/ --out-dir data/out/

# 3. Recommend a cognitive graph from the report
python scripts/recommend_graph.py \
  --report data/out/cognitive_dataset_report.json \
  --out reports/recommended_graph.json

# 4. Evaluate graphs against each other (monolithic / minimal / stub / oracle)
python scripts/run_graph_evaluation.py \
  --recommended reports/recommended_graph.json \
  --report      reports/cognitive_dataset_report.json \
  --out-dir     data/out --reports-dir reports

# Other entry points
python scripts/run_baseline.py --dataset data/dev/tool_calling_micro.jsonl
python scripts/run_lab.py      --dataset data/dev/tool_calling_micro.jsonl
python scripts/evaluate_trace.py

pytest
```

Requires Python >= 3.11. Runtime dependencies: `pydantic>=2.0`, `rich>=13.0`.

> **Note:** the converter reads tau-bench-style traces you supply at
> `data/raw/results.json` (not committed). Small ready-to-run datasets ship under
> `data/dev/` and `data/test/` for the baseline and lab scripts.

---

## 10. Repository Structure

```
src/cognitive_tool_agent/
  schemas/            Pydantic contracts for every stage, artifact, and graph
  trace_converter/    Deterministic tau-style trace -> cognitive artifacts
  reports/            Cognitive topology report builders
  recommender/        Signal extraction, capability inference, graph recommendation,
                      and the revision advisor
  agents/             Cognitive stage agents (perceive ... learn); stub today,
                      model_adapter seam for future LLM backends
  graph/              GraphExecutor + RunContext (node-driven execution)
  graph_runner/       Multi-graph evaluation harness + trace writer
  graph_builder/      Lab loop (profiler, candidate generator, optimizer)
  evals/              Metrics and evaluator
  tools/              Tool registry + fake tools for offline runs
  datasets/           JSONL dataset loader

data/   raw/ (your input traces), out/ (artifacts+reports), dev/, test/
scripts/  CLI entry points          tests/  pytest suite
```

---

## 11. Evaluation Metrics

| Metric                  | Stage     | Description                             |
| ----------------------- | --------- | --------------------------------------- |
| `end_to_end_success`    | -         | Final action matches expected           |
| `tool_name_accuracy`    | plan      | Correct tool selected                   |
| `argument_exact_match`  | act       | Arguments exactly match expected        |
| `policy_violation_rate` | readiness | Fraction of rows with policy violations |
| `stage_failure_rate`    | -         | Fraction with any stage failure         |

---

## 12. Design Principle

> **No optimization before measurement.**

Datasets are treated as behavior-space specifications, not example collections.
Metrics are selection pressure. Graph search is driven by measured failures, not
prompt intuition. Every change - including every LLM node - must beat the stub it
replaces on a metric you can see.

### Long-term goal

A measurable cognitive operating system for production agents: datasets reveal
required capabilities, reports expose cognitive topology, graph recommendations
are evidence-based, each node has explicit typed contracts, failures map to
specific stages, and optimizers search over a measurable behavior space.
