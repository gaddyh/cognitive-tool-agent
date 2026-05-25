from __future__ import annotations

# ---------------------------------------------------------------------------
# Capability inference thresholds (v1 — hardcoded, config-injectable later)
# ---------------------------------------------------------------------------
# Each constant is a fraction in [0, 1] unless noted otherwise.
# These will become a ThresholdConfig(BaseModel) in a future iteration once
# the semantics stabilise across multiple datasets.

# memory: fraction of arg instances that arrive via tool-chaining
# above this level, a persistent memory/state node is warranted
MEMORY_CHAINING_THRESHOLD: float = 0.50

# grounding: fraction of arg instances that require grounding (NL→structured)
# above this level, an explicit grounding stage is warranted
GROUNDING_THRESHOLD: float = 0.40

# readiness: max(write_fraction, write_failure_fraction)
# above this level, a readiness/judge gate is warranted
# write_fraction  = write_calls / total_calls
# write_failure_fraction = write-type failures / total failures
READINESS_WRITE_FRACTION_THRESHOLD: float = 0.15

# deep_planning: average chain depth (number of preceding tool calls) at call time
# above this level, an explicit reasoning/planning node is warranted
REASONING_DEPTH_THRESHOLD: float = 2.5

# grounding: minimum instance count for an arg to qualify for peak_grounding_strength
# prevents rare args (e.g. summary: 2 instances, 100% grounding) from dominating the peak
MIN_GROUNDING_INSTANCES: int = 10
