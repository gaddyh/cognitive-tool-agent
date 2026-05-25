from __future__ import annotations

from ..evals.evaluator import Evaluator
from ..graph.cognitive_graph import GraphExecutor
from ..schemas.dataset import DatasetRow
from ..schemas.experiment import ExperimentSpec
from ..schemas.graph_builder import GraphCandidate
from ..schemas.trace import CognitiveTrace
from ..tools.registry import ToolRegistry


class BaselineRunner:
    def __init__(self) -> None:
        self._executor = GraphExecutor()
        self._evaluator = Evaluator()

    def run(
        self,
        candidate: GraphCandidate,
        rows: list[DatasetRow],
        registry: ToolRegistry,
    ) -> tuple[list[CognitiveTrace], dict[str, float]]:
        experiment = ExperimentSpec(graph=candidate.graph_spec)
        traces: list[CognitiveTrace] = []
        for row in rows:
            trace = self._executor.run(experiment, row, registry)
            traces.append(trace)

        scores = self._evaluator.score(traces, rows)
        return traces, scores
