from typing import Literal
from pydantic import BaseModel


class FailureAnalysis(BaseModel):
    failed_stage: Literal[
        "none",
        "perception",
        "reasoning",
        "readiness",
        "planning",
        "acting",
        "unknown",
    ]

    failure_type: str | None = None
    explanation: str = ""


class LearningResult(BaseModel):
    should_add_to_dataset: bool
    dataset_split_suggestion: Literal["train", "dev", "test", "none"] = "none"

    failure_analysis: FailureAnalysis
    regression_tags: list[str] = []
    optimization_target: str | None = None
