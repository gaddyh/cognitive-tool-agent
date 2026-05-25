from pydantic import BaseModel
from .common import Confidence


class ReadinessResult(BaseModel):
    ready: bool
    blocking_reasons: list[str] = []
    policy_violations: list[str] = []
    missing_required_fields: list[str] = []
    confidence: Confidence
