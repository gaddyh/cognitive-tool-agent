from pydantic import BaseModel
from .common import UserInput
from .grounding import GroundingResult
from .perceive import PerceptionResult
from .reason import ReasoningResult
from .readiness import ReadinessResult
from .plan import PlanResult
from .act import ActionResult
from .learn import LearningResult


class CognitiveTrace(BaseModel):
    input: UserInput

    perception: PerceptionResult | None = None
    reasoning: ReasoningResult | None = None
    grounding: GroundingResult | None = None
    readiness: ReadinessResult | None = None
    plan: PlanResult | None = None
    action: ActionResult | None = None
    learning: LearningResult | None = None
