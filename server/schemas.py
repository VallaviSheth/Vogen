from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union, Optional

class Garment(BaseModel):
    """A garment item."""
    id: str
    silhouette: str
    fabric: str
    color_lab: List[float] = Field(min_length=3, max_length=3)
    era: str
    origin: str
    price_tier: int

class Brief(BaseModel):
    """Task brief."""
    text: str
    constraints: Dict[str, Any]
    occasion: str
    budget: float
    tier: int

class Observation(BaseModel):
    """Environment observation."""
    brief: Brief
    wardrobe_handle: List[str]
    context_vector: List[float]
    history: List[Dict[str, Any]]

class Outfit(BaseModel):
    garment_ids: List[str]
    justification: str
    self_predicted_score: float

class Prediction(BaseModel):
    predicted_score: float
    reasoning: str

class NegotiationMove(BaseModel):
    offer: Dict[str, Any]
    counter: Optional[Dict[str, Any]] = None

class DesignMutation(BaseModel):
    base_outfit: Outfit
    changes: List[Dict[str, Any]]

Action = Union[Outfit, Prediction, NegotiationMove, DesignMutation]

class Reward(BaseModel):
    critic: float
    novelty: float
    calibration: float
    teaching: float
    difficulty: float

class StepResult(BaseModel):
    """Step result."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]

class CriticScore(BaseModel):
    """Structured critic score."""
    aesthetics: float
    coherence: float
    constraint_compliance: float
    originality: float
    commercial_fit: float
    justification: str