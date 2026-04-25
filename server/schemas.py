from pydantic import BaseModel, Field
from typing import List, Dict, Any

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

class Action(BaseModel):
    """Styling action."""
    garment_ids: List[str]
    justification: str
    self_predicted_score: float = Field(ge=0.0, le=1.0)

class StepResult(BaseModel):
    """Step result."""
    observation: Observation
    reward_dict: Dict[str, float]
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