from pydantic import BaseModel
from typing import List, Dict, Any, Union, Optional

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
    observation: 'Observation'
    reward: Reward
    done: bool
    info: Dict[str, Any]

class Observation(BaseModel):
    brief: Dict[str, Any]
    wardrobe_handle: List[str]
    context_vector: List[float]
    history: List[Dict[str, Any]]

class State(BaseModel):
    current_obs: Optional[Observation] = None

StepResult.model_rebuild()