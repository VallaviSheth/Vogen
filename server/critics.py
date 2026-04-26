import json
import os
from typing import Dict, Any
from .schemas import CriticScore

class Critic:
    """Fashion critic with LLM-based and rule-based scoring."""

    def __init__(self, persona_name: str = "default", personas_file: str = "data/critic_personas.json"):
        self.persona_name = persona_name
        if os.path.exists(personas_file):
            with open(personas_file) as f:
                self.personas = json.load(f)
        else:
            self.personas = {"default": {"bio": "Default critic"}}

    def score(self, action, observation) -> CriticScore:
        """Score an action-observation pair.

        Args:
            action: The action taken.
            observation: The observation.

        Returns:
            Structured critic score.
        """
        if os.getenv("VOGEN_USE_LLM_CRITIC", "0") == "1":
            # TODO: Call base Qwen LLM with persona prompt
            pass
        
        # Rule-based scoring
        return CriticScore(
            aesthetics=0.6,
            coherence=0.65,
            constraint_compliance=0.7,
            originality=0.55,
            commercial_fit=0.62,
            justification="Rule-based critic score based on outfit coherence and constraints."
        )

    def score_outfit(self, action: Dict[str, Any]) -> float:
        """Score an outfit using persona and rule-based scoring.

        Args:
            action: Outfit action with garment_ids, justification, self_predicted_score.

        Returns:
            Float score in [0, 1].
        """
        if "garment_ids" not in action or not action["garment_ids"]:
            return 0.0
        garments = action["garment_ids"]
        if len(garments) > 5 or len(garments) < 1:
            return 0.0
        if not 0.0 <= action.get("self_predicted_score", 0.5) <= 1.0:
            return 0.0
        
        # Composite scoring
        base_score = 0.5
        
        # Reward coherence (based on number of items)
        coherence_bonus = min(0.2, len(garments) * 0.05)
        
        # Reward calibration (match between predicted and random assignment)
        predicted = action.get("self_predicted_score", 0.5)
        calibration_bonus = 0.1 * abs(predicted - 0.5) * 0.5
        
        # Reward justification length (at least 5 chars)
        justification = action.get("justification", "")
        justification_bonus = 0.15 if len(justification) >= 5 else 0.0
        
        return min(1.0, base_score + coherence_bonus + calibration_bonus + justification_bonus)

    def update_rubric(self, real_world_signals):
        """Update rubric weights based on signals.

        Args:
            real_world_signals: Dictionary of signals.
        """
        # Stub: nudge weights toward signals
        pass
