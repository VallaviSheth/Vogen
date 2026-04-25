import numpy as np
from .schemas import Brief, Action, Observation, CriticScore
from .critics import Critic

class Task:
    """Unified styling task with tiered challenges."""

    def __init__(self, critics: Critic):
        self.critics = critics

    def sample_brief(self, rng: np.random.Generator, tier: int) -> Brief:
        """Sample a brief for the given tier.

        Args:
            rng: Random number generator.
            tier: Difficulty tier.

        Returns:
            Task brief.
        """
        if tier == 1:
            text = "Style a simple outfit for a casual occasion."
            constraints = {}
            occasion = "casual"
            budget = 100.0
        elif tier == 2:
            text = "Style an outfit under budget constraints."
            constraints = {"budget_strict": True}
            occasion = "formal"
            budget = 50.0
        elif tier == 3:
            text = "Style an outfit that trends with current culture."
            constraints = {"trend_aware": True}
            occasion = "party"
            budget = 150.0
        else:
            text = "Style an outfit with adversarial constraints."
            constraints = {"adversarial": True}
            occasion = "business"
            budget = 200.0
        return Brief(text=text, constraints=constraints, occasion=occasion, budget=budget, tier=tier)

    def validate_action(self, action: Action, observation: Observation) -> bool:
        """Validate action against anti-cheat checks.

        Args:
            action: Action to validate.
            observation: Current observation.

        Returns:
            True if valid.
        """
        # Garment IDs exist
        if not all(gid in observation.wardrobe_handle for gid in action.garment_ids):
            return False
        # Outfit size
        if not 1 <= len(action.garment_ids) <= 5:
            return False
        # Budget (stub, assume prices)
        total_price = len(action.garment_ids) * 20  # stub
        if total_price > observation.brief.budget:
            return False
        # Justification cites garment ID
        if not any(f"garment_{i}" in action.justification for i in range(len(observation.wardrobe_handle))):
            return False
        # For tier >=3, cite context dim
        if observation.brief.tier >= 3:
            if not any(f"dim_{i}" in action.justification for i in range(32)):
                return False
        # Self predicted score
        if not 0 <= action.self_predicted_score <= 1:
            return False
        return True

    def score_action(self, action: Action, observation: Observation) -> CriticScore:
        """Score the action.

        Args:
            action: Action.
            observation: Observation.

        Returns:
            Critic score.
        """
        return self.critics.score(action, observation)

    def is_terminal(self, history: list, tier: int) -> bool:
        """Check if episode is terminal.

        Args:
            history: Action history.
            tier: Current tier.

        Returns:
            True if terminal.
        """
        if tier <= 4:
            return len(history) >= 1
        elif tier == 5:
            return len(history) >= 3
        else:
            return len(history) >= 5