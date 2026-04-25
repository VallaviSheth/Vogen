from typing import Dict, List, Tuple
from server.rubrics import RUBRICS

class RewardAggregator:
    """Aggregates rubrics into scalar reward."""

    def __init__(self, weights: Dict[str, float]):
        self.weights = weights
        self.rubrics = {name: cls() for name, cls in RUBRICS.items()}

    def aggregate(self, trajectory: List) -> Tuple[float, Dict[str, float]]:
        """Aggregate rewards.

        Args:
            trajectory: List of (prompt, response, reward_dict).

        Returns:
            Total reward, per-rubric dict.
        """
        scores = {}
        for name, rubric in self.rubrics.items():
            scores[name] = rubric.compute(trajectory)
        total = sum(scores.get(name, 0) * self.weights.get(name, 0) for name in scores)
        return total, scores