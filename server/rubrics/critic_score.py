class CriticScoreRubric:
    """Rubric for critic score."""
    name = "critic_score"
    version = "1.0"

    def compute(self, trajectory) -> float:
        """Compute critic score from trajectory.

        Args:
            trajectory: List of (prompt, response, reward_dict).

        Returns:
            Score between 0 and 1.
        """
        if not trajectory:
            return 0.0
        return trajectory[-1][2].get("critic_score", 0.5)