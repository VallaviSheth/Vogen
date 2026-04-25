class DifficultyRubric:
    """Rubric for difficulty multiplier."""
    name = "difficulty"
    version = "1.0"

    def compute(self, trajectory) -> float:
        """Compute difficulty score.

        Args:
            trajectory: List of (prompt, response, reward_dict).

        Returns:
            Score between 0 and 1.
        """
        # Tier scaled
        tier = 1  # stub
        return min(0.6 + 0.1 * tier, 1.2)