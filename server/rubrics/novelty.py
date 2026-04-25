class NoveltyRubric:
    """Rubric for action novelty."""
    name = "novelty"
    version = "1.0"

    def compute(self, trajectory) -> float:
        """Compute novelty score.

        Args:
            trajectory: List of (prompt, response, reward_dict).

        Returns:
            Score between 0 and 1.
        """
        # Stub: 1 - similarity
        return 0.5