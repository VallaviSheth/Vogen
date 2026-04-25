class JustificationRubric:
    """Rubric for justification quality."""
    name = "justification"
    version = "1.0"

    def compute(self, trajectory) -> float:
        """Compute justification score.

        Args:
            trajectory: List of (prompt, response, reward_dict).

        Returns:
            Score between 0 and 1.
        """
        # Stub: check if justification cites items
        if not trajectory:
            return 0.0
        response = trajectory[-1][1]
        if "garment_" in response:
            return 0.8
        return 0.3