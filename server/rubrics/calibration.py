class CalibrationRubric:
    """Rubric for self-critique calibration."""
    name = "calibration"
    version = "1.0"

    def compute(self, trajectory) -> float:
        """Compute calibration score.

        Args:
            trajectory: List of (prompt, response, reward_dict).

        Returns:
            Score between 0 and 1.
        """
        # 1 - |self_pred - actual|
        if not trajectory:
            return 0.0
        self_pred = 0.5  # stub
        actual = trajectory[-1][2].get("critic_score", 0.5)
        return 1 - abs(self_pred - actual)