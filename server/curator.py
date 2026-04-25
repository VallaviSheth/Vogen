import numpy as np
from .task import Task
from .runway import Wardrobe, CulturalContext

class Curator:
    """Meta-agent that generates challenges."""

    def __init__(self, task: Task, runway):
        self.task = task
        self.runway = runway
        self.current_tier = 1
        self.perf_history = []

    def next_challenge(self, agent_perf_history: list) -> Brief:
        """Generate next challenge based on performance.

        Args:
            agent_perf_history: List of past performances.

        Returns:
            Task brief.
        """
        # Update tier based on promotion gate
        if len(agent_perf_history) >= 50:
            success_rate = sum(agent_perf_history[-50:]) / 50
            if success_rate >= 0.65:
                self.current_tier = min(self.current_tier + 1, 6)
            elif success_rate < 0.25:
                self.current_tier = max(self.current_tier - 1, 1)
        rng = np.random.Generator(np.random.PCG64(42))
        return self.task.sample_brief(rng, self.current_tier)