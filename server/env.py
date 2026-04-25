import logging
from .schemas import Observation, Action, StepResult, Brief

logger = logging.getLogger(__name__)

class VogenEnv:
    """Vogen environment for fashion styling tasks."""

    def __init__(self):
        self.current_obs = None

    async def reset(self, task_spec: dict) -> Observation:
        """Reset the environment with a task specification.

        Args:
            task_spec: Dictionary containing task parameters, including seed.

        Returns:
            Initial observation.
        """
        # Stub implementation
        seed = task_spec.get("seed", 0)
        self.current_obs = Observation(
            brief=Brief(text="Style an outfit for a casual occasion.", constraints={}, occasion="casual", budget=100.0, tier=1),
            wardrobe_handle=["item1", "item2"],
            context_vector=[0.1, 0.2, 0.3],
            history=[]
        )
        return self.current_obs

    async def step(self, action: Action) -> StepResult:
        """Take a step in the environment.

        Args:
            action: The action to take.

        Returns:
            Step result with observation, reward, done, info.
        """
        # Stub
        reward_dict = {"critic_score": 0.5, "justification": 0.5, "novelty": 0.5, "difficulty": 0.5, "calibration": 0.5}
        done = True
        info = {}
        return StepResult(observation=self.current_obs, reward_dict=reward_dict, done=done, info=info)

    async def state(self) -> dict:
        """Get the current state of the environment.

        Returns:
            State dictionary.
        """
        return {"current_obs": self.current_obs.model_dump() if self.current_obs else None}

    async def close(self) -> None:
        """Close the environment."""
        pass