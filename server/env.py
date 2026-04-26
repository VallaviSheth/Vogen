import logging
from .schemas import Observation, Action, StepResult, Brief, Reward
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class VogenEnv:
    """Vogen environment for fashion styling tasks."""

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    async def reset(self, task_spec: dict, session_id: str) -> Observation:
        """Reset the environment with a task specification.

        Args:
            task_spec: Dictionary containing task parameters, including seed.
            session_id: Session identifier.

        Returns:
            Initial observation.
        """
        # Stub implementation
        seed = task_spec.get("seed", 0)
        obs = Observation(
            brief=Brief(text="Style an outfit for a casual occasion.", constraints={}, occasion="casual", budget=100.0, tier=1),
            wardrobe_handle=["item1", "item2"],
            context_vector=[0.1, 0.2, 0.3],
            history=[]
        )
        self.sessions[session_id] = {"obs": obs, "history": []}
        return obs

    async def step(self, action: Action, session_id: str) -> StepResult:
        """Take a step in the environment.

        Args:
            action: The action to take.
            session_id: Session identifier.

        Returns:
            Step result with observation, reward, done, info.
        """
        from .critics import Critic
        session = self.sessions.setdefault(session_id, {})
        obs = session.get("obs")
        if obs is None:
            obs = Observation(
                brief=Brief(text="Style an outfit for a casual occasion.", constraints={}, occasion="casual", budget=100.0, tier=1),
                wardrobe_handle=["item1", "item2"],
                context_vector=[0.1, 0.2, 0.3],
                history=[]
            )
            session["obs"] = obs
        history = session.setdefault("history", [])
        if hasattr(action, 'model_dump'):
            action_dict = action.model_dump()
        else:
            action_dict = action
        history.append(action_dict)
        session["history"] = history
        obs.history = history
        
        # Real critic scoring
        critic = Critic(persona_name="default")
        critic_score = critic.score_outfit(action_dict)
        
        # Composite reward
        reward = Reward(
            critic=max(0.0, min(1.0, critic_score)),
            novelty=0.4 + 0.2 * (len(history) * 0.1),  # Increases slightly with episode length
            calibration=abs(action_dict.get('self_predicted_score', 0.5) - critic_score),
            teaching=0.3 + 0.4 * critic_score,  # Teaching signal based on critic
            difficulty=0.5 * (1.0 + 0.1 * len(history))  # Scales with complexity
        )
        
        done = len(history) >= 3  # Episode ends after 3 actions
        info = {
            "critics": {
                "Vignette": critic_score * 0.9,
                "Orin": critic_score * 0.85,
                "Madame Liu": critic_score * 0.95,
                "Kestrel": critic_score * 0.88,
                "Null": critic_score * 0.92
            },
            "episode_length": len(history)
        }
        return StepResult(observation=obs, reward=reward, done=done, info=info)

    async def state(self, session_id: str) -> dict:
        """Get the current state of the environment.

        Args:
            session_id: Session identifier.

        Returns:
            State dictionary.
        """
        session = self.sessions.get(session_id, {})
        return {"current_obs": session.get("obs").model_dump() if session.get("obs") else None}

    async def close(self, session_id: str) -> None:
        """Close the environment."""
        self.sessions.pop(session_id, None)

    async def score(self, traj: List[Dict[str, Any]], session_id: str) -> Reward:
        """Score a trajectory.

        Args:
            traj: Trajectory list.
            session_id: Session identifier.

        Returns:
            Reward dict.
        """
        # Stub
        return Reward(critic=0.5, novelty=0.5, calibration=0.5, teaching=0.5, difficulty=0.5)