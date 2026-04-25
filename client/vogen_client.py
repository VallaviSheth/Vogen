import requests
from typing import Dict, Any

class VogenClient:
    """Thin HTTP client for Vogen environment, Gym-style."""

    def __init__(self, url: str):
        self.url = url

    @classmethod
    def from_hub(cls, repo_id: str) -> 'VogenClient':
        # Stub: assume repo_id maps to URL
        return cls("http://localhost:8000")

    @classmethod
    def from_url(cls, url: str) -> 'VogenClient':
        return cls(url)

    def reset(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Reset the environment.

        Args:
            task_spec: Task specification.

        Returns:
            Initial observation as dict.
        """
        resp = requests.post(f"{self.url}/reset", json=task_spec)
        resp.raise_for_status()
        return resp.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Take a step.

        Args:
            action: Action as dict.

        Returns:
            Step result as dict.
        """
        resp = requests.post(f"{self.url}/step", json=action)
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        """Get state.

        Returns:
            State as dict.
        """
        resp = requests.get(f"{self.url}/state")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        """Close the environment."""
        requests.post(f"{self.url}/close")