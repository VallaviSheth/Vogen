import requests
import time
import uuid
from typing import Dict, Any, Tuple, List, Optional
from openenv.core import Environment
from .models import Action, Reward, StepResult, Observation, State

class VogenClient(Environment):
    """Vogen client inheriting from MCPEnvironment."""

    def __init__(self, url: str, session_id: Optional[str] = None, timeout: float = 60.0):
        super().__init__()
        self.url = url
        self.session_id = session_id or str(uuid.uuid4())
        self.timeout = timeout

    @classmethod
    def from_hub(cls, repo_id: str) -> 'VogenClient':
        # Resolve HF Space URL via Hugging Face Spaces metadata or URL pattern.
        owner, name = repo_id.split('/')
        api_url = f"https://huggingface.co/api/spaces/{owner}/{name}"
        try:
            resp = requests.get(api_url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            runtime_url = data.get('runtime', {}).get('url')
            if runtime_url:
                return cls(runtime_url)
        except requests.exceptions.RequestException:
            pass
        return cls(f"https://{owner}-{name}.hf.space")

    @classmethod
    def from_url(cls, url: str) -> 'VogenClient':
        return cls(url)

    def _make_request(self, method: str, endpoint: str, json_data: Optional[Dict] = None, long_running: bool = False) -> Dict[str, Any]:
        url = f"{self.url}{endpoint}"
        timeout = 21600 if long_running else self.timeout  # 6h for long running
        for attempt in range(3):
            try:
                resp = requests.request(method, url, json=json_data, timeout=timeout, params={'session_id': self.session_id})
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)  # backoff

    def reset(self, task_spec: Dict[str, Any]) -> Observation:
        data = self._make_request('POST', '/reset', task_spec)
        self.session_id = data.get('session_id', self.session_id)
        return Observation(**data['observation'])

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        payload = action.dict()
        data = self._make_request('POST', '/step', payload)
        step_result = StepResult(**data)
        return step_result.observation, step_result.reward, step_result.done, step_result.info

    def state(self) -> 'State':
        data = self._make_request('GET', '/state')
        from .models import State
        return State(**data)

    def close(self) -> None:
        self._make_request('POST', '/close')

    def score(self, traj: List[Dict[str, Any]]) -> Reward:
        data = self._make_request('POST', '/score', {'traj': traj})
        return Reward(**data['reward'])

    def vogen_style(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._make_request('POST', '/tools/vogen.style', params)

    def vogen_negotiate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._make_request('POST', '/tools/vogen.negotiate', params)

    def vogen_predict(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._make_request('POST', '/tools/vogen.predict', params)

    def vogen_evolve(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._make_request('POST', '/tools/vogen.evolve', params, long_running=True)