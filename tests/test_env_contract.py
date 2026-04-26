import pytest
from server.env import VogenEnv
from server.schemas import Observation, Outfit, StepResult

@pytest.mark.asyncio
async def test_env_methods():
    env = VogenEnv()
    assert hasattr(env, 'reset')
    assert hasattr(env, 'step')
    assert hasattr(env, 'state')
    assert hasattr(env, 'close')
    assert hasattr(env, 'score')

@pytest.mark.asyncio
async def test_reset_deterministic():
    env = VogenEnv()
    obs1 = await env.reset({"seed": 42}, "session1")
    obs2 = await env.reset({"seed": 42}, "session2")
    assert obs1 == obs2

@pytest.mark.asyncio
async def test_return_types():
    env = VogenEnv()
    obs = await env.reset({}, "session1")
    assert isinstance(obs, Observation)
    action = Outfit(garment_ids=[], justification="test", self_predicted_score=0.5)
    result = await env.step(action, "session1")
    assert isinstance(result, StepResult)
    reward = await env.score([], "session1")
    assert hasattr(reward, 'critic')

@pytest.mark.asyncio
async def test_session_isolation():
    env = VogenEnv()
    await env.reset({"seed": 1}, "session_a")
    await env.reset({"seed": 2}, "session_b")
    action_a = Outfit(garment_ids=["item1"], justification="A", self_predicted_score=0.3)
    action_b = Outfit(garment_ids=["item2"], justification="B", self_predicted_score=0.7)
    await env.step(action_a, "session_a")
    await env.step(action_b, "session_b")
    state_a = await env.state("session_a")
    state_b = await env.state("session_b")
    assert state_a != state_b

def test_client_inherits_environment():
    from client.vogen_client import VogenClient
    from openenv.core import Environment
    client = VogenClient.from_url("http://test")
    assert isinstance(client, Environment)

def test_score_returns_five_fields():
    from client.vogen_client import VogenClient
    from client.models import Reward
    from unittest.mock import patch

    client = VogenClient.from_url("http://test")
    with patch.object(client, '_make_request', return_value={
        'reward': {
            'critic': 0.1,
            'novelty': 0.2,
            'calibration': 0.3,
            'teaching': 0.4,
            'difficulty': 0.5
        }
    }):
        reward = client.score([])
        assert isinstance(reward, Reward)
        assert reward.dict() == {
            'critic': 0.1,
            'novelty': 0.2,
            'calibration': 0.3,
            'teaching': 0.4,
            'difficulty': 0.5
        }

def test_state_returns_typed_state():
    from client.vogen_client import VogenClient
    from client.models import State
    from unittest.mock import patch

    client = VogenClient.from_url("http://test")
    with patch.object(client, '_make_request', return_value={
        'current_obs': {
            'brief': {'text': 'Hi', 'constraints': {}, 'occasion': 'casual', 'budget': 100.0, 'tier': 1},
            'wardrobe_handle': ['item1'],
            'context_vector': [0.0],
            'history': []
        }
    }):
        state = client.state()
        assert isinstance(state, State)
        assert state.current_obs is not None

def test_from_hub_no_hardcode():
    from client.vogen_client import VogenClient
    from unittest.mock import patch

    with patch('client.vogen_client.requests.get') as mock_get:
        mock_resp = mock_get.return_value
        mock_resp.status_code = 200
        mock_resp.json.return_value = {'runtime': {'url': 'https://test.space'}}
        client = VogenClient.from_hub("test/repo")
        assert "localhost" not in client.url
