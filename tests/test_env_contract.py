import pytest
from server.env import VogenEnv
from server.schemas import Observation, Action, StepResult

@pytest.mark.asyncio
async def test_env_methods():
    env = VogenEnv()
    assert hasattr(env, 'reset')
    assert hasattr(env, 'step')
    assert hasattr(env, 'state')
    assert hasattr(env, 'close')

@pytest.mark.asyncio
async def test_reset_deterministic():
    env = VogenEnv()
    obs1 = await env.reset({"seed": 42})
    obs2 = await env.reset({"seed": 42})
    assert obs1 == obs2

@pytest.mark.asyncio
async def test_return_types():
    env = VogenEnv()
    obs = await env.reset({})
    assert isinstance(obs, Observation)
    action = Action(garment_ids=[], justification="test", self_predicted_score=0.5)
    result = await env.step(action)
    assert isinstance(result, StepResult)