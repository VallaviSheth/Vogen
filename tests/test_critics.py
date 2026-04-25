import pytest
from server.critics import Critic
from server.schemas import CriticScore

@pytest.mark.critics
def test_critic_score():
    c = Critic("data/critic_personas.json")
    score = c.score(None, None)
    assert isinstance(score, CriticScore)
    assert 0 <= score.aesthetics <= 1
    assert 0 <= score.coherence <= 1
    assert 0 <= score.constraint_compliance <= 1
    assert 0 <= score.originality <= 1
    assert 0 <= score.commercial_fit <= 1
    assert isinstance(score.justification, str)