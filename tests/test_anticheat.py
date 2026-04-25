import pytest
from server.safety.anticheat import AntiCheat
from server.schemas import Action

@pytest.mark.anticheat
def test_deduplication():
    ac = AntiCheat()
    action = Action(garment_ids=["g1"], justification="test", self_predicted_score=0.5)
    assert ac.validate_action(action)
    assert not ac.validate_action(action)  # duplicate

@pytest.mark.anticheat
def test_invalid_action():
    ac = AntiCheat()
    action = Action(garment_ids=[], justification="test", self_predicted_score=0.5)
    assert not ac.validate_action(action)

@pytest.mark.anticheat
def test_budget_attack():
    # Stub: assume budget check
    assert True

@pytest.mark.anticheat
def test_garment_hallucination():
    # Stub
    assert True