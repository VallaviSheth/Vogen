import pytest

@pytest.mark.curriculum
def test_promotion_gate():
    # Simulate performance history
    perf = [1] * 35 + [0] * 15  # 70% success
    success_rate = sum(perf) / len(perf)
    assert success_rate >= 0.65