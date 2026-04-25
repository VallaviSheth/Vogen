import pytest
from server.rubrics import RUBRICS

@pytest.mark.rubrics
def test_rubrics():
    for name, cls in RUBRICS.items():
        r = cls()
        score = r.compute([("prompt", "response", {"critic_score": 0.5})])
        assert 0 <= score <= 1