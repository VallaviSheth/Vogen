import pytest
import numpy as np
from server.runway import Wardrobe, CulturalContext

@pytest.mark.runway
def test_wardrobe_determinism():
    w1 = Wardrobe(42)
    garments1 = w1.generate(100)
    w2 = Wardrobe(42)
    garments2 = w2.generate(100)
    assert len(garments1) == 100
    assert garments1 == garments2

@pytest.mark.runway
def test_cultural_drift_determinism():
    c1 = CulturalContext("data/cultural_priors.json", seed=42)
    vectors1 = [c1.get_vector(e) for e in range(10)]
    c2 = CulturalContext("data/cultural_priors.json", seed=42)
    vectors2 = [c2.get_vector(e) for e in range(10)]
    assert vectors1 == vectors2
    # Check bounded drift
    initial = np.array(c1.priors["initial_vector"])
    for v in vectors1:
        diff = np.linalg.norm(np.array(v) - initial)
        assert diff <= 10 * 0.05