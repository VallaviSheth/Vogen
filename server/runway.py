import numpy as np
import json
from .schemas import Garment

class Wardrobe:
    """Deterministic wardrobe generator."""

    def __init__(self, seed: int):
        self.rng = np.random.Generator(np.random.PCG64(seed))
        with open("data/seed_wardrobe.json") as f:
            self.priors = json.load(f)

    def generate(self, n: int) -> list[Garment]:
        """Generate n garments deterministically.

        Args:
            n: Number of garments.

        Returns:
            List of garments.
        """
        garments = []
        for i in range(n):
            silhouette = self.rng.choice(self.priors["silhouettes"])
            fabric = self.rng.choice(self.priors["fabrics"])
            color_lab = [
                self.rng.uniform(0, 100),
                self.rng.uniform(-128, 127),
                self.rng.uniform(-128, 127)
            ]
            era = self.rng.choice(self.priors["eras"])
            origin = self.rng.choice(self.priors["origins"])
            price_tier = int(self.rng.choice(self.priors["price_tiers"]))
            garment = Garment(
                id=f"garment_{i}",
                silhouette=silhouette,
                fabric=fabric,
                color_lab=color_lab,
                era=era,
                origin=origin,
                price_tier=price_tier
            )
            garments.append(garment)
        return garments

class Market:
    """Market with drifting demand curves."""

    def __init__(self, seed: int):
        self.rng = np.random.Generator(np.random.PCG64(seed))
        # 10 demand centers in a simple feature space
        self.centers = [self.rng.uniform(0, 1, 10) for _ in range(10)]

    def drift(self):
        """Drift the demand centers."""
        for center in self.centers:
            center += self.rng.normal(0, 0.01, 10)

class CulturalContext:
    """Cultural context vector with bounded drift."""

    def __init__(self, priors_file: str, seed: int):
        with open(priors_file) as f:
            self.priors = json.load(f)
        self.seed = seed

    def get_vector(self, epoch: int) -> list[float]:
        """Get the cultural vector at a given epoch.

        Args:
            epoch: Epoch number.

        Returns:
            32-dim vector.
        """
        rng = np.random.Generator(np.random.PCG64(self.priors["drift_schedule"]["seed"] + epoch))
        vector = np.array(self.priors["initial_vector"])
        for e in range(epoch):
            step = rng.normal(0, self.priors["drift_schedule"]["max_step"] / 3, 32)
            step = np.clip(step, -self.priors["drift_schedule"]["max_step"], self.priors["drift_schedule"]["max_step"])
            vector += step
        return vector.tolist()