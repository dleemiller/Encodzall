# encodzall/config/noise_config.py
from dataclasses import dataclass


@dataclass
class NoiseConfig:
    """
    Configuration for noise and masking in the ByteLevelTokenizer.

    Attributes:
        prob (float): Total probability for noise and masking.
        mask_ratio (float): Ratio of masking within the total probability.
        noise_ratio (float): Ratio of noise within the total probability.
    """

    prob: float = 0.0
    mask_ratio: float = 0.2
    noise_ratio: float = 0.8

    def set_prob(self, prob: float):
        """
        Set the total noise and masking probability.

        Args:
            prob (float): Total probability for noise and masking.
        """
        self.prob = prob

    @property
    def noise_prob(self) -> float:
        """Calculate the noise probability based on the total prob and noise ratio."""
        return self.prob * self.noise_ratio

    @property
    def mask_prob(self) -> float:
        """Calculate the mask probability based on the total prob and mask ratio."""
        return self.prob * self.mask_ratio
