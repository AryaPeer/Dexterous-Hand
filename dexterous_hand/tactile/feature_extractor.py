import gymnasium
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch

from dexterous_hand.tactile.encoder import TactileEncoder


class TactileFeatureExtractor(BaseFeaturesExtractor):
    """SB3 feature extractor: splits proprio from tactile, encodes tactile via CNN.

    Obs split at proprio_dim — proprio passes through, current tactile (80 dims)
    gets encoded to 32 features. Output: (batch, proprio_dim + 32).
    """

    def __init__(
        self,
        observation_space: gymnasium.spaces.Box,
        proprio_dim: int = 125,
        tactile_dim: int = 240,
    ) -> None:
        """Split obs into proprio + tactile with CNN encoding.

        @param observation_space: SB3 obs space (flat Box)
        @type observation_space: gymnasium.spaces.Box
        @param proprio_dim: proprioceptive dims
        @type proprio_dim: int
        @param tactile_dim: tactile dims (current + prev + change)
        @type tactile_dim: int
        """

        features_dim = proprio_dim + 32
        super().__init__(observation_space, features_dim=features_dim)

        self.proprio_dim = proprio_dim
        self.tactile_start = proprio_dim
        self.tactile_current_end = proprio_dim + 80

        self.tactile_encoder = TactileEncoder()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """(batch, 371) -> (batch, 163) proprio + encoded tactile."""

        proprio = observations[:, : self.proprio_dim]

        tactile_current = observations[
            :, self.tactile_start : self.tactile_current_end
        ]

        tactile_features = self.tactile_encoder(tactile_current)

        return torch.cat([proprio, tactile_features], dim=1)
