import gymnasium
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch

from dexterous_hand.tactile.encoder import TactileEncoder


class TactileFeatureExtractor(BaseFeaturesExtractor):
    """SB3 feature extractor that splits proprio from tactile and runs
    the tactile part through a CNN encoder.

    The flat obs is split at proprio_dim: everything before goes through as-is,
    the next 80 dims (current tactile) get encoded to 32 features.
    """

    def __init__(
        self,
        observation_space: gymnasium.spaces.Box,
        proprio_dim: int = 125,
        tactile_dim: int = 240,
    ) -> None:
        """Set up the feature extractor.

        Args:
            observation_space: SB3 obs space (flat Box)
            proprio_dim: how many dims are proprioceptive
            tactile_dim: how many dims are tactile (current + prev + change)
        """
        features_dim = proprio_dim + 32
        super().__init__(observation_space, features_dim=features_dim)

        self.proprio_dim = proprio_dim
        self.tactile_start = proprio_dim
        self.tactile_current_end = proprio_dim + 80

        self.tactile_encoder = TactileEncoder()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Split obs into proprio and tactile, encode tactile, concatenate back.

        Args:
            observations: (batch, 365) flat observation tensor

        Returns:
            (batch, 157) proprio (125) + encoded tactile (32)
        """
        proprio = observations[:, : self.proprio_dim]  # (batch, 125)
        tactile_current = observations[
            :, self.tactile_start : self.tactile_current_end
        ]  # (batch, 80)

        tactile_features = self.tactile_encoder(tactile_current)  # (batch, 32)
        return torch.cat([proprio, tactile_features], dim=1)  # (batch, 157)
