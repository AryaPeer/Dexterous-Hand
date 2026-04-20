
import gymnasium
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch

from dexterous_hand.tactile.encoder import TactileEncoder

class TactileFeatureExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: gymnasium.spaces.Box,
        proprio_dim: int = 131,
    ) -> None:

        features_dim = proprio_dim + 32
        super().__init__(observation_space, features_dim=features_dim)

        self.proprio_dim = proprio_dim
        self.tactile_start = proprio_dim
        self.tactile_current_end = proprio_dim + 80

        self.tactile_encoder = TactileEncoder()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        proprio = observations[:, : self.proprio_dim]

        tactile_current = observations[:, self.tactile_start : self.tactile_current_end]

        tactile_features = self.tactile_encoder(tactile_current)

        return torch.cat([proprio, tactile_features], dim=1)
