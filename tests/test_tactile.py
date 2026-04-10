import gymnasium
import numpy as np
import torch

from dexterous_hand.tactile.encoder import TactileEncoder
from dexterous_hand.tactile.feature_extractor import TactileFeatureExtractor


class TestTactileEncoder:
    def test_output_shape(self):
        enc = TactileEncoder()
        x = torch.randn(4, 80)
        out = enc(x)
        assert out.shape == (4, 32)

    def test_single_sample(self):
        enc = TactileEncoder()
        x = torch.randn(1, 80)
        out = enc(x)
        assert out.shape == (1, 32)

    def test_output_finite(self):
        enc = TactileEncoder()
        x = torch.randn(8, 80)
        out = enc(x)

        assert torch.all(torch.isfinite(out))

    def test_zero_input(self):
        enc = TactileEncoder()
        x = torch.zeros(2, 80)
        out = enc(x)
        assert out.shape == (2, 32)
        assert torch.all(torch.isfinite(out))

    def test_gradient_flows(self):
        enc = TactileEncoder()
        x = torch.randn(4, 80, requires_grad=True)
        out = enc(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == (4, 80)


class TestTactileFeatureExtractor:
    def make_extractor(self):
        obs_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(371,), dtype=np.float32)
        return TactileFeatureExtractor(obs_space, proprio_dim=131)

    def test_output_shape(self):
        ext = self.make_extractor()
        x = torch.randn(4, 371)
        out = ext(x)

        assert out.shape == (4, 163)  # 131 proprio + 32 tactile

    def test_features_dim(self):
        ext = self.make_extractor()
        assert ext.features_dim == 163

    def test_output_finite(self):
        ext = self.make_extractor()
        x = torch.randn(8, 371)
        out = ext(x)

        assert torch.all(torch.isfinite(out))
