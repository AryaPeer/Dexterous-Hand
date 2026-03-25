import torch
import torch.nn as nn


class TactileEncoder(nn.Module):
    """Small CNN that turns raw taxel readings into a compact feature vector.

    Takes (batch, 80) raw readings (5 fingers x 16 taxels each), runs a shared
    conv net on each finger's 4x4 grid, then projects down to 32 dims.
    """

    def __init__(self) -> None:
        """Set up the per-finger conv layers and the final projection head."""
        super().__init__()

        self.finger_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (1, 4, 4) -> (16, 4, 4)
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (16, 4, 4) -> (32, 4, 4)
            nn.ELU(),
            nn.Flatten(),  # -> (512,)
        )

        self.fc = nn.Sequential(
            nn.Linear(5 * 32 * 4 * 4, 64),  # 2560 -> 64
            nn.ELU(),
            nn.Linear(64, 32),  # 64 -> 32
        )

    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        """Process tactile readings through the CNN.

        Args:
            tactile: (batch, 80) raw taxel readings

        Returns:
            (batch, 32) learned tactile features
        """
        batch = tactile.shape[0]
        x = tactile.view(batch, 5, 1, 4, 4)

        finger_features = []
        for i in range(5):
            finger_features.append(self.finger_conv(x[:, i]))  # (batch, 512)

        combined = torch.cat(finger_features, dim=1)  # (batch, 2560)
        return self.fc(combined)  # type: ignore[no-any-return]  # (batch, 32)
