import torch
import torch.nn as nn


class TactileEncoder(nn.Module):
    """CNN: (batch, 80) raw taxel readings -> (batch, 32) features.

    Shared conv net per finger's 4x4 grid, then projection to 32 dims.
    """

    def __init__(self) -> None:
        super().__init__()

        # per-finger conv (shared weights)
        self.finger_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (1, 4, 4) -> (16, 4, 4)
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (16, 4, 4) -> (32, 4, 4)
            nn.ELU(),
            nn.Flatten(),  # -> (512,)
        )

        # projection head
        self.fc = nn.Sequential(
            nn.Linear(5 * 32 * 4 * 4, 64),  # 2560 -> 64
            nn.ELU(),
            nn.Linear(64, 32),  # 64 -> 32
        )

    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        """(batch, 80) -> (batch, 32) tactile features."""

        batch = tactile.shape[0]
        x = tactile.view(batch, 5, 1, 4, 4)

        finger_features = []
        for i in range(5):
            finger_features.append(self.finger_conv(x[:, i]))

        combined = torch.cat(finger_features, dim=1)  # (batch, 2560)

        return self.fc(combined)  # type: ignore[no-any-return]
