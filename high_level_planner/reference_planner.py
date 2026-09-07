"""Small MLP that predicts a five-point position/velocity reference sequence."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReferenceSequencePlanner(nn.Module):
    """Map 27D task context to five future 3D positions and velocities.

    The network predicts 30 learned values.  Relative timestamps are fixed by
    the interface and are deliberately not regressed.  Targets are standardized
    using train-split statistics stored as model buffers so inference and
    checkpoint loading cannot silently disagree about units.
    """

    def __init__(self, input_dim: int = 27, num_points: int = 5):
        super().__init__()
        if int(input_dim) != 27:
            raise ValueError("The first-stage planner contract requires a 27D input.")
        if int(num_points) != 5:
            raise ValueError("The first-stage planner contract requires five future points.")
        self.input_dim = int(input_dim)
        self.num_points = int(num_points)
        self.output_dim = self.num_points * 6
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim),
        )
        self.register_buffer("target_mean", torch.zeros(self.num_points, 6))
        self.register_buffer("target_std", torch.ones(self.num_points, 6))

    def forward(self, planner_input: torch.Tensor) -> torch.Tensor:
        return self.net(planner_input)

    def set_target_normalization(self, mean, std) -> None:
        mean_tensor = torch.as_tensor(mean, dtype=self.target_mean.dtype).reshape(
            self.num_points, 6
        )
        std_tensor = torch.as_tensor(std, dtype=self.target_std.dtype).reshape(
            self.num_points, 6
        )
        if not torch.isfinite(mean_tensor).all() or not torch.isfinite(std_tensor).all():
            raise ValueError("Planner target normalization must be finite.")
        if torch.any(std_tensor <= 0.0):
            raise ValueError("Planner target standard deviations must be positive.")
        self.target_mean.copy_(mean_tensor)
        self.target_std.copy_(std_tensor)

    def denormalize(self, normalized_output: torch.Tensor) -> torch.Tensor:
        shaped = normalized_output.reshape(-1, self.num_points, 6)
        return shaped * self.target_std + self.target_mean

    def predict_sequence(self, planner_input) -> np.ndarray:
        """Return physical-unit ``[5, 6]`` offsets/velocities for one sample."""

        device = next(self.parameters()).device
        value = torch.as_tensor(
            planner_input, dtype=torch.float32, device=device
        ).reshape(1, self.input_dim)
        with torch.no_grad():
            prediction = self.denormalize(self(value))[0]
        return prediction.cpu().numpy().astype(np.float32)


def reference_imitation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    velocity_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted normalized position/velocity imitation loss."""

    if float(velocity_weight) < 0.0:
        raise ValueError("velocity_weight cannot be negative.")
    predicted = prediction.reshape(-1, 5, 6)
    expected = target.reshape(-1, 5, 6)
    position_loss = F.mse_loss(predicted[..., :3], expected[..., :3])
    velocity_loss = F.mse_loss(predicted[..., 3:], expected[..., 3:])
    total = position_loss + float(velocity_weight) * velocity_loss
    return total, {
        "position_loss": position_loss,
        "velocity_loss": velocity_loss,
    }
