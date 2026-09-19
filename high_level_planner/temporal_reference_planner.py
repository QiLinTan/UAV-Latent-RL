"""History-stack and GRU variants of the frozen-latent reference planner."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .reference_utils import PLANNER_INPUT_DIM
from .temporal_context import HISTORY_LENGTH, TEMPORAL_FRAME_DIM


HISTORY_STACK_INPUT_DIM = PLANNER_INPUT_DIM + HISTORY_LENGTH * TEMPORAL_FRAME_DIM
TEMPORAL_GRU_INPUT_DIM = PLANNER_INPUT_DIM + 32


class TemporalReferenceSequencePlanner(nn.Module):
    """Predict the unchanged five-point reference from explicit history.

    ``history_stack`` tests whether the information itself is useful.  ``gru``
    tests whether a compact sequence encoder adds value beyond the same eight
    frames.  Both retain the old 256-256 MLP head and 30-D output contract.
    """

    def __init__(self, variant: str, num_points: int = 5):
        super().__init__()
        if variant not in ("history_stack", "gru"):
            raise ValueError("variant must be 'history_stack' or 'gru'.")
        if int(num_points) != 5:
            raise ValueError("The reference contract requires five future points.")
        self.variant = str(variant)
        self.num_points = int(num_points)
        self.output_dim = 30
        if self.variant == "history_stack":
            self.input_dim = HISTORY_STACK_INPUT_DIM
            self.gru = None
        else:
            self.input_dim = TEMPORAL_GRU_INPUT_DIM
            self.gru = nn.GRU(
                input_size=TEMPORAL_FRAME_DIM,
                hidden_size=32,
                num_layers=1,
                batch_first=True,
            )
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim),
        )
        self.register_buffer("target_mean", torch.zeros(self.num_points, 6))
        self.register_buffer("target_std", torch.ones(self.num_points, 6))

    def forward(self, planner_input: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        if planner_input.ndim != 2 or planner_input.shape[-1] != PLANNER_INPUT_DIM:
            raise ValueError("planner_input must have shape [batch, 27].")
        if history.ndim != 3 or tuple(history.shape[1:]) != (HISTORY_LENGTH, TEMPORAL_FRAME_DIM):
            raise ValueError("history must have shape [batch, 8, 16].")
        if self.variant == "history_stack":
            feature = torch.cat((planner_input, history.flatten(start_dim=1)), dim=1)
        else:
            assert self.gru is not None
            _, hidden = self.gru(history)
            feature = torch.cat((planner_input, hidden[-1]), dim=1)
        return self.net(feature)

    def set_target_normalization(self, mean, std) -> None:
        mean_tensor = torch.as_tensor(mean, dtype=self.target_mean.dtype).reshape(5, 6)
        std_tensor = torch.as_tensor(std, dtype=self.target_std.dtype).reshape(5, 6)
        if not torch.isfinite(mean_tensor).all() or not torch.isfinite(std_tensor).all():
            raise ValueError("Target normalization must be finite.")
        if torch.any(std_tensor <= 0.0):
            raise ValueError("Target standard deviations must be positive.")
        self.target_mean.copy_(mean_tensor)
        self.target_std.copy_(std_tensor)

    def denormalize(self, normalized_output: torch.Tensor) -> torch.Tensor:
        return normalized_output.reshape(-1, 5, 6) * self.target_std + self.target_mean

    def predict_sequence(self, planner_input, history) -> np.ndarray:
        device = next(self.parameters()).device
        current = torch.as_tensor(planner_input, dtype=torch.float32, device=device).reshape(1, 27)
        sequence = torch.as_tensor(
            history, dtype=torch.float32, device=device
        ).reshape(1, HISTORY_LENGTH, TEMPORAL_FRAME_DIM)
        with torch.no_grad():
            prediction = self.denormalize(self(current, sequence))[0]
        return prediction.cpu().numpy().astype(np.float32)
