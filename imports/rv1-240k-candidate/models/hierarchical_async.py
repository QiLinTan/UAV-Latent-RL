from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class ReferenceSequenceHead(nn.Module):
    """Decode a task latent into a normalized future relative-position sequence."""

    def __init__(self, latent_dim: int, sequence_length: int = 15, reference_dim: int = 3):
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.reference_dim = int(reference_dim)
        self.net = nn.Sequential(
            nn.Linear(int(latent_dim), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.sequence_length * self.reference_dim),
            nn.Tanh(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        sequence = self.net(latent)
        return sequence.reshape(-1, self.sequence_length, self.reference_dim)


@dataclass(frozen=True)
class ReferenceCacheSnapshot:
    sequence: np.ndarray
    latent: np.ndarray
    created_step: int
    point_interval: int
    valid_steps: int
    version: int


class AsyncReferenceCache:
    """Runtime cache used by the high-rate controller between high-level updates."""

    def __init__(self):
        self._snapshot: ReferenceCacheSnapshot | None = None
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    @property
    def has_value(self) -> bool:
        return self._snapshot is not None

    @property
    def created_step(self) -> int | None:
        return None if self._snapshot is None else self._snapshot.created_step

    def clear(self):
        self._snapshot = None

    def update(
        self,
        sequence,
        latent,
        *,
        created_step: int,
        point_interval: int,
        valid_steps: int,
    ) -> ReferenceCacheSnapshot:
        sequence = np.asarray(sequence, dtype=np.float32)
        latent = np.asarray(latent, dtype=np.float32).reshape(-1)
        if sequence.ndim != 2 or sequence.shape[0] < 1:
            raise ValueError(f"Expected a non-empty [sequence, reference] array, got {sequence.shape}.")
        if point_interval <= 0:
            raise ValueError("point_interval must be positive.")
        if valid_steps <= 0:
            raise ValueError("valid_steps must be positive.")

        self._version += 1
        self._snapshot = ReferenceCacheSnapshot(
            sequence=sequence.copy(),
            latent=latent.copy(),
            created_step=int(created_step),
            point_interval=int(point_interval),
            valid_steps=int(valid_steps),
            version=self._version,
        )
        return self._snapshot

    def sample(self, step: int, lookahead_points: int = 1) -> dict:
        if self._snapshot is None:
            raise RuntimeError("The asynchronous reference cache is empty.")

        snapshot = self._snapshot
        age_steps = max(0, int(step) - snapshot.created_step)
        valid = age_steps <= snapshot.valid_steps
        progress = age_steps / float(snapshot.point_interval)
        lower_idx = min(int(np.floor(progress)), snapshot.sequence.shape[0] - 1)
        upper_idx = min(lower_idx + 1, snapshot.sequence.shape[0] - 1)
        alpha = float(np.clip(progress - np.floor(progress), 0.0, 1.0))
        current = (1.0 - alpha) * snapshot.sequence[lower_idx] + alpha * snapshot.sequence[upper_idx]
        lookahead_idx = min(upper_idx + max(0, int(lookahead_points) - 1), snapshot.sequence.shape[0] - 1)

        return {
            "current": current.astype(np.float32, copy=False),
            "lookahead": snapshot.sequence[lookahead_idx].astype(np.float32, copy=False),
            "latent": snapshot.latent,
            "age_steps": age_steps,
            "age_ratio": float(np.clip(age_steps / float(snapshot.valid_steps), 0.0, 1.0)),
            "valid": bool(valid),
            "version": snapshot.version,
            "lower_index": lower_idx,
            "upper_index": upper_idx,
        }
