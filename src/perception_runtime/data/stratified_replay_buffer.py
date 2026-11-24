from __future__ import annotations

import numpy as np
import torch

from data.replay_buffer import ReplayBuffer


class DifficultyStratifiedReplayBuffer:
    """Successful transitions partitioned by route-obstacle difficulty."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        max_size: int,
        difficulties=(1, 2, 3, 5),
    ):
        levels = tuple(dict.fromkeys(int(level) for level in difficulties))
        if not levels or min(levels) < 0:
            raise ValueError("difficulties must contain non-negative integer levels.")
        if int(max_size) < len(levels):
            raise ValueError("max_size must be at least the number of difficulty levels.")
        self.difficulties = levels
        base_capacity, remainder = divmod(int(max_size), len(levels))
        self.buffers = {
            level: ReplayBuffer(
                state_dim,
                action_dim,
                max_size=base_capacity + (index < remainder),
            )
            for index, level in enumerate(levels)
        }
        self.last_sample_counts = {level: 0 for level in levels}

    @property
    def size(self) -> int:
        return sum(buffer.size for buffer in self.buffers.values())

    @property
    def strata_sizes(self) -> dict[int, int]:
        return {level: buffer.size for level, buffer in self.buffers.items()}

    def push(self, difficulty, state, action, reward, next_state, done):
        difficulty = int(difficulty)
        if difficulty not in self.buffers:
            difficulty = min(self.difficulties, key=lambda level: abs(level - difficulty))
        self.buffers[difficulty].push(state, action, reward, next_state, done)

    def sample(self, batch_size):
        batch_size = int(batch_size)
        available = [
            (level, buffer)
            for level, buffer in self.buffers.items()
            if buffer.size > 0
        ]
        if batch_size <= 0 or not available:
            raise ValueError("Cannot sample an empty stratified replay batch.")

        base_count, remainder = divmod(batch_size, len(available))
        batches = []
        self.last_sample_counts = {level: 0 for level in self.difficulties}
        for index, (level, buffer) in enumerate(available):
            count = base_count + (index < remainder)
            if count <= 0:
                continue
            batches.append(buffer.sample(count))
            self.last_sample_counts[level] = count

        merged = tuple(torch.cat(items, dim=0) for items in zip(*batches))
        permutation = torch.as_tensor(np.random.permutation(batch_size), dtype=torch.long)
        return tuple(item[permutation] for item in merged)
