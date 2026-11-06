"""Deterministic per-episode 30/70 Stage 0/3 schedule for Full fine-tuning."""

import numpy as np


class FullFTStageMixer:
    """Shuffle exact three Stage 0 and seven Stage 3 episodes per block."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(int(seed) + 90401)
        self._block = np.empty(0, dtype=np.int8)
        self.episode_count = 0
        self.stage_counts = {0: 0, 3: 0}

    def next_stage(self):
        if self.episode_count % 10 == 0:
            self._block = np.array([0] * 3 + [3] * 7, dtype=np.int8)
            self.rng.shuffle(self._block)
        stage = int(self._block[self.episode_count % 10])
        self.episode_count += 1
        self.stage_counts[stage] += 1
        return stage
