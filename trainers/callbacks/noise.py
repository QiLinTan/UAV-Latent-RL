import numpy as np

class NoiseScheduler:
    def __init__(self, start=0.5, end=0.1, decay_steps=100000):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps

    def get_noise(self, step):
        return max(
            self.end,
            self.start - step / self.decay_steps * (self.start - self.end)
        )