import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from algos.td3.latent_regularizers import posture_labels_np


class LatentEmbeddingCallback:
    """Log latent vectors with posture metadata for TensorBoard Embedding Projector."""

    def __init__(
        self,
        log_dir,
        interval=10000,
        num_samples=1024,
        start_after=10000,
        seed=0,
        tag="latent/posture",
    ):
        self.writer = SummaryWriter(log_dir)
        self.interval = int(interval)
        self.num_samples = int(num_samples)
        self.start_after = int(start_after)
        self.tag = str(tag)
        self.rng = np.random.default_rng(seed)
        self._last_logged_step = None

    def on_step(self, trainer):
        if self.interval <= 0:
            return
        if trainer.total_steps < self.start_after or trainer.total_steps % self.interval != 0:
            return
        if self._last_logged_step == trainer.total_steps:
            return

        agent = trainer.agent
        if not hasattr(agent, "encoder"):
            return
        buffer = trainer.buffer
        if buffer.size <= 0:
            return

        sample_size = min(self.num_samples, buffer.size)
        if sample_size <= 0:
            return
        indices = self.rng.choice(buffer.size, size=sample_size, replace=False)
        states_np = buffer.state[indices].astype(np.float32, copy=False)

        device = getattr(agent, "device", torch.device("cpu"))
        states = torch.as_tensor(states_np, dtype=torch.float32, device=device)

        was_training = agent.encoder.training
        agent.encoder.eval()
        with torch.no_grad():
            latent = agent.encoder(states).detach().cpu()
        if was_training:
            agent.encoder.train()

        labels = posture_labels_np(states_np)
        metadata = [[label] for label in labels]
        metadata_header = ["posture"]

        self.writer.add_embedding(
            latent,
            metadata=metadata,
            metadata_header=metadata_header,
            global_step=trainer.total_steps,
            tag=self.tag,
        )

        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        for label, count in counts.items():
            self.writer.add_scalar(f"{self.tag}_count/{label}", count, trainer.total_steps)
        self.writer.add_scalar(f"{self.tag}_samples", sample_size, trainer.total_steps)
        self.writer.flush()
        self._last_logged_step = trainer.total_steps

    def on_episode_end(self, trainer):
        pass

    def on_train_end(self, trainer):
        self.writer.close()
