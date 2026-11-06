import torch
import torch.nn as nn


class ActionConditionedAuxHead(nn.Module):
    """Small z+a prediction head for latent-only auxiliary objectives."""

    def __init__(self, latent_dim=16, action_dim=4, output_dim=1, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=1))
