import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
	def __init__(self, state_dim, latent_dim=16):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(state_dim, 128),
			nn.ReLU(),
			nn.Linear(128, latent_dim),
		)

	def forward(self, s):
		return self.net(s)
