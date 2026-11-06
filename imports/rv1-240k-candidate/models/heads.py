import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconHead(nn.Module):
	def __init__(self, latent_dim=16, state_dim=12):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(latent_dim, 64),
			nn.ReLU(),
			nn.Linear(64, state_dim),
		)

	def forward(self, z):
		return self.net(z)


class DynHead(nn.Module):
	def __init__(self, latent_dim=16, action_dim=4, state_dim=12):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(latent_dim + action_dim, 64),
			nn.ReLU(),
			nn.Linear(64, state_dim),
		)

	def forward(self, z, a):
		return self.net(torch.cat([z, a], dim=1))


class ProgressHead(nn.Module):
	def __init__(self, latent_dim=16, action_dim=4):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(latent_dim + action_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 1),
		)

	def forward(self, z, a):
		return self.net(torch.cat([z, a], dim=1))
