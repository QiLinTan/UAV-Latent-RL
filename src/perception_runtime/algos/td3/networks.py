import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class ResidualMotorActor(nn.Module):
    """Bounded collective thrust plus bounded zero-mean motor differentials."""

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        collective_fraction=0.60,
        differential_fraction=0.25,
        differential_headroom=1.0,
    ):
        super().__init__()
        if int(action_dim) != 4:
            raise ValueError("ResidualMotorActor currently expects four motor actions.")
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.collective_head = nn.Linear(256, 1)
        self.differential_head = nn.Linear(256, action_dim)
        self.max_action = float(max_action)
        self.collective_limit = self.max_action * float(collective_fraction)
        self.differential_limit = self.max_action * float(differential_fraction)
        self.differential_headroom = max(1.0, float(differential_headroom))

    def forward(self, state):
        hidden = F.relu(self.l1(state))
        hidden = F.relu(self.l2(hidden))
        collective = self.collective_limit * torch.tanh(self.collective_head(hidden))
        differential_raw = self.differential_head(hidden)
        differential_raw = differential_raw - differential_raw.mean(dim=1, keepdim=True)
        differential = (
            self.differential_limit
            * self.differential_headroom
            * torch.tanh(differential_raw)
        ).clamp(-self.differential_limit, self.differential_limit)
        return (collective + differential).clamp(-self.max_action, self.max_action)


class StructuredMotorActor(nn.Module):
    """Collective plus roll/pitch/yaw virtual channels mapped to four motors."""

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        collective_fraction=0.60,
        differential_fraction=0.25,
        virtual_headroom=1.6,
    ):
        super().__init__()
        if int(action_dim) != 4:
            raise ValueError("StructuredMotorActor expects four motor actions.")
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.collective_head = nn.Linear(256, 1)
        self.virtual_head = nn.Linear(256, 3)
        self.max_action = float(max_action)
        self.collective_limit = self.max_action * float(collective_fraction)
        self.differential_limit = self.max_action * float(differential_fraction)
        self.virtual_headroom = max(1.0, float(virtual_headroom))
        # CF2X-compatible orthogonal roll, pitch and yaw allocation patterns.
        self.register_buffer(
            "mixer",
            torch.tensor(
                [
                    [-1.0, -1.0, 1.0, 1.0],
                    [-1.0, 1.0, 1.0, -1.0],
                    [-1.0, 1.0, -1.0, 1.0],
                ],
                dtype=torch.float32,
            ),
        )

    def forward(self, state):
        hidden = F.relu(self.l1(state))
        hidden = F.relu(self.l2(hidden))
        collective = self.collective_limit * torch.tanh(self.collective_head(hidden))
        virtual = self.virtual_headroom * torch.tanh(self.virtual_head(hidden))
        differential_raw = virtual @ self.mixer
        scale = differential_raw.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
        differential = self.differential_limit * differential_raw / scale
        return (collective + differential).clamp(-self.max_action, self.max_action)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
