"""Independent MSGP Actor and Critic encoders with the frozen RV1 TD3 update."""
import copy

import torch
from torch import nn

from experiments.direct_velocity_full_v1.rl.td3_adapter import TD3, project
from perception import GRID, observation_width, unpack


class MSGPConvEncoder(nn.Module):
    def __init__(self, use_far_branch=True, fusion_mode="concat"):
        super().__init__()
        if fusion_mode != "concat":
            raise ValueError(f"unsupported fusion mode: {fusion_mode}")
        self.use_far_branch = use_far_branch
        self.fusion_mode = fusion_mode
        axis = torch.linspace(-1, 1, GRID)
        self.register_buffer("coord_grid", torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"))[None])
        self.near_net = nn.Sequential(
            nn.Conv3d(5, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, 3, stride=2), nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3), nn.ReLU(inplace=True),
            nn.Flatten(), nn.Linear(32, 32), nn.LayerNorm(32),
        )
        if use_far_branch:
            self.far_conv = nn.Sequential(
                nn.Conv1d(5, 16, 3, padding=1, padding_mode="circular"), nn.ReLU(inplace=True),
                nn.Conv1d(16, 32, 3, padding=1, padding_mode="circular"), nn.ReLU(inplace=True),
            )
            self.far_head = nn.Sequential(nn.Linear(64, 32), nn.LayerNorm(32))
        # Keep the initial scale through the ReLU stacks comparable at different depths.
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Conv3d)):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)
        self.output_dim = 82 if use_far_branch else 50
        self.probe = False
        self.feature_norms = {}

    def forward(self, near, far, state):
        coords = self.coord_grid.expand(near.shape[0], -1, -1, -1, -1)
        z_near = self.near_net(torch.cat((near, coords), dim=1))
        z_far = None
        if self.use_far_branch:
            encoded = self.far_conv(far)
            z_far = self.far_head(torch.cat((encoded.mean(dim=-1), encoded.amax(dim=-1)), dim=1))
        if self.probe:
            self.feature_norms = {
                f"{key}_feature_l2": float(value.detach().norm(dim=1).mean())
                for key, value in (("near", z_near), ("far", z_far)) if value is not None
            }
        outputs = (z_near, z_far, state) if self.use_far_branch else (z_near, state)
        return torch.cat(outputs, dim=1)

    def gradient_norms(self):
        branches = dict(near=self.near_net)
        if self.use_far_branch:
            branches["far"] = self.far_conv
        return {
            f"{key}_gradient_l2": float(torch.stack([p.grad.detach().norm() for p in branch.parameters()
                                                      if p.grad is not None]).norm())
            for key, branch in branches.items()
        }


class Actor(nn.Module):
    def __init__(self, use_far_branch):
        super().__init__()
        self.encoder = MSGPConvEncoder(use_far_branch)
        self.policy = nn.Sequential(nn.Linear(self.encoder.output_dim, 256), nn.ReLU(),
                                    nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 3), nn.Tanh())

    def forward(self, observation):
        return self.policy(self.encoder(*observation))


class Critic(nn.Module):
    def __init__(self, use_far_branch):
        super().__init__()
        self.encoder = MSGPConvEncoder(use_far_branch)

        def head():
            return nn.Sequential(nn.Linear(self.encoder.output_dim + 3, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

        self.q1, self.q2 = head(), head()

    def _features(self, observation, action):
        return torch.cat((self.encoder(*observation), action), dim=1)

    def forward(self, observation, action):
        x = self._features(observation, action)
        return self.q1(x), self.q2(x)

    def Q1(self, observation, action):
        return self.q1(self._features(observation, action))


class DeviceReplay:
    def __init__(self, replay, device, use_far_branch):
        self.replay = replay
        self.device = device
        self.use_far_branch = use_far_branch

    def sample(self, size):
        state, action, next_state, reward, mask = (v.to(self.device) for v in self.replay.sample(size))
        return (unpack(state, self.use_far_branch), action, unpack(next_state, self.use_far_branch),
                reward, mask)


class MSGPAgent(TD3):
    def __init__(self, config, use_far_branch, device):
        self.config = config
        self.use_far_branch = use_far_branch
        self.device = torch.device(device)
        self.dim = observation_width(use_far_branch)
        self.actor = Actor(use_far_branch).to(self.device)
        self.critic = Critic(use_far_branch).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config["critic_lr"])
        self.updates = self.actor_updates = 0
        self.probes = {}
        for name in ("actor", "critic"):
            getattr(self, name + "_optimizer").register_step_pre_hook(self._probe_hook(name))

    def _probe_hook(self, name):
        encoder = getattr(self, name).encoder

        def hook(optimizer, args, kwargs):
            if encoder.probe:
                self.probes[name] = dict(step=self.updates + self.config["update_after"] - 1,
                                         **encoder.feature_norms, **encoder.gradient_norms())

        return hook

    @torch.no_grad()
    def action(self, x):
        batch = torch.as_tensor(x, dtype=torch.float32, device=self.device).reshape(1, self.dim)
        return project(self.actor(unpack(batch, self.use_far_branch))).cpu().numpy().reshape(3)

    def update(self, replay):
        step = self.updates + self.config["update_after"]
        probe = step % 1000 in (0, 999)
        self.actor.encoder.probe = self.critic.encoder.probe = probe
        try:
            result = super().update(DeviceReplay(replay, self.device, self.use_far_branch))
        finally:
            self.actor.encoder.probe = self.critic.encoder.probe = False
        if step % 1000 == 0:
            result["branches"] = dict(self.probes)
        return result
