"""T5 observation, fusion and Replay adapters for the frozen T3 trainer."""

from types import MethodType

import numpy as np
import torch

from agent import MSGPAgent
from experiments.direct_velocity_full_v1.contracts import yaw_from_quaternion, yaw_rotation
from experiments.direct_velocity_full_v1.rl.td3_adapter import TD3, project
from experiments.ms_geometric_directional.directional_agent import BalancedReplay, Ring
from modulation_v22 import RESULT_FIELDS, state_scores
from perception import ObservationBuilder, observation_width, unpack

MODE = "phys_dynamic_v22"
BASE_WIDTH = observation_width(True)
MOD_WIDTH = len(RESULT_FIELDS)
WIDTH = BASE_WIDTH + MOD_WIDTH


def attach_frame_observer(env):
    observation = env._observation

    def observe(state):
        raw = observation(state)
        if env._defer_scan:
            return raw
        rotation = yaw_rotation(yaw_from_quaternion(env.base_env.quat[0]))
        values = state_scores(env.lidar.cache, rotation, raw["state"])
        raw["modulation"] = np.asarray([values[key] for key in RESULT_FIELDS], dtype=np.float32)
        raw["modulation_d_min"] = values["d_min"]
        return raw

    env._observation = observe
    return env


class DynamicBuilder(ObservationBuilder):
    def __call__(self, obs):
        base = super().__call__(obs)
        return torch.cat((base, self._tensor(obs["modulation"])))


def unpack_dynamic(batch):
    return (*unpack(batch[..., :BASE_WIDTH], True), batch[..., BASE_WIDTH:])


def dynamic_forward(self, near, far, state, modulation):
    coords = self.coord_grid.expand(near.shape[0], -1, -1, -1, -1)
    z_near = self.near_net(torch.cat((near, coords), dim=1))
    encoded = self.far_conv(far)
    z_far = self.far_head(torch.cat((encoded.mean(dim=-1), encoded.amax(dim=-1)), dim=1))
    alpha, scale = modulation[:, 4:5], modulation[:, 5:6]
    near_out = z_near * (scale * (2 * alpha))
    far_out = z_far * (scale * (2 * (1 - alpha)))
    if self.probe:
        self.feature_norms = {
            f"{name}_feature_l2": float(value.detach().norm(dim=1).mean())
            for name, value in (("near", z_near), ("far", z_far))
        }
        self.feature_norms.update({f"{name}_modulated_l2": float(value.detach().norm(dim=1).mean())
                                   for name, value in (("near", near_out), ("far", far_out))})
    return torch.cat((near_out, far_out, state), dim=1)


class DynamicDeviceReplay:
    def __init__(self, replay, device):
        self.replay, self.device = replay, device

    def sample(self, size):
        state, action, next_state, reward, mask = (v.to(self.device) for v in self.replay.sample(size))
        return unpack_dynamic(state), action, unpack_dynamic(next_state), reward, mask


class DynamicAgent(MSGPAgent):
    def __init__(self, config, use_far_branch, device):
        if not use_far_branch:
            raise ValueError("v2.2 requires both branches")
        super().__init__(config, use_far_branch, device)
        self.dim = WIDTH
        for network in (self.actor, self.critic, self.actor_target, self.critic_target):
            encoder = network.encoder
            # Preserve the initialized tensors, state-dict names, and optimizer references.
            encoder.fusion_mode = MODE
            encoder.forward = MethodType(dynamic_forward, encoder)

    @torch.no_grad()
    def action(self, x):
        batch = torch.as_tensor(x, dtype=torch.float32, device=self.device).reshape(1, self.dim)
        return project(self.actor(unpack_dynamic(batch))).cpu().numpy().reshape(3)

    def update(self, replay):
        step = self.updates + self.config["update_after"]
        probe = step % 1000 in (0, 999)
        self.actor.encoder.probe = self.critic.encoder.probe = probe
        try:
            result = TD3.update(self, DynamicDeviceReplay(replay, self.device))
        finally:
            self.actor.encoder.probe = self.critic.encoder.probe = False
        if step % 1000 == 0:
            result["branches"] = dict(self.probes)
        return result


class DynamicRing(Ring):
    """Store one modulation vector per row, plus terminal and live edge values."""

    def __init__(self, capacity, seed):
        super().__init__(capacity, BASE_WIDTH, seed)
        self.modulation = np.zeros((capacity, MOD_WIDTH), dtype=np.float32)
        self.terminal_modulation = {}
        self.latest_next = None

    def add(self, x, action, nxt, reward, terminal, truncated, stage):
        i = self.pointer
        if self.size:
            previous = (i - 1) % self.capacity
            if (previous not in self.terminal_modulation
                    and not np.array_equal(x[BASE_WIDTH:], self.latest_next)):
                raise ValueError("next-frame modulation differs from the subsequent observation")
        if terminal or truncated:
            # The first frame after reset does not share the terminal frame.
            self.terminal_modulation[i] = nxt[BASE_WIDTH:].copy()
        else:
            self.terminal_modulation.pop(i, None)
        self.modulation[i] = x[BASE_WIDTH:]
        self.latest_next = nxt[BASE_WIDTH:].copy()
        super().add(x[:BASE_WIDTH], action, nxt[:BASE_WIDTH], reward, terminal, truncated, stage)

    def sample(self, n):
        if not self.size:
            raise ValueError("empty replay")
        idx = self.rng.integers(self.size, size=n)
        terminal = self.terminal[idx] | self.truncated[idx]
        next_modulation = self.modulation[(idx + 1) % self.capacity]
        next_modulation[idx == (self.pointer - 1) % self.capacity] = self.latest_next
        for j in np.flatnonzero(terminal.ravel()):
            next_modulation[j] = self.terminal_modulation[idx[j]]
        values = (np.column_stack((self.states[idx], self.modulation[idx])), self.actions[idx],
                  np.column_stack((self.next_states[idx], next_modulation)), self.rewards[idx],
                  (~terminal).astype(np.float32))
        return tuple(torch.from_numpy(value) for value in values), np.bincount(self.stage[idx], minlength=4)


class DynamicReplay(BalancedReplay):
    def __init__(self, config, dim, seed):
        cap = config["success_replay_size"]
        if dim != WIDTH or cap < 4 or cap % 4:
            raise ValueError("unexpected T5 Replay dimensions")
        self.dim = dim
        self.fraction = config["success_replay_fraction"]
        self.main = DynamicRing(config["buffer_size"], seed)
        self.success = [DynamicRing(cap // 4, seed + 10 * (stage + 1)) for stage in range(4)]
        self.pending = []
        self.exposure = np.zeros(4, dtype=np.int64)
        self.sampled_main = np.zeros(4, dtype=np.int64)
        self.sampled_success = np.zeros(4, dtype=np.int64)
