"""Verify exact frozen-T3 equivalence and v2.2 Replay across resets and wrapping."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from agent import MSGPAgent
from dynamic_v22 import BASE_WIDTH, WIDTH, DynamicAgent, DynamicReplay, unpack_dynamic
from perception import unpack
from train import sha


def verify_fusion(runs):
    root = runs / "msgp_conv7_circ16_dual_seed43_240k"
    path = root / "train/checkpoint_240000.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    cfg = checkpoint["contract"]["td3"]
    base, dynamic = MSGPAgent(cfg, True, "cpu"), DynamicAgent(cfg, True, "cpu")
    for model in (base, dynamic):
        for name in ("actor", "critic"):
            getattr(model, name).load_state_dict(checkpoint["agent"][name])
    panel = torch.from_numpy(np.load(root / "fixed_actor_panel.npy", allow_pickle=False)[:64])
    extra = torch.zeros((len(panel), WIDTH - BASE_WIDTH))
    extra[:, 4] = .5
    extra[:, 5] = 1.
    static = unpack(panel, True)
    modulated = unpack_dynamic(torch.cat((panel, extra), dim=1))
    actions = torch.linspace(-.3, .3, len(panel) * 3).reshape(-1, 3)
    with torch.inference_mode():
        assert torch.equal(base.actor(static), dynamic.actor(modulated))
        assert all(torch.equal(x, y) for x, y in zip(base.critic(static, actions),
                                                    dynamic.critic(modulated, actions)))
    counts = lambda m: sum(p.numel() for p in m.parameters())
    assert counts(base.actor) == counts(dynamic.actor) == 136611
    assert counts(base.critic) == counts(dynamic.critic) == 224930
    return cfg, sha(path)


def verify_replay(cfg):
    replay = DynamicReplay(dict(cfg, buffer_size=5, success_replay_size=12), WIDTH, 43)
    expected = {}

    def observation(i):
        return np.full(WIDTH, i, dtype=np.float32) + np.arange(WIDTH, dtype=np.float32) / 1024

    for episode, stage, length, success in ((0, 0, 4, True), (1, 0, 2, True), (2, 1, 3, False)):
        for t in range(length):
            x = observation(episode * 100 + t)
            nxt = observation(episode * 100 + t + 1)
            end = t == length - 1
            replay.add(x, np.zeros(3, dtype=np.float32), nxt, 0, end and success, end and not success, stage)
            expected[x[0]] = (nxt, not end)
            state, _, following, _, mask = replay.sample(32)
            for a, b, c in zip(state.numpy(), following.numpy(), mask.numpy()):
                assert np.array_equal(b, expected[a[0]][0]) and bool(c[0]) == expected[a[0]][1]
        replay.finish(success, stage)
    resumed = pickle.loads(pickle.dumps(replay))
    torch.manual_seed(99)
    before = replay.sample(64)
    torch.manual_seed(99)
    after = resumed.sample(64)
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg, digest = verify_fusion(args.runs)
    verify_replay(cfg)
    path = Path(__file__).with_name("interface_check.json")
    path.write_text(json.dumps(dict(exact_actor=True, exact_twin_critic=True, replay_next_frame=True,
                                    replay_checkpoint_roundtrip=True, parameter_counts_unchanged=True,
                                    static_checkpoint_sha256=digest), indent=2), encoding="utf-8")
    print("T5 interface passed", flush=True)


if __name__ == "__main__":
    main()
