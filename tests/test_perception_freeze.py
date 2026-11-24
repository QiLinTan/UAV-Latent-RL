"""Regression tests for frozen fusion, Replay and Suite D."""

import os
from pathlib import Path
import unittest

import numpy as np
import torch

from perception_reproduce import DOCS, read, materialize_pool
from fusion_modes import make_agent
from dynamic_v22 import unpack_dynamic, WIDTH, BASE_WIDTH
from check_interface import verify_replay
from perception import unpack
from test_suite_d import SuiteDTests


class FrozenPerceptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        materialize_pool()
        archive = Path(os.environ["PERCEPTION_ARCHIVE"])
        index = read(DOCS / "checkpoint_hashes.json")
        ref, = [r for r in index["checkpoints"]
                if (r["model"], r["seed"], r["step"]) == ("t3", 43, 240000)]
        cls.checkpoint = torch.load(archive / ref["archive_relative_path"], map_location="cpu",
                                    mmap=True, weights_only=False)

    def test_exact_actor_and_twin_critic_at_neutral_gate(self):
        cfg = self.checkpoint["contract"]["td3"]
        static = make_agent(cfg, "static", "cpu")
        dynamic = make_agent(cfg, "phys_dynamic_v22", "cpu")
        for agent in (static, dynamic):
            for name in ("actor", "critic"):
                getattr(agent, name).load_state_dict(self.checkpoint["agent"][name])
        x = torch.from_numpy(np.random.default_rng(43).normal(size=(64, BASE_WIDTH)).astype(np.float32))
        gate = torch.zeros((x.shape[0], WIDTH - BASE_WIDTH), dtype=x.dtype)
        gate[:, 4], gate[:, 5] = .5, 1.
        left, right = unpack(x, True), unpack_dynamic(torch.cat((x, gate), dim=1))
        a = torch.linspace(-.3, .3, len(x) * 3).reshape(-1, 3)
        with torch.inference_mode():
            self.assertTrue(torch.equal(static.actor(left), dynamic.actor(right)))
            for p, q in zip(static.critic(left, a), dynamic.critic(right, a)):
                self.assertTrue(torch.equal(p, q))
        for name, count in (("actor", 136611), ("critic", 224930)):
            for agent in (static, dynamic):
                self.assertEqual(sum(p.numel() for p in getattr(agent, name).parameters()), count)

    def test_replay_next_observation_and_checkpoint(self):
        verify_replay(self.checkpoint["contract"]["td3"])
