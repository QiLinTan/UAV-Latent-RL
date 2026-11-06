import tempfile
import unittest

import numpy as np
import torch

from algos.td3.networks import Actor
from algos.td3.td3_upper_semantic_latent import TD3UpperSemanticLatent


class UpperSemanticLatentTest(unittest.TestCase):
    def test_zero_initialized_residual_exactly_preserves_base_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = f"{tmp}/base"
            base_actor = Actor(29, 3, 1.0)
            torch.save(base_actor.state_dict(), prefix + "_actor")
            agent = TD3UpperSemanticLatent(
                state_dim=77,
                action_dim=3,
                max_action=1.0,
                base_policy_checkpoint=prefix,
            )
            state = np.random.default_rng(4).normal(size=77).astype(np.float32)
            with torch.no_grad():
                expected = base_actor(torch.as_tensor(state[:29]).reshape(1, -1)).numpy()[0]
            np.testing.assert_allclose(agent.select_action(state), expected, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
