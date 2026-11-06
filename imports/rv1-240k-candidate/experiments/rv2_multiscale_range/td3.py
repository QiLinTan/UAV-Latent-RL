"""Frozen TD3 equations with rv2 independent encoders and version gates."""
import copy
import numpy as np
import torch
from experiments.direct_velocity_full_v1.rl.td3_adapter import TD3 as FrozenTD3, project
from .contracts import SCHEMA, validate_contract
from .features import canonical_image
from .models import Actor, Critic

class RangeTD3(FrozenTD3):
    def __init__(self, config, contract, spec):
        if contract.get('schema') != SCHEMA:
            raise ValueError('rv2 contract required')
        from dataclasses import asdict
        from .contracts import digest
        if digest(contract['perception']) != digest(asdict(spec)):
            raise ValueError('model configuration mismatch')
        self.config = copy.deepcopy(config); self.contract = copy.deepcopy(contract); self.spec = spec
        self.actor = Actor(spec); self.critic = Critic(spec)
        self.actor_target = copy.deepcopy(self.actor); self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['critic_lr'])
        self.updates = self.actor_updates = 0

    def tensor_observation(self, obs):
        return dict(state=torch.from_numpy(np.asarray(obs['state'], np.float32)).reshape(1, 18),
                    image=torch.from_numpy(canonical_image(obs)).unsqueeze(0))

    @torch.no_grad()
    def action(self, obs):
        return project(self.actor(self.tensor_observation(obs))).numpy().reshape(3)

    def checkpoint(self):
        return dict(format='rv2_engineering_checkpoint_v1', schema=SCHEMA, contract=self.contract,
                    training_config=self.config, agent=copy.deepcopy(self.state()), torch_rng=torch.get_rng_state())

    def restore_checkpoint(self, payload, restore_rng=False):
        if payload.get('format') != 'rv2_engineering_checkpoint_v1':
            raise ValueError('old checkpoint rejected')
        validate_contract(payload['contract'], self.contract)
        if payload['training_config'] != self.config:
            raise ValueError('TD3 hyperparameter mismatch')
        self.restore(copy.deepcopy(payload['agent']))
        if restore_rng:
            torch.set_rng_state(payload['torch_rng'])
