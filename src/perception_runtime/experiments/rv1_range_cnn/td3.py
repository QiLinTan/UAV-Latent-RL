"""Same TD3 update equations as dv0, independent image networks and schemas."""
import copy
import numpy as np
import torch
from experiments.direct_velocity_full_v1.rl.td3_adapter import TD3 as FrozenAlgorithm,project
from .models import Actor,Critic
from .codec import RangeCodec
from .provenance import SCHEMA,validate_contract

class RangeTD3(FrozenAlgorithm):
    # Inherit update/state/restore equations unchanged. Networks accept structured
    # sensor observations, not the old 114D vector. Actor and Critic encoders are
    # independent; twin Q heads share the Critic encoder. Targets are deep copies.
    def __init__(self,config,contract,device='cpu'):
        if contract.get('schema')!=SCHEMA:raise ValueError('rv1 network contract required')
        self.config=config;self.contract=contract;s=contract['sensor'];shape=(s['height'],s['width'])
        self.codec=RangeCodec(shape,s['max_range_m'])
        kwargs=dict(shape=shape,latent_dim=contract['latent_dim'],circular=s['horizontal_fov_deg']==360.)
        self.actor=Actor(**kwargs);self.critic=Critic(**kwargs)
        self.actor_target=copy.deepcopy(self.actor);self.critic_target=copy.deepcopy(self.critic)
        self.device=torch.device(device)
        if self.device.type not in ('cpu','cuda'):raise ValueError('unsupported device')
        for module in (self.actor,self.critic,self.actor_target,self.critic_target):module.to(self.device)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=config['actor_lr'])
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=config['critic_lr'])
        self.updates=0;self.actor_updates=0

    def tensor_observation(self,obs):
        q,m=self.codec.encode(obs['range_m'],obs['hit_mask'],obs['scanned_mask'])
        return dict(state=torch.from_numpy(np.asarray(obs['state'],np.float32)).reshape(1,18).to(self.device),
                    image=torch.from_numpy(self.codec.policy_image(q,m)).unsqueeze(0).to(self.device))

    @torch.no_grad()
    def action(self,obs):return project(self.actor(self.tensor_observation(obs))).cpu().numpy().reshape(3)

    def _batch(self,replay,batch_size):
        def move(value):
            return {k:move(v) for k,v in value.items()} if isinstance(value,dict) else value.to(self.device)
        return tuple(move(x) for x in replay.sample(batch_size))

    def update(self,replay):
        agent=self
        class DeviceReplay:
            def sample(self,batch_size):return agent._batch(replay,batch_size)
        return super().update(DeviceReplay())

    def checkpoint(self):return dict(schema=SCHEMA,contract=self.contract,agent=self.state())
    def restore(self,state):
        # Adam.load_state_dict may retain tensor storage when dtype/device match.
        # Isolate restored instances even when callers pass a live state_dict.
        super().restore(copy.deepcopy(state))
    def restore_checkpoint(self,payload):
        if payload.get('schema')!=SCHEMA:raise ValueError('114D checkpoint rejected')
        validate_contract(payload['contract'],self.contract);self.restore(payload['agent'])

    @torch.no_grad()
    def diagnostics(self,replay,batch_size=256):
        observation,_,_,_,_=self._batch(replay,batch_size);raw=self.actor(observation);actions=project(raw)
        q1,q2=self.critic(observation,actions);latent,spatial=self.actor.encoder(observation['image'],return_spatial=True)
        return dict(actor_mean=actions.mean(0).tolist(),actor_std=actions.std(0,unbiased=False).tolist(),
            planar_boundary_rate=float((torch.linalg.vector_norm(actions[:,:2],dim=1)>=.99).float().mean()),
            raw_axis_boundary_rate=(raw.abs()>=.99).float().mean(0).tolist(),q1_mean=float(q1.mean()),q2_mean=float(q2.mean()),
            latent_mean=float(latent.mean()),latent_std=float(latent.std()),spatial_shape=list(spatial.shape),
            actor_parameter_abs_max=max(float(p.abs().max()) for p in self.actor.parameters()))
