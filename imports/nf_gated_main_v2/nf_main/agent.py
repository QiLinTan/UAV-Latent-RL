"""From-scratch NF networks using the existing feedforward RangeTD3 update."""
import copy
import torch
from experiments.rv1_range_cnn.td3 import RangeTD3
from experiments.rv1_range_cnn.codec import RangeCodec
from experiments.rv1_range_cnn.provenance import SCHEMA
from experiments.direct_velocity_full_v1.rl.td3_adapter import project
from .models import Actor,Critic

class NFRangeTD3(RangeTD3):
    def __init__(self,config,contract,device='cpu'):
        if contract.get('schema')!=SCHEMA or contract.get('state_dim')!=18:
            raise ValueError('incompatible base observation contract')
        sensor=contract['sensor']
        self.codec=RangeCodec((sensor['height'],sensor['width']),sensor['max_range_m'])
        self.config=config;self.contract=contract;self.device=torch.device(device)
        self.actor=Actor().to(self.device);self.critic=Critic().to(self.device)
        self.actor_target=copy.deepcopy(self.actor);self.critic_target=copy.deepcopy(self.critic)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=config['actor_lr'])
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=config['critic_lr'])
        self.updates=0;self.actor_updates=0

    @torch.no_grad()
    def diagnostics(self,replay,batch_size=256):
        observation,_,_,_,_=self._batch(replay,batch_size)
        raw=self.actor(observation);actions=project(raw);q1,q2=self.critic(observation,actions)
        z,detail=self.actor.encoder(observation,return_details=True)
        return {'actor_mean':actions.mean(0).tolist(),'actor_std':actions.std(0,unbiased=False).tolist(),
                'q1_mean':float(q1.mean()),'q2_mean':float(q2.mean()),'latent_std':float(z.std()),
                'near_gate_mean':float(detail['gate'][:,0].mean()),'far_gate_mean':float(detail['gate'][:,1].mean())}
