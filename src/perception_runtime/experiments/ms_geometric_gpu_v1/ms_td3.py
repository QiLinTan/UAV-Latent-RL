"""Experimental multi-scale TD3; inherits RV1's frozen TD3 update equation."""
import copy
import numpy as np
import torch
from torch import nn
from experiments.direct_velocity_full_v1.rl.td3_adapter import TD3, project
from fast_multiscale import FastEncoder
from multiscale import rv1_directions

SCHEMA = 'rv1_msgp_320_v1'

class ObservationEncoder:
    def __init__(self, lidar_config):
        from experiments.rv1_range_cnn.lidar import LidarConfig
        if lidar_config != LidarConfig(): raise ValueError('only frozen default RV1 lidar geometry supported')
        self.encoder=FastEncoder(); self.directions=lidar_config.directions()
    def __call__(self,obs):
        state=np.asarray(obs['state'],np.float32)
        if state.shape!=(18,) or not np.isfinite(state).all(): raise ValueError('State18 required')
        features,_=self.encoder.encode(self.directions,obs['range_m'],obs['hit_mask'],obs['scanned_mask'])
        result=np.concatenate((state,features)).astype(np.float32)
        if result.shape!=(320,) or not np.isfinite(result).all(): raise ValueError('invalid feature vector')
        return result

class Actor(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(320,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,3),nn.Tanh())
    def forward(self,s): return self.net(s)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        def head():return nn.Sequential(nn.Linear(323,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,1))
        self.q1=head(); self.q2=head()
    def forward(self,s,a):
        x=torch.cat((s,a),dim=1);return self.q1(x),self.q2(x)
    def Q1(self,s,a): return self.q1(torch.cat((s,a),dim=1))

class GeometricTD3(TD3):
    def __init__(self,config,device='cpu'):
        self.config=copy.deepcopy(config);self.device=torch.device(device)
        self.actor=Actor().to(self.device);self.critic=Critic().to(self.device)
        self.actor_target=copy.deepcopy(self.actor);self.critic_target=copy.deepcopy(self.critic)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=config['actor_lr']);self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=config['critic_lr'])
        self.updates=0;self.actor_updates=0
    @torch.no_grad()
    def action(self,s):return project(self.actor(torch.as_tensor(s,dtype=torch.float32,device=self.device).reshape(1,320))).cpu().numpy().reshape(3)
    def update(self,replay):
        agent=self
        class DeviceReplay:
            def sample(self,n): return tuple(x.to(agent.device) for x in replay.sample(n))
        return super().update(DeviceReplay())

class Ring:
    def __init__(self,capacity,seed):
        self.capacity=int(capacity);self.size=0;self.pointer=0;self.total=0;self.rng=np.random.default_rng(seed)
        self.s=np.zeros((capacity,320),np.float32);self.n=np.zeros_like(self.s);self.a=np.zeros((capacity,3),np.float32)
        self.r=np.zeros((capacity,1),np.float32);self.t=np.zeros((capacity,1),bool);self.u=np.zeros_like(self.t)
    def add(self,s,a,n,r,t,u):
        s=np.asarray(s,np.float32);n=np.asarray(n,np.float32);a=np.asarray(a,np.float32)
        if s.shape!=(320,) or n.shape!=(320,) or a.shape!=(3,) or not all(np.isfinite(x).all() for x in (s,n,a)) or not np.isfinite(r): raise ValueError('invalid transition')
        if np.max(abs(a))>1.000001 or np.linalg.norm(a[:2])>1.000001:raise ValueError('action projection')
        i=self.pointer;self.s[i]=s;self.n[i]=n;self.a[i]=a;self.r[i]=r;self.t[i]=t;self.u[i]=u
        self.pointer=(i+1)%self.capacity;self.size=min(self.size+1,self.capacity);self.total+=1
    def sample(self,count):
        if self.size==0:raise ValueError('empty ring')
        i=self.rng.integers(self.size,size=count)
        return tuple(torch.from_numpy(x) for x in (self.s[i],self.a[i],self.n[i],self.r[i],(~(self.t[i]|self.u[i])).astype(np.float32)))

class BalancedReplay:
    """Same success fraction, strata and done mask as RV1; stores frozen features."""
    def __init__(self,capacity,success_capacity,fraction,seed=43):
        if success_capacity<4 or success_capacity%4:raise ValueError('success strata')
        self.schema=SCHEMA;self.fraction=fraction;self.main=Ring(capacity,seed);self.success={d:Ring(success_capacity//4,seed+d) for d in (10,20,30,40)};self.pending=[]
    def add(self,s,a,n,r,t,u):
        self.main.add(s,a,n,r,t,u);self.pending.append((s.copy(),a.copy(),n.copy(),r,t,u))
    def finish(self,success,density):
        if density not in self.success:raise ValueError('density')
        if success:
            for tr in self.pending:self.success[density].add(*tr)
        self.pending=[]
    def sample(self,batch_size):
        avail=[d for d,b in self.success.items() if b.size];n=int(batch_size*self.fraction) if avail else 0
        batches=[self.main.sample(batch_size-n)]
        for i,d in enumerate(avail):
            count=n//len(avail)+(i<n%len(avail))
            if count:batches.append(self.success[d].sample(count))
        perm=torch.randperm(batch_size)
        return tuple(torch.cat(x)[perm] for x in zip(*batches))
