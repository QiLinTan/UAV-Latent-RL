"""Dimension-aware TD3 and stage-tracked replay; uses the frozen TD3 update."""
import copy
import numpy as np
import torch
from torch import nn

from experiments.direct_velocity_full_v1.rl.td3_adapter import TD3, project


class Critic(nn.Module):
    def __init__(self, dim):
        super().__init__()
        def head():
            return nn.Sequential(nn.Linear(dim+3,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,1))
        self.q1 = head()
        self.q2 = head()

    def forward(self, state, action):
        x = torch.cat((state,action),dim=1)
        return self.q1(x),self.q2(x)

    def Q1(self, state, action):
        return self.q1(torch.cat((state,action),dim=1))


class Agent(TD3):
    def __init__(self, config, dim, device):
        self.config = copy.deepcopy(config)
        self.dim = dim
        self.device = torch.device(device)
        self.actor = nn.Sequential(nn.Linear(dim,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),
                                   nn.Linear(256,3),nn.Tanh()).to(self.device)
        self.critic = Critic(dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=config["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=config["critic_lr"])
        self.updates = self.actor_updates = 0

    @torch.no_grad()
    def action(self, x):
        t=torch.as_tensor(x,dtype=torch.float32,device=self.device).reshape(1,self.dim)
        return project(self.actor(t)).cpu().numpy().reshape(3)

    def update(self, replay):
        device=self.device
        class DeviceReplay:
            def sample(self, n):
                return tuple(x.to(device) for x in replay.sample(n))
        return super().update(DeviceReplay())


class Ring:
    def __init__(self, capacity, dim, seed):
        self.capacity=capacity
        self.dim=dim
        self.size=self.pointer=self.total=0
        self.rng=np.random.default_rng(seed)
        self.states=np.zeros((capacity,dim),np.float32)
        self.next_states=np.zeros_like(self.states)
        self.actions=np.zeros((capacity,3),np.float32)
        self.rewards=np.zeros((capacity,1),np.float32)
        self.terminal=np.zeros((capacity,1),bool)
        self.truncated=np.zeros_like(self.terminal)
        self.stage=np.zeros(capacity,np.uint8)

    def add(self, x, action, nxt, reward, terminal, truncated, stage):
        i=self.pointer
        self.states[i]=x
        self.actions[i]=action
        self.next_states[i]=nxt
        self.rewards[i]=reward
        self.terminal[i]=terminal
        self.truncated[i]=truncated
        self.stage[i]=stage
        self.pointer=(i+1)%self.capacity
        self.size=min(self.size+1,self.capacity)
        self.total+=1

    def sample(self,n):
        if not self.size:
            raise ValueError("empty replay")
        idx=self.rng.integers(self.size,size=n)
        values=(self.states[idx],self.actions[idx],self.next_states[idx],self.rewards[idx],
                (~(self.terminal[idx]|self.truncated[idx])).astype(np.float32))
        return tuple(torch.from_numpy(v) for v in values),np.bincount(self.stage[idx],minlength=4)

    def distribution(self):
        return np.bincount(self.stage[:self.size],minlength=4).tolist()


class BalancedReplay:
    def __init__(self,config,dim,seed):
        cap=config["success_replay_size"]
        if cap<4 or cap%4:
            raise ValueError("success strata")
        self.dim=dim
        self.fraction=config["success_replay_fraction"]
        self.main=Ring(config["buffer_size"],dim,seed)
        self.success=[Ring(cap//4,dim,seed+10*(stage+1)) for stage in range(4)]
        self.pending=[]
        self.exposure=np.zeros(4,dtype=np.int64)
        self.sampled_main=np.zeros(4,dtype=np.int64)
        self.sampled_success=np.zeros(4,dtype=np.int64)

    def add(self,x,action,nxt,reward,terminal,truncated,stage):
        if x.shape!=(self.dim,) or nxt.shape!=(self.dim,) or action.shape!=(3,):
            raise ValueError("transition shape does not match policy")
        if not (np.isfinite(reward) and np.isfinite(action).all() and 0<=stage<4):
            raise ValueError("invalid transition")
        if np.max(abs(action))>1.000001 or np.linalg.norm(action[:2])>1.000001:
            raise ValueError("action projection")
        transition=(x,action,nxt,reward,terminal,truncated,stage)
        self.main.add(*transition)
        self.exposure[stage]+=1
        self.pending.append((x.copy(),action.copy(),nxt.copy(),reward,terminal,truncated,stage))

    def finish(self,success,stage):
        if success:
            for transition in self.pending:
                if transition[-1]!=stage:
                    raise ValueError("episode changed stage")
                self.success[stage].add(*transition)
        self.pending=[]

    def sample(self,batch_size):
        available=[ring for ring in self.success if ring.size]
        count=int(batch_size*self.fraction) if available else 0
        main,dist=self.main.sample(batch_size-count)
        self.sampled_main+=dist
        batches=[main]
        for i,ring in enumerate(available):
            n=count//len(available)+(i<count%len(available))
            if n:
                batch,d=ring.sample(n)
                batches.append(batch)
                self.sampled_success+=d
        shuffle=torch.randperm(batch_size)
        return tuple(torch.cat(v)[shuffle] for v in zip(*batches))

    def stats(self):
        return dict(exposure=self.exposure.tolist(),main_inventory=self.main.distribution(),
                    success_inventory=[r.size for r in self.success],
                    sampled_main=self.sampled_main.tolist(),sampled_success=self.sampled_success.tolist())
