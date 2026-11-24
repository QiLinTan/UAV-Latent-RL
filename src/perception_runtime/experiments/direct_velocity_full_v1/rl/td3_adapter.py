"""Feedforward TD3 with one action contract throughout all learning paths."""
import copy
import numpy as np
import torch
from torch import nn
from algos.td3.networks import Actor,Critic


def project(action):
    a=action.clamp(-1,1)
    xy=a[...,:2]/torch.linalg.vector_norm(a[...,:2],dim=-1,keepdim=True).clamp_min(1)
    return torch.cat((xy,a[...,2:3]),dim=-1)


def finite(module):
    return all(torch.isfinite(p).all() for p in module.parameters())


def gradient_norm(module):
    for p in module.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError('nonfinite gradient')
    return float(torch.sqrt(sum((p.grad.detach()**2).sum() for p in module.parameters() if p.grad is not None)))


class TD3:
    def __init__(self,config):
        self.config=config;self.actor=Actor(114,3,1.);self.critic=Critic(114,3)
        self.actor_target=copy.deepcopy(self.actor);self.critic_target=copy.deepcopy(self.critic)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=config['actor_lr'])
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=config['critic_lr'])
        self.updates=0;self.actor_updates=0

    @torch.no_grad()
    def action(self,state):
        return project(self.actor(torch.as_tensor(state,dtype=torch.float32).reshape(-1,114))).numpy().reshape(-1)

    def update(self,replay):
        c=self.config;s,a,n,r,mask=replay.sample(c['batch_size']);self.updates+=1
        with torch.no_grad():
            noise=(torch.randn_like(a)*c['policy_noise']).clamp(-c['noise_clip'],c['noise_clip'])
            na=project(self.actor_target(n)+noise)
            tq1,tq2=self.critic_target(n,na);target=r+mask*c['gamma']*torch.minimum(tq1,tq2)
        q1,q2=self.critic(s,a);loss=nn.functional.mse_loss(q1,target)+nn.functional.mse_loss(q2,target)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite critic loss')
        self.critic_optimizer.zero_grad(set_to_none=True);loss.backward();cg=gradient_norm(self.critic)
        nn.utils.clip_grad_norm_(self.critic.parameters(),c['grad_clip_norm']);self.critic_optimizer.step()
        diag=dict(critic_loss=float(loss.detach()),critic_gradient_norm=cg,actor_loss=None,actor_gradient_norm=None,
            q1_mean=float(q1.detach().mean()),q2_mean=float(q2.detach().mean()),
            q_abs_max=float(torch.maximum(q1.detach().abs().max(),q2.detach().abs().max())),
            td_target_mean=float(target.mean()),td_target_abs_max=float(target.abs().max()),
            td_error_abs_mean=float(((q1.detach()-target).abs().mean()+(q2.detach()-target).abs().mean())/2),actor_updated=False)
        if self.updates%c['policy_delay']==0:
            for p in self.critic.parameters():p.requires_grad_(False)
            aloss=-self.critic.Q1(s,project(self.actor(s))).mean()
            self.actor_optimizer.zero_grad(set_to_none=True);aloss.backward();ag=gradient_norm(self.actor)
            nn.utils.clip_grad_norm_(self.actor.parameters(),c['grad_clip_norm']);self.actor_optimizer.step()
            for p in self.critic.parameters():p.requires_grad_(True)
            self.actor_updates+=1
            with torch.no_grad():
                for m,t in ((self.actor,self.actor_target),(self.critic,self.critic_target)):
                    for p,tp in zip(m.parameters(),t.parameters()):tp.mul_(1-c['tau']).add_(p,alpha=c['tau'])
            diag.update(actor_loss=float(aloss.detach()),actor_gradient_norm=ag,actor_updated=True)
        for module in (self.actor,self.critic,self.actor_target,self.critic_target):
            if not finite(module):raise FloatingPointError('nonfinite network parameter')
        return diag

    def state(self):
        return {**{name:getattr(self,name).state_dict() for name in
            ('actor','critic','actor_target','critic_target','actor_optimizer','critic_optimizer')},
            'updates':self.updates,'actor_updates':self.actor_updates}

    def restore(self,state):
        for name in ('actor','critic','actor_target','critic_target','actor_optimizer','critic_optimizer'):
            getattr(self,name).load_state_dict(state[name])
        self.updates=state['updates'];self.actor_updates=state['actor_updates']

    @torch.no_grad()
    def diagnostics(self,states):
        s=torch.as_tensor(states,dtype=torch.float32).reshape(-1,114)
        raw=self.actor(s);a=project(raw)
        q1,q2=self.critic(s,a);zero=self.critic.Q1(s,torch.zeros_like(a))
        return dict(actor_mean=a.mean(0).tolist(),actor_std=a.std(0,unbiased=False).tolist(),
            raw_axis_boundary_rate=(raw.abs()>=.99).float().mean(0).tolist(),
            projected_axis_boundary_rate=(a.abs()>=.99).float().mean(0).tolist(),
            planar_boundary_rate=float((torch.linalg.vector_norm(a[:,:2],dim=1)>=.99).float().mean()),
            policy_q_mean=float(q1.mean()),q2_mean=float(q2.mean()),zero_action_q_mean=float(zero.mean()),
            parameter_abs_max=max(float(p.abs().max()) for m in (self.actor,self.critic) for p in m.parameters()))
