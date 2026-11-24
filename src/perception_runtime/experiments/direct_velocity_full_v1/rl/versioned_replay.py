"""Fresh, schema-checked ring replay with balanced successful density strata."""
import copy
import numpy as np
import torch


class Ring:
    def __init__(self,capacity):
        self.capacity=int(capacity);self.size=0;self.ptr=0
        self.arrays={k:np.zeros((capacity,d),dtype=np.float32) for k,d in
            [('state',114),('action',3),('next_state',114),('reward',1),('not_done',1),('world_command',3),('lower_steps',1)]}

    def push(self,s,a,r,n,done,world,lower):
        vals=dict(state=np.asarray(s).reshape(114),action=np.asarray(a).reshape(3),next_state=np.asarray(n).reshape(114),
                  reward=[r],not_done=[1.-float(done)],world_command=np.asarray(world).reshape(3),lower_steps=[lower])
        if not all(np.isfinite(v).all() for v in vals.values()):raise ValueError('nonfinite replay')
        if done not in (True,False,0,1) or not 1<=lower<=8:raise ValueError('replay timing/mask')
        if np.linalg.norm(vals['action'][:2])>1.00001 or abs(vals['action'][2])>1.00001:raise ValueError('noncanonical replay action')
        for k,v in vals.items():self.arrays[k][self.ptr]=v
        self.ptr=(self.ptr+1)%self.capacity;self.size=min(self.size+1,self.capacity)

    def sample(self,count):
        if self.size<=0:raise ValueError('empty replay')
        i=np.random.randint(self.size,size=count)
        return tuple(torch.from_numpy(self.arrays[k][i]) for k in ('state','action','next_state','reward','not_done'))

    def state(self):
        return dict(capacity=self.capacity,size=self.size,ptr=self.ptr,arrays={k:a[:self.size].copy() for k,a in self.arrays.items()})

    def restore(self,state):
        if state['capacity']!=self.capacity or not 0<=state['size']<=self.capacity or not 0<=state['ptr']<self.capacity:raise ValueError('ring metadata')
        for k,a in state['arrays'].items():
            if a.shape!=self.arrays[k][:state['size']].shape or not np.isfinite(a).all():raise ValueError('replay shape/finiteness')
            self.arrays[k][:state['size']]=a
        self.size=state['size'];self.ptr=state['ptr']


class VersionedReplay:
    def __init__(self,contract,capacity=60000,success_capacity=20000,fraction=.2):
        self.contract=copy.deepcopy(contract);self.main=Ring(capacity)
        self.success={d:Ring(success_capacity//4) for d in (10,20,30,40)}
        self.fraction=fraction;self.pending=[];self.last_sample_counts={}

    def push(self,s,a,r,n,done,world,lower):
        vals=(np.asarray(s).copy(),np.asarray(a).copy(),r,np.asarray(n).copy(),bool(done),np.asarray(world).copy(),lower)
        self.main.push(*vals);self.pending.append(vals)

    def finish(self,success,density):
        if density not in self.success:raise ValueError('success density mismatch')
        if success:
            for vals in self.pending:self.success[density].push(*vals)
        self.pending=[]

    def sample(self,batch=256):
        available=[d for d,b in self.success.items() if b.size]
        n=int(batch*self.fraction) if available else 0
        batches=[self.main.sample(batch-n)];self.last_sample_counts={'main':batch-n}
        for i,d in enumerate(available):
            count=n//len(available)+(i<n%len(available))
            self.last_sample_counts[d]=count
            if count:batches.append(self.success[d].sample(count))
        perm=torch.randperm(batch)
        return tuple(torch.cat(parts)[perm] for parts in zip(*batches))

    def state(self):
        return dict(contract=self.contract,main=self.main.state(),success={d:b.state() for d,b in self.success.items()},pending=self.pending)

    def restore(self,payload):
        if payload.get('contract')!=self.contract:raise ValueError('Replay schema/config/source mismatch; old data rejected')
        self.main.restore(payload['main'])
        for d,b in self.success.items():b.restore(payload['success'][d])
        self.pending=copy.deepcopy(payload['pending'])

    def diagnostics(self):
        r=self.main
        return dict(size=r.size,done_fraction=float(1-r.arrays['not_done'][:r.size].mean()) if r.size else 0,
            success_sizes={str(d):b.size for d,b in self.success.items()},sample_counts=self.last_sample_counts,
            lower_steps_min=float(r.arrays['lower_steps'][:r.size].min()) if r.size else 0)
