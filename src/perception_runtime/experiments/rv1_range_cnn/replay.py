"""Versioned raw sensor Replay: bounded uint16 + bit masks, no stale latent."""
from pathlib import Path
import numpy as np
import torch
from .codec import RangeCodec
from .provenance import SCHEMA,validate_contract

class Replay:
    def __init__(self,capacity,contract,seed=43):
        if contract.get('schema')!=SCHEMA or contract.get('state_dim')!=18:raise ValueError('rv1 Replay schema required; reject 114D')
        self.capacity=int(capacity);self.contract=contract
        if self.capacity<1:raise ValueError('capacity')
        s=contract['sensor'];self.codec=RangeCodec((s['height'],s['width']),s['max_range_m']);h,w=self.codec.shape
        self.states=np.zeros((capacity,18),np.float32);self.next_states=np.zeros_like(self.states)
        self.ranges=np.zeros((capacity,h,w),np.uint16);self.next_ranges=np.zeros_like(self.ranges)
        self.masks=np.zeros((capacity,2,(h*w+7)//8),np.uint8);self.next_masks=np.zeros_like(self.masks)
        self.actions=np.zeros((capacity,3),np.float32);self.rewards=np.zeros((capacity,1),np.float32)
        self.terminated=np.zeros((capacity,1),bool);self.truncated=np.zeros_like(self.terminated)
        self.episode_ids=np.zeros(capacity,np.int64);self.sensor_times=np.zeros((capacity,2),np.float64)
        self.size=0;self.pointer=0;self.total=0;self.rng=np.random.default_rng(seed)

    def encode_observation(self,obs):
        if set(obs)!=set(('state','range_m','hit_mask','scanned_mask')):raise ValueError('policy observation keys/schema')
        s=np.asarray(obs['state'],np.float32)
        if s.shape!=(18,) or not np.isfinite(s).all():raise ValueError('State18 required')
        q,m=self.codec.encode(obs['range_m'],obs['hit_mask'],obs['scanned_mask']);return s,q,m

    def add(self,obs,action,next_obs,reward,terminated,truncated,*,episode_id=0,timestamps=(0.,1/15)):
        # Validate everything before mutating the ring.
        s,q,m=self.encode_observation(obs);n,nq,nm=self.encode_observation(next_obs)
        a=np.asarray(action,np.float32);ts=np.asarray(timestamps,float)
        if a.shape!=(3,) or not np.isfinite(a).all() or np.max(abs(a))>1.000001 or np.linalg.norm(a[:2])>1.000001:raise ValueError('canonical projected action')
        if not np.isfinite(reward) or ts.shape!=(2,) or not np.isfinite(ts).all() or ts[1]<=ts[0]:raise ValueError('reward/simulation timestamp')
        i=self.pointer
        self.states[i]=s;self.next_states[i]=n;self.ranges[i]=q;self.next_ranges[i]=nq;self.masks[i]=m;self.next_masks[i]=nm
        self.actions[i]=a;self.rewards[i]=reward;self.terminated[i]=terminated;self.truncated[i]=truncated
        self.episode_ids[i]=episode_id;self.sensor_times[i]=ts
        self.pointer=(i+1)%self.capacity;self.size=min(self.size+1,self.capacity);self.total+=1

    def _batch(self,indices,next_state=False):
        states=self.next_states if next_state else self.states;ranges=self.next_ranges if next_state else self.ranges;masks=self.next_masks if next_state else self.masks
        q=ranges[indices];m=masks[indices];shape=(len(indices),*self.codec.shape)
        hit=np.unpackbits(m[:,0],axis=-1,bitorder='little',count=self.codec.size).reshape(shape)
        scanned=np.unpackbits(m[:,1],axis=-1,bitorder='little',count=self.codec.size).reshape(shape)
        normalized=np.where(scanned,q.astype(np.float32)/65534.,0)
        image=np.stack((normalized,hit,scanned),axis=1).astype(np.float32)
        return {'state':torch.from_numpy(states[indices]),'image':torch.from_numpy(image)}

    def sample(self,batch_size):
        if self.size<1:raise ValueError('empty Replay')
        i=self.rng.integers(self.size,size=batch_size)
        return self._batch(i),torch.from_numpy(self.actions[i]),self._batch(i,True),torch.from_numpy(self.rewards[i]),torch.from_numpy((~(self.terminated[i]|self.truncated[i])).astype(np.float32))

    def payload(self):
        names=('states','next_states','ranges','next_ranges','masks','next_masks','actions','rewards','terminated','truncated','episode_ids','sensor_times')
        return dict(schema=SCHEMA,contract=self.contract,capacity=self.capacity,size=self.size,pointer=self.pointer,total=self.total,
                    rng=self.rng.bit_generator.state,arrays={name:getattr(self,name)[:self.size].copy() for name in names})
    def save(self,path):torch.save(self.payload(),Path(path))
    @classmethod
    def load(cls,path,expected_contract):
        data=torch.load(path,weights_only=False,map_location='cpu')
        if data.get('schema')!=SCHEMA:raise ValueError('old Replay rejected')
        validate_contract(data['contract'],expected_contract);obj=cls(data['capacity'],expected_contract)
        if not 0<=data['size']<=obj.capacity or not 0<=data['pointer']<obj.capacity:raise ValueError('invalid Replay ring')
        for name,array in data['arrays'].items():
            target=getattr(obj,name)
            if target[:data['size']].shape!=array.shape or target.dtype!=array.dtype:raise ValueError('corrupt Replay array')
            target[:data['size']]=array
        obj.size=data['size'];obj.pointer=data['pointer'];obj.total=data['total'];obj.rng.bit_generator.state=data['rng']
        return obj

class SuccessReplay:
    """Frozen dv0 sampling rule: 80% main + 20% balanced successful Stage0-3."""
    def __init__(self,contract,capacity=60000,success_capacity=20000,fraction=.2,seed=43):
        if success_capacity<4 or success_capacity%4:raise ValueError('four equal success strata')
        self.contract=contract;self.fraction=fraction;self.main=Replay(capacity,contract,seed)
        self.success={d:Replay(success_capacity//4,contract,seed+d) for d in (10,20,30,40)}
        self.pending=[];self.last_sample_counts={}
    def add(self,*args,**kwargs):
        self.main.add(*args,**kwargs)
        # Raw inputs, not encoder features; retained only until episode outcome.
        import copy
        self.pending.append((copy.deepcopy(args),copy.deepcopy(kwargs)))
    def finish(self,success,density):
        if density not in self.success:raise ValueError('unknown density stratum')
        if success:
            for args,kwargs in self.pending:self.success[density].add(*args,**kwargs)
        self.pending=[]
    def sample(self,batch_size):
        available=[d for d,b in self.success.items() if b.size]
        n=int(batch_size*self.fraction) if available else 0
        batches=[self.main.sample(batch_size-n)];self.last_sample_counts={'main':batch_size-n}
        for i,d in enumerate(available):
            count=n//len(available)+(i<n%len(available));self.last_sample_counts[str(d)]=count
            if count:batches.append(self.success[d].sample(count))
        perm=torch.randperm(batch_size)
        def merge(items):
            if isinstance(items[0],dict):return {k:torch.cat([x[k] for x in items])[perm] for k in items[0]}
            return torch.cat(items)[perm]
        return tuple(merge(list(parts)) for parts in zip(*batches))
    def save(self,path):
        torch.save(dict(schema=SCHEMA,contract=self.contract,kind='balanced_success_v1',fraction=self.fraction,
            main=self.main.payload(),success={d:b.payload() for d,b in self.success.items()},pending=self.pending),path)
    @classmethod
    def load(cls,path,expected_contract):
        data=torch.load(path,weights_only=False,map_location='cpu')
        if data.get('schema')!=SCHEMA or data.get('kind')!='balanced_success_v1':raise ValueError('old Replay rejected')
        validate_contract(data['contract'],expected_contract)
        obj=cls(expected_contract,data['main']['capacity'],sum(x['capacity'] for x in data['success'].values()),data['fraction'])
        def restore(ring,payload):
            validate_contract(payload['contract'],expected_contract)
            if not 0<=payload['size']<=ring.capacity or not 0<=payload['pointer']<ring.capacity:raise ValueError('invalid ring')
            for name,array in payload['arrays'].items():
                target=getattr(ring,name)
                if target[:payload['size']].shape!=array.shape or target.dtype!=array.dtype:raise ValueError('corrupt Replay')
                target[:payload['size']]=array
            ring.size=payload['size'];ring.pointer=payload['pointer'];ring.total=payload['total'];ring.rng.bit_generator.state=payload['rng']
        restore(obj.main,data['main'])
        for d,payload in data['success'].items():restore(obj.success[d],payload)
        obj.pending=data['pending'];return obj
    def diagnostics(self):
        b=self.main
        return dict(size=b.size,total=b.total,terminated_fraction=float(b.terminated[:b.size].mean()) if b.size else 0,
            truncated_fraction=float(b.truncated[:b.size].mean()) if b.size else 0,
            success_sizes={str(k):r.size for k,r in self.success.items()},sample_counts=self.last_sample_counts,
            pending=len(self.pending),allocated_bytes=sum(a.nbytes for r in [b,*self.success.values()] for a in vars(r).values() if isinstance(a,np.ndarray)))
