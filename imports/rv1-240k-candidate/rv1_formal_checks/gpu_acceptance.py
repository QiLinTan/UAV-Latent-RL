"""RTX acceptance: actual rv1 Replay transfer and TD3 update, no environment training."""
import copy,json,time,resource
import numpy as np
import torch
from experiments.direct_velocity_full_v1.task_factory import load_config
from experiments.rv1_range_cnn.lidar import LidarConfig
from experiments.rv1_range_cnn.provenance import contract
from experiments.rv1_range_cnn.replay import SuccessReplay
from experiments.rv1_range_cnn.td3 import RangeTD3
assert torch.cuda.is_available(), 'CUDA unavailable'
cfg=copy.deepcopy(load_config()['td3']); cfg['batch_size']=32
c=contract(LidarConfig(),'gpu_acceptance_only')
a=RangeTD3(cfg,c,device='cuda');r=SuccessReplay(c,capacity=64,success_capacity=16)
def obs(seed):
    rng=np.random.default_rng(seed)
    return dict(state=rng.normal(size=18).astype(np.float32),range_m=rng.uniform(.1,8,(32,128)).astype(np.float32),hit_mask=np.ones((32,128),bool),scanned_mask=np.ones((32,128),bool))
for i in range(32):r.add(obs(i),[.1,.1,.1],obs(i+1),.1,False,False,episode_id=0,timestamps=(i,i+1))
batch=a._batch(r,32)
assert all(t.is_cuda for x in batch for t in (x.values() if isinstance(x,dict) else [x]))
for _ in range(4):a.update(r)
torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
for _ in range(10):
    start=time.perf_counter();d=a.update(r);torch.cuda.synchronize();times.append(1000*(time.perf_counter()-start))
    assert np.isfinite(d['critic_loss']) and (d['actor_loss'] is None or np.isfinite(d['actor_loss']))
assert a.actor_updates>0 and a.action(obs(0)).shape==(3,)
for name in ('actor','critic','actor_target','critic_target'):assert next(getattr(a,name).parameters()).is_cuda
print(json.dumps({'gpu':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'torch':torch.__version__,'cuda':torch.version.cuda,'batch':32,'updates':10,'median_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'note':'real CPU Replay sampling and transfer, excludes physical simulation'},indent=2))
