import sys,torch,numpy as np
from collections import deque
from pathlib import Path
base=Path(sys.argv[1])
a=torch.load(base/'direct/checkpoint_000004.pt',map_location='cpu',weights_only=False)
b=torch.load(base/'split/checkpoint_000004.pt',map_location='cpu',weights_only=False)
def compare(a,b,path):
    if isinstance(a,dict):
        assert set(a)==set(b),(path,'keys')
        for k in a:
            if k not in ('bullet_snapshot','replay_sha256'):compare(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple,deque)):
        assert len(a)==len(b),(path,'length')
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
    elif isinstance(a,torch.Tensor): assert torch.equal(a,b),path
    elif isinstance(a,np.ndarray): assert np.array_equal(a,b,equal_nan=True),path
    else: assert a==b,(path,a,b)
compare(a,b,'checkpoint')
r1=torch.load(base/'direct/replay_000004.pt',map_location='cpu',weights_only=False)
r2=torch.load(base/'split/replay_000004.pt',map_location='cpu',weights_only=False)
compare(r1,r2,'replay')
print('PASS: trainer, networks, optimizers, targets, RNG, course, reconstructed physical state, observation and replay contents match; bullet byte snapshot excluded')

