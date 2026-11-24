import json, pathlib, sys
import numpy as np
from multiscale import Encoder,rv1_directions
p=pathlib.Path(sys.argv[1]); data=np.load(p); images=data['image']; labels=data['labels']; enc=Encoder(); out=[]
for i in [0,132,264,396]:
    im=images[i]; scan=im[2].astype(bool); hit=im[1].astype(bool); r=im[0]*8
    r[~scan]=np.nan
    f,g=enc.encode(rv1_directions(),r,hit,scan)
    np.savez_compressed(pathlib.Path(__file__).parent/f'archive_frame_{i}.npz',grid=g,features=f,image=im,labels=labels[i])
    far=f[250:].reshape(26,2)
    out.append(dict(index=i,labels=labels[i].tolist(),hits=int(hit.sum()),near_occ=int(f[:250].reshape(-1,2)[:,0].sum()),far_max_occ=float(far[:,0].max()),far_max_free=float(far[:,1].max())))
print(json.dumps(out,indent=2))
