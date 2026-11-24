import time, json, numpy as np
from experiments.rv1_range_cnn.environment import RangeImageEnv
from multiscale import Encoder,rv1_directions
enc=Encoder(); env=RangeImageEnv()
try:
 t=time.perf_counter(); obs,info=env.reset(seed=43); reset=time.perf_counter()-t
 d=rv1_directions(); t=time.perf_counter(); feature,grid=enc.encode(d,obs['range_m'],obs['hit_mask'],obs['scanned_mask']); elapsed=time.perf_counter()-t
 np.savez_compressed('rv1_reset_seed43.npz', **obs, grid=grid, features=feature)
 print(json.dumps(dict(reset_seconds=reset, encode_seconds=elapsed, scanned=int(obs['scanned_mask'].sum()), hits=int(obs['hit_mask'].sum()),near_occupied=int(feature[:250].reshape(-1,2)[:,0].sum()),far_max_occupied=float(feature[250:].reshape(26,2)[:,0].max()),info=info),default=str,indent=2))
finally: env.close()
