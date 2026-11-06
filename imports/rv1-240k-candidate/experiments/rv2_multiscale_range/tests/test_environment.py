import numpy as np
from experiments.rv1_range_cnn.environment import RangeImageEnv
from experiments.rv2_multiscale_range.contracts import PerceptionSpec
from experiments.rv2_multiscale_range.models import Actor
from experiments.rv2_multiscale_range.features import canonical_image
import torch

def test_frozen_environment_no_extra_scan_or_task_change():
    env=RangeImageEnv();control=RangeImageEnv()
    try:
        a,_=env.reset(seed=43);b,_=control.reset(seed=43)
        actor=Actor(PerceptionSpec(bypass=True))
        for i in range(12):
            before=env.lidar.scan_count
            x=dict(state=torch.from_numpy(a['state']).unsqueeze(0),image=torch.from_numpy(canonical_image(a)).unsqueeze(0))
            with torch.no_grad():
                for _ in range(3):actor(x)
            assert env.lidar.scan_count==before
            action=np.array([.1,-.05,.02],np.float32)
            a,r,t,u,info=env.step(action);b,rr,tt,uu,ii=control.step(action)
            assert (r,t,u,info['done_reason'])==(rr,tt,uu,ii['done_reason'])
            np.testing.assert_array_equal(a['range_m'],b['range_m']);np.testing.assert_array_equal(a['state'],b['state'])
            assert info['lower_steps_executed']==8 and info['physics_substeps_executed']==16
            assert env.lidar.scan_count==i+2 and env.base_env.ray_casts==0
        env.reset(seed=43)
        assert env.controller.tick==0 and env.lidar.scan_count==1 and np.linalg.norm(env.controller.pi.integral)==0
    finally:env.close();control.close()
