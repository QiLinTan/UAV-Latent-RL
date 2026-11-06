import copy, json, shutil
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.direct_velocity_full_v1 import task_factory as tf
from experiments.rv1_range_cnn.td3 import RangeTD3
from experiments.rv1_range_cnn.provenance import SCHEMA
from experiments.rv1_range_cnn.replay import SuccessReplay
from experiments.rv1_range_cnn.lidar import LidarConfig
from experiments.rv1_range_cnn.provenance import contract

def test_gate_rejects_changed_physical_source_hash(tmp_path,monkeypatch):
    original=tf.EXP
    for p in ['contracts.py','controls/velocity_tracking.py','controls/velocity_control_adapter.py','configs/control_acceptance_v2.json','outputs/acceptance_v2/metadata.json']:
        target=tmp_path/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original/p,target)
    metadata=tmp_path/'outputs/acceptance_v2/metadata.json';data=json.loads(metadata.read_text())
    key=next(x for x in data['source_hashes'] if x.startswith('/home/tan/UAV-Latent-RL/envs/'))
    data['source_hashes'][key]='0'*64;metadata.write_text(json.dumps(data))
    monkeypatch.setattr(tf,'EXP',tmp_path)
    with pytest.raises(RuntimeError,match='shared physical/control source changed'):tf.gate_a_check(tf.load_config())

def test_device_cpu_batches_networks_updates_and_action():
    cfg=copy.deepcopy(tf.load_config()['td3']);cfg['batch_size']=8
    c=contract(LidarConfig(),'device_test'); agent=RangeTD3(cfg,c,device='cpu')
    replay=SuccessReplay(c,capacity=16,success_capacity=16)
    def observation(seed):
        rng=np.random.default_rng(seed)
        return dict(state=rng.normal(size=18).astype(np.float32),range_m=rng.uniform(.1,8,(32,128)).astype(np.float32),hit_mask=np.ones((32,128),bool),scanned_mask=np.ones((32,128),bool))
    for i in range(12):replay.add(observation(i),[.1,.1,.1],observation(i+1),.1,False,False,episode_id=0,timestamps=(i,i+1))
    for name in ('actor','critic','actor_target','critic_target'):assert next(getattr(agent,name).parameters()).device.type=='cpu'
    for _ in range(2): assert np.isfinite(agent.update(replay)['critic_loss'])
    assert agent.actor_updates==1 and agent.action(observation(0)).shape==(3,)
    state=agent.state();restored=RangeTD3(cfg,c,device='cpu');restored.restore(state)
    for optimizer in (restored.actor_optimizer,restored.critic_optimizer):
        for row in optimizer.state.values():
            for v in row.values():
                if isinstance(v,torch.Tensor):assert v.device.type=='cpu'
    batch=agent._batch(replay,8)
    assert all(t.device.type=='cpu' for x in batch for t in (x.values() if isinstance(x,dict) else [x]))


def test_next_update_after_fresh_process(tmp_path):
    import subprocess,sys
    from experiments.rv1_range_cnn.checkpoint import rng_state,restore_rng
    cfg=copy.deepcopy(tf.load_config()['td3']);cfg['batch_size']=8
    c=contract(LidarConfig(),'process_test');agent=RangeTD3(cfg,c,device='cpu')
    replay=SuccessReplay(c,capacity=16,success_capacity=16)
    def observation(seed):
        rng=np.random.default_rng(seed)
        return dict(state=rng.normal(size=18).astype(np.float32),range_m=rng.uniform(.1,8,(32,128)).astype(np.float32),hit_mask=np.ones((32,128),bool),scanned_mask=np.ones((32,128),bool))
    for i in range(12):replay.add(observation(i),[.1,.1,.1],observation(i+1),.1,False,False,episode_id=0,timestamps=(i,i+1))
    for _ in range(2):agent.update(replay)
    replay.save(tmp_path/'replay.pt');state=rng_state()
    torch.save({'agent':agent.state(),'rng':state,'config':cfg,'contract':c},tmp_path/'before.pt')
    result=agent.update(replay);torch.save({'result':result,'agent':agent.state()},tmp_path/'reference.pt')
    child="""import sys,torch
from pathlib import Path
from experiments.rv1_range_cnn.td3 import RangeTD3
from experiments.rv1_range_cnn.replay import SuccessReplay
from experiments.rv1_range_cnn.checkpoint import restore_rng
p=Path(sys.argv[1]); x=torch.load(p/'before.pt',map_location='cpu',weights_only=False)
a=RangeTD3(x['config'],x['contract']);a.restore(x['agent']);r=SuccessReplay.load(p/'replay.pt',x['contract']);restore_rng(x['rng']);d=a.update(r)
torch.save({'result':d,'agent':a.state()},p/'child.pt')
"""
    subprocess.run([sys.executable,'-B','-c',child,str(tmp_path)],check=True)
    expected=torch.load(tmp_path/'reference.pt',weights_only=False);actual=torch.load(tmp_path/'child.pt',weights_only=False)
    assert expected['result']==actual['result']
    for name in ('actor','critic','actor_target','critic_target'):
        for key,value in expected['agent'][name].items():torch.testing.assert_close(value,actual['agent'][name][key],rtol=0,atol=0)
    assert actual['agent']['updates']==3 and actual['agent']['actor_updates']==1
