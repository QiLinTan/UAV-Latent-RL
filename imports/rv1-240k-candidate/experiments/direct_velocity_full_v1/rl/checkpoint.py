"""Versioned atomic checkpoint and explicit rigid-body episode restoration."""
import copy,os,random,tempfile
from pathlib import Path
import numpy as np
import pybullet as p
import torch
from ..contracts import VelocityCommand


def capture_episode(env):
    b=env.base_env
    with tempfile.TemporaryDirectory(prefix='dv0_bullet_') as directory:
        snapshot=Path(directory)/'state.bullet';p.saveBullet(str(snapshot),physicsClientId=b.CLIENT)
        bullet=snapshot.read_bytes()
    return dict(bullet_snapshot=bullet,course=env.course_state(),pool_index=env._pool_index,density=env._density_episode_stage,
        start_level=env._start_level,start_source=env._start_source,start_position=env._episode_start_position,
        remaining=env._remaining_path_length,position=b.pos[0].copy(),quat=b.quat[0].copy(),
        velocity=b.vel[0].copy(),omega=b.ang_v[0].copy(),controller=env.controller.snapshot(),
        upper_tick=env.upper_tick,active=env.active,episode_actions=copy.deepcopy(env.episode_actions),base_rng=copy.deepcopy(b._rng.bit_generator.state),
        base={k:copy.deepcopy(getattr(b,k)) for k in ('step_counter','physics_ticks','accepted_world','_prev_pos',
            '_prev_goal_dist','_last_collision','_last_clearance','_last_reward_terms','_last_done_reason','_last_motor_rpm',
            '_reset_count','action_buffer','last_clipped_action')})


def restore_episode(env,state):
    b=env.base_env
    env._pool_index=state['pool_index'];env._density_episode_stage=state['density']
    b.MAP_SPEC=env.pool.spec(env.pool_split,env._pool_index,env._density_episode_stage);b.map_validation=None
    b.reset()
    # Bullet's serialized state misses link-frame force caches. Reconstruct the
    # short current episode from its reset and accepted raw action tape instead.
    # This replay creates no training transitions and performs no optimizer work.
    env.restore_course(state['course']);env._start_level=state['start_level'];env._start_source=state['start_source']
    env._episode_start_position=state['start_position'];env._remaining_path_length=state['remaining']
    if env._place_reset(state['start_position']):raise RuntimeError('resume initial pose collides')
    from envs.forest.rewards import default_reward_terms
    b._prev_pos=np.asarray(b.pos[0],dtype=np.float32).copy()
    b._prev_goal_dist=float(np.linalg.norm(b.TARGET_POS-b._prev_pos))
    b._last_collision=False;b._last_clearance=b._computeNearestTreeClearance(b._prev_pos)
    b._last_reward_terms=default_reward_terms();b._last_done_reason='running'
    env.initialize_control()
    for action in state['episode_actions']:env.step(action)
    if not np.allclose(b.pos[0],state['position'],atol=1e-10,rtol=0) or env.controller.snapshot()!=state['controller']:
        raise RuntimeError('deterministic current-episode reconstruction failed')
    b._reset_count=state['base']['_reset_count']
    b._rng.bit_generator.state=state['base_rng']
    env.last_ticks=[]
    return b._computeObs()


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])


def atomic_save(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temporary)
    with temporary.open('rb') as stream:os.fsync(stream.fileno())
    os.replace(temporary,path)


def validate(payload,contract):
    if payload.get('format')!='dv0_training_v1' or payload.get('contract')!=contract:
        raise ValueError('Checkpoint schema/config/source mismatch; old models rejected')
