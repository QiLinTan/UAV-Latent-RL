"""Atomic rv1 saves and exact current-episode physical reconstruction."""
import numpy as np
from experiments.direct_velocity_full_v1.rl.checkpoint import capture_episode,rng_state,restore_rng,atomic_save
from envs.forest.rewards import default_reward_terms
from .provenance import SCHEMA,validate_contract

def capture(env):
    return {**capture_episode(env), 'sensor_scan_count':env.lidar.scan_count,'sensor_batch_calls':env.lidar.batch_calls,
            'sensor_range':env.lidar.cache.range_m.copy(),'sensor_time_s':env.lidar.cache.timestamp_s}

def restore_environment(env,state):
    b=env.base_env;env._pool_index=state['pool_index'];env._density_episode_stage=state['density']
    b.MAP_SPEC=env.pool.spec(env.pool_split,env._pool_index,env._density_episode_stage);b.map_validation=None;b.reset()
    env.restore_course(state['course']);env._start_level=state['start_level'];env._start_source=state['start_source']
    env._episode_start_position=state['start_position'];env._remaining_path_length=state['remaining']
    if env._place_reset(state['start_position']):raise RuntimeError('resume initial pose collides')
    b._prev_pos=np.asarray(b.pos[0],dtype=np.float32).copy();b._prev_goal_dist=float(np.linalg.norm(b.TARGET_POS-b._prev_pos))
    b._last_collision=False;b._last_clearance=b._computeNearestTreeClearance(b._prev_pos)
    b._last_reward_terms=default_reward_terms();b._last_done_reason='running'
    env.initialize_control();env.lidar.reset();observation=env._observation(b._computeObs())
    for action in state['episode_actions']:observation,_,_,_,_=env.step(action)
    if not np.allclose(b.pos[0],state['position'],atol=1e-10,rtol=0) or env.controller.snapshot()!=state['controller']:raise RuntimeError('physical reconstruction mismatch')
    for key,actual in (('quat',b.quat[0]),('velocity',b.vel[0]),('omega',b.ang_v[0])):
        if not np.allclose(actual,state[key],atol=1e-10,rtol=0):raise RuntimeError('physical reconstruction mismatch: '+key)
    if env.upper_tick!=state['upper_tick'] or env.active!=state['active']:raise RuntimeError('episode reconstruction mismatch')
    if not np.array_equal(env.lidar.cache.range_m,state['sensor_range']) or env.lidar.scan_count!=state['sensor_scan_count']:raise RuntimeError('sensor reconstruction mismatch')
    b._reset_count=state['base']['_reset_count'];b._rng.bit_generator.state=state['base_rng'];env.last_ticks=[]
    return observation

def validate(payload,expected_contract):
    if payload.get('schema')!=SCHEMA or payload.get('format')!='rv1_training_smoke_v1':raise ValueError('old checkpoint rejected')
    validate_contract(payload['contract'],expected_contract)
