"""One immutable configuration for training, evaluation and data compatibility."""
from pathlib import Path
import hashlib,json
import gym_pybullet_drones
import numpy as np
from envs.forest.rewards import TerminalGuidanceRateIntegratedReward
from envs.map_atlas.forest_pool import ForestPool
from gym_pybullet_drones.utils.enums import DroneModel,Physics,ObservationType,ActionType
from .contracts import ControlSpec
from .envs.direct_velocity_full import DirectVelocityFullEnv
from .envs.curriculum_adapter import CurriculumEnv

EXP=Path(__file__).resolve().parent
ROOT=EXP.parents[1]
CONFIG=EXP/'configs/dv0_navigation_v1.json'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def load_config():
    c=json.loads(CONFIG.read_text())
    if c['version']!='dv0_velocity_full_nav_v1':raise ValueError('unknown config')
    return c


def source_hashes():
    paths=[p for p in EXP.rglob('*.py') if 'outputs' not in p.parts and '__pycache__' not in p.parts]
    paths += [ROOT/x for x in ('envs/forest/core.py','envs/forest/rewards.py','envs/forest/start_curriculum.py',
        'envs/forest/obstacle_curriculum.py','envs/map_atlas/forest_pool.py','envs/map_atlas/runtime.py',
        'envs/map_atlas/spec.py','algos/td3/networks.py','models/motor_action_codec.py')]
    paths += list((ROOT/'envs/forest').glob('*.py'))+list((ROOT/'envs/map_atlas').glob('*.py'))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
    package=Path(gym_pybullet_drones.__file__).resolve().parent
    for name in ('control/DSLPIDControl.py','envs/BaseAviary.py','envs/BaseRLAviary.py','assets/cf2x.urdf'):
        hashes['.wsl2/drones-cpu/lib/python3.10/site-packages/gym_pybullet_drones/'+name]=hashlib.sha256((package/name).read_bytes()).hexdigest()
    return hashes


def gate_a_check(config):
    report_path=ROOT/'evidence/gate_a.json'
    report=json.loads(report_path.read_text())
    if report['status']!='PASS_STEP1_COMPLETE':raise RuntimeError('Gate A not passed')
    old=json.loads((EXP/'outputs/acceptance_v2/metadata.json').read_text())['source_hashes']
    for name in ('contracts.py','controls/velocity_tracking.py','controls/velocity_control_adapter.py'):
        p=EXP/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=old[name]:raise RuntimeError('Gate A controller changed')
    old_root='/home/tan/UAV-Latent-RL/'
    package_prefix='.wsl2/drones-cpu/lib/python3.10/site-packages/gym_pybullet_drones/'
    for name,expected in old.items():
        if name.startswith('/'):
            if not name.startswith(old_root):raise RuntimeError('unknown Gate A source path: '+name)
            rel=name[len(old_root):]
            if '..' in Path(rel).parts:raise RuntimeError('invalid Gate A path')
            p=(Path(gym_pybullet_drones.__file__).resolve().parent/rel[len(package_prefix):]) if rel.startswith(package_prefix) else ROOT/rel
            if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
                raise RuntimeError('Gate A shared physical/control source changed: '+name)
    frozen=json.loads((EXP/'configs/control_acceptance_v2.json').read_text())['control_spec']
    if config['control_spec']!=frozen:raise RuntimeError('control parameters changed')


def make_reward(config):
    return TerminalGuidanceRateIntegratedReward(**config['reward_kwargs'])


def make_env(config=None,*,training=True,pool=None,empty=False):
    config=config or load_config();gate_a_check(config)
    kwargs=dict(drone_model=DroneModel.CF2X,physics=Physics.PYB,pyb_freq=240,ctrl_freq=120,gui=False,
        record=False,obs=ObservationType.KIN,act=ActionType.RPM,range_azimuth_count=32,
        range_elevation_degrees=(-30.,0.,30.),curriculum=False,route_count_curriculum=False,
        curriculum_simple_mix_probability=0.,reward_model=make_reward(config))
    spec=ControlSpec(**config['control_spec'])
    if empty:
        kwargs.update(num_trees=0,route_blocking_tree=False,initial_xyzs=np.array([[-3.5,0,1]]),target_pos=[3.5,0,1],episode_len_sec=20.)
        return DirectVelocityFullEnv(control_spec=spec,plant_kwargs=kwargs)
    pool=pool or ForestPool(ROOT/config['forest_pool_path'])
    return CurriculumEnv(control_spec=spec,plant_kwargs=kwargs,pool=pool,training_course=training,
                         split='train' if training else 'validation')


def data_contract(config,pool):
    return dict(schema='dv0_114D_yaw_velocity_replay_v1',observation=114,action=3,
        observation_spec=dict(frame='yaw_horizontal',body_omega_gravity=True,height_bounds=[.08,2.75],
            distance_scale=20.,velocity_scale=[.8,.8,.3],omega_scale=4.,no_history=True),
        action_spec=dict(frame='yaw_horizontal_at_upper_tick',projection='clip_then_xy_disk',limits=[.8,.3]),
        range_spec=dict(count=96,elevations=[-30.,0.,30.],azimuths=32,mode='yaw_rotated_raycast_every_lower_tick'),
        task=dict(reward_kwargs=config['reward_kwargs'],reward_scale=.1,done_mask='terminated_OR_truncated',
            pool_fingerprint=pool.fingerprint,curriculum=config['curriculum']),
        config_sha256=digest(config),sources_sha256=digest(source_hashes()))
