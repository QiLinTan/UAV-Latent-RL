from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from envs.map_atlas import MapCatalog, MapSpec
from envs.map_atlas.ego_forest import EgoForestConfig, generate_ego_forest
from envs.learned_upper_dslpid import LearnedUpperDSLPIDForestEnv
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


CATALOG = Path('data/map_atlas/catalog_v1.json')
MAP_ID = 'ego_forest_20x10_seed43_v1'


def test_generated_forest_is_reproducible_bounded_and_clearance_reachable():
    record,meta,path=generate_ego_forest(seed=43)
    again,meta2,path2=generate_ego_forest(seed=43)
    assert record==again and meta['geometry_sha256']==meta2['geometry_sha256']
    np.testing.assert_array_equal(path,path2)
    different,other,_=generate_ego_forest(seed=44)
    assert other['geometry_sha256']!=meta['geometry_sha256']
    assert len(record['obstacles'])==40
    xy=np.array([r['position'][:2] for r in record['obstacles']])
    radius=np.array([r['radius'] for r in record['obstacles']])
    assert np.all(np.abs(xy)+radius[:,None]<[10,5])
    spacing=np.linalg.norm(xy[:,None,:]-xy[None,:,:],axis=2)
    np.fill_diagonal(spacing,np.inf)
    assert spacing.min()>=1.
    assert np.all((radius>=.2)&(radius<=.3))
    # Check swept QA segments, not just individual path nodes.
    interpolated=np.concatenate([np.linspace(a,b,5) for a,b in zip(path,path[1:])])
    clearance=np.linalg.norm(interpolated[:,None,:2]-xy[None,:,:],axis=2)-radius[None,:]-.06
    assert clearance.min()>=.2
    assert np.all(np.abs(interpolated[:,:2])<=[10,5])


def test_invalid_or_unplaceable_generator_fails_with_bounded_work():
    with pytest.raises(ValueError):
        generate_ego_forest(replace(EgoForestConfig(),tree_count=0))
    with pytest.raises(RuntimeError,match='Placed'):
        generate_ego_forest(replace(EgoForestConfig(),tree_count=2,minimum_center_distance=100))


def test_rectangular_route_cannot_escape_around_a_wall_on_short_axis():
    record,_,_=generate_ego_forest(seed=43)
    record['obstacles']=[dict(kind='box',position=[0,0,1],half_extents=[.2,5.1,2])]
    spec=MapSpec.from_dict(record,root=Path('.'))
    assert not spec.validate_route(resolution=.5).reachable
    assert spec.endpoint_inside_bounds(np.array([9,4,1]))
    assert not spec.endpoint_inside_bounds(np.array([0,6,1]))
    record['xy_half_extents']=[10,-5]
    with pytest.raises(ValueError,match='xy_half_extents'):
        MapSpec.from_dict(record,root=Path('.'))


def test_full_environment_uses_rectangle_task_time_and_static_geometry():
    spec=MapCatalog(CATALOG).get(MAP_ID)
    env=LearnedUpperDSLPIDForestEnv(map_spec=spec,obs=ObservationType.KIN,act=ActionType.RPM,
        pyb_freq=240,ctrl_freq=120,gui=False,range_azimuth_count=32,
        range_elevation_degrees=(-30,0,30),semantic_history_length=8)
    try:
        obs,info=env.reset(seed=43)
        assert obs.shape==(1,917) and info['num_obstacles']==40
        assert env.EPISODE_LEN_SEC==40
        assert not env.CURRICULUM_ENABLED and not env.ROUTE_COUNT_CURRICULUM
        assert info['goal_distance']==16.
        np.testing.assert_allclose(env.pos[0],[-8,0,1])
        assert not env.base_env._checkCollision()
        geometry=[r.position for r in env.base_env._map_runtime.spec.obstacles]
        env.reset(seed=44)
        assert geometry==[r.position for r in env.base_env._map_runtime.spec.obstacles]
        obs,reward,terminated,truncated,info=env.step(np.zeros((1,3),dtype=np.float32))
        assert np.isfinite(obs).all() and np.isfinite(reward)
        assert info['lower_steps_executed']==8 and info['reference_packet_valid']
        assert not terminated and not truncated
        state=env.base_env._getDroneStateVector(0).copy()
        state[:3]=[0,5.01,1]
        assert env.base_env._classifyDoneReason(state,collision=False)=='xy_bound'
        state[:3]=[10.01,0,1]
        assert env.base_env._classifyDoneReason(state,collision=False)=='xy_bound'
        env.base_env.step_counter=int(39*env.PYB_FREQ)
        assert not env.base_env._computeTruncated()
        env.base_env.step_counter=int(41*env.PYB_FREQ)
        assert env.base_env._computeTruncated()
    finally:
        env.close()
