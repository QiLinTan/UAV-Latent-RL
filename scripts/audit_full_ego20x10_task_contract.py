"""M0 task contract audit: read-only learners, isolated Full probes, no training."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.audit_ego20x10_value import (
    SOURCE, LOGS, load, env_for, relocate_reset, describe,
)
from scripts.audit_stage0_feasibility import make_stage0_env
from envs.learned_upper_dslpid import LearnedUpperDSLPIDForestEnv
from envs.preprocess import preprocess_state
from envs.map_atlas.ego_forest import clearance_route

ROOT = Path('analysis_outputs/full_ego20x10_task_contract_audit_v1')
VALUE = Path('analysis_outputs/full_scratch_ego20x10_curr_v1_value_audit_seed43')
COSTS = ('time_penalty','height_penalty','vertical_speed_penalty','lateral_penalty','proximity_penalty',
         'attitude_penalty','speed_penalty','near_goal_speed_penalty','near_goal_away_penalty',
         'route_overshoot_penalty','safety_boundary_penalty','near_goal_velocity_tracking_penalty')


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+'\n')


def json_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def raw_state(position, velocity, rpy=None):
    state=np.zeros(20,dtype=np.float32)
    state[:3]=position
    state[7:10]=np.zeros(3) if rpy is None else rpy
    state[10:13]=velocity
    return state


def reward_on(model, env, state, previous, clearance=np.inf, collision=False, terminal='running'):
    return model.compute(state=state,prev_goal_dist=float(np.linalg.norm(previous[:3]-env.TARGET_POS)),
                         prev_pos=previous[:3],start_pos=env.START_POS,target_pos=env.TARGET_POS,
                         goal_tolerance=env.GOAL_TOLERANCE,safe_distance=env.SAFE_DISTANCE,
                         clearance=clearance,collision=collision,terminal_reason=terminal)


def trace_reward(path, metadata, env):
    """Recompute every actual lower reward from the frozen production model."""
    # Load each compressed array once; indexing an NpzFile repeatedly otherwise
    # decompresses the entire trajectory on every tick.
    with np.load(path) as archive:
        data={key:archive[key] for key in archive.files}
    model=env.base_env._reward_model
    previous=raw_state(metadata['initial_position'],metadata.get('initial_velocity',[0,0,0]))
    totals={'progress_reward':0.,'terminal_outcome_reward':0.,**{k:0. for k in COSTS}}
    region={'outside_guidance':{k:0. for k in totals},'inside_guidance':{k:0. for k in totals}}
    discounted=0.;samples=[];max_error=0.
    for index in range(len(data['lower_reward'])):
        state=raw_state(data['lower_position'][index],data['lower_velocity'][index],data['lower_rpy'][index])
        # Saved float times can round 40.0 upward; use the actual recorded
        # episode outcome, since the environment exits on its first terminal tick.
        terminal=metadata['done_reason'] if index==len(data['lower_reward'])-1 else 'running'
        if terminal=='p0_budget_boundary':terminal='running'
        value,terms,distance=reward_on(model,env,state,previous,float(data['lower_clearance'][index]),bool(data['lower_collision'][index]),terminal)
        error=abs(value-float(data['lower_reward'][index]));max_error=max(max_error,error)
        assert error<2e-5,(path,index,error,value,data['lower_reward'][index])
        label='inside_guidance' if distance<1.2 else 'outside_guidance'
        for key in totals:
            totals[key]+=float(terms[key]);region[label][key]+=float(terms[key])
        # Learner gamma is per upper transition, not per lower tick.
        upper_index=int(np.floor(index/8))
        discounted+=.99**upper_index*value
        samples.append(dict(distance=distance,speed=float(np.linalg.norm(state[10:13])),height_error=float(state[2]-1),
                            closing_speed=float(np.dot(state[10:13],(env.TARGET_POS-state[:3])/max(distance,1e-9))),time=float(data['lower_time_seconds'][index])))
        previous=state
    reconstructed=totals['progress_reward']-sum(totals[k] for k in COSTS)+totals['terminal_outcome_reward']
    np.testing.assert_allclose(reconstructed,data['lower_reward'].sum(),atol=2e-5,rtol=1e-7)
    result=dict(source=str(path),done_reason=metadata['done_reason'],duration_seconds=float(data['lower_time_seconds'][-1]),
                steps_upper=len(data['upper_reward']),raw_return=float(data['lower_reward'].sum()),discounted_raw_return=discounted,
                reward_components=totals,regions=region,maximum_lower_reward_error=max_error,
                guidance_time_seconds=sum(s['distance']<1.2 for s in samples)/120)
    reached={}
    for radius in (.2,.3,.5,1.2):
        crossings=[i for i,s in enumerate(samples) if s['distance']<radius]
        reached[str(radius)]=dict(reached=bool(crossings),first_time_s=samples[crossings[0]]['time'] if crossings else None,
                                 first_speed_mps=samples[crossings[0]]['speed'] if crossings else None,
                                 first_height_error_m=samples[crossings[0]]['height_error'] if crossings else None,
                                 first_collision=bool(data['lower_collision'][crossings[0]]) if crossings else None,
                                 time_inside_s=len(crossings)/120)
    result['radius_arrivals']=reached
    result['minimum_distance']=min(float(np.linalg.norm(np.array(metadata['initial_position'])-env.TARGET_POS)),min(s['distance'] for s in samples))
    return result


def trajectory_files():
    result=[]
    for step,context in ((2000,'final_raw'),(5000,'periodic_raw'),(6000,'final_raw')):
        folder=next((LOGS/'validation'/context).glob(f'step{step}_source{step}_*'))/'density0'
        for index in range(20):
            path=folder/f'map{index:03d}.npz'
            result.append((f'validation16m_{step}',path,json.loads(path.with_suffix('.json').read_text())))
    for path in sorted((LOGS/'p0_audit/traces').glob('episode_at_*.npz')):
        if path.stem.endswith('_replay'):continue
        result.append(('training_saved_traces',path,json.loads(path.with_suffix('.json').read_text())))
    return result


def state_reward(root):
    saved=load(6000)
    states=saved['replay']['arrays']['state'];next_states=saved['replay']['arrays']['next_state']
    env=env_for(saved,training_course=False,pool_split='validation')
    try:
        model=env.base_env._reward_model
        write(root/'resolved_reward_parameters.json',vars(model))
        positions=states[:,:3]*.5+np.array([0,0,1])
        delta=np.array([8,0,1])-positions
        expected=np.clip(delta/np.array([10,10,2]),-1,1)
        np.testing.assert_allclose(states[:,12:15],expected,atol=1e-6,rtol=0)
        alias=[dict(x=float(x),distance=float(8-x),goal_x=float(np.clip((8-x)/10,-1,1)),normalized_position_x=float(x/.5)) for x in (-10,-8,-6,-4,-2,0,4,7,7.7,7.8)]
        history=states[:,117:].reshape(-1,8,100)
        old=states[:,117:].reshape(-1,8,100)
        new=next_states[:,117:].reshape(-1,8,100)
        np.testing.assert_array_equal(old[:,1:],new[:,:-1])
        np.testing.assert_array_equal(states[:,15:111],history[:,-1,:96])
        blocks={'position':states[:,:3],'rpy':states[:,3:6],'velocity':states[:,6:9],'angular_velocity':states[:,9:12],
                'goal':states[:,12:15],'rays':states[:,15:111],'reference_error':states[:,111:114],
                'reference_velocity':states[:,114:117],'history_rays':history[:,:,:96],
                'history_commands':history[:,:,96:99],'history_tracking_error':history[:,:,99]}
        m0a=dict(shape=[117,800,917],replay_count=len(states),all_finite=bool(np.isfinite(states).all()),
                 feature_statistics={k:dict(shape=list(a.shape),min=float(a.min()),max=float(a.max()),rms=float(np.sqrt(np.mean(a**2)))) for k,a in blocks.items()},
                 goal_scaling=[10,10,2],goal_alias_examples=alias,
                 goal_x_clipping_fraction=float(np.mean(delta[:,0]>10)),
                 original_route_initial_saturated_length_m=6,original_route_initial_saturated_fraction=6/16,
                 absolute_position_inverse_is_unclipped=True,history_shift_exact=True,last_history_rays_duplicate_current=True,
                 history_intervals_seconds=8/15,oldest_current_time_offset_seconds=7/15,
                 history_fields=['96 rays','3 normalized last commands','one unsigned mean tracking-error scalar'],
                 no_explicit_fields=['target world position','distance scalar','time remaining','pose/velocity history','PID integrals','motor RPM history','persistent map'],
                 recurrent_memory='Explicit eight-frame GRU input; hidden state is initialized anew for each forward, not a 40s persistent memory.')
        write(root/'M0A_state.json',m0a)
        results=[]
        for label,path,metadata in trajectory_files():
            record=trace_reward(path,metadata,env);record['group']=label;results.append(record)
            print(f'[M0C replay] {label} {path.stem} error={record["maximum_lower_reward_error"]:.3g}',flush=True)
        write(root/'M0C_real_reward_components.json',results)
        # State-synthetic reward rates: no environment steps, no policy updates.
        sweep=[]
        for distance in (16.,8.,4.,1.2,1.1,1.,.8,.5,.3,.21):
            for label,velocity in (('toward_max',[.8,0,0]),('hover',[0,0,0]),('away_max',[-.8,0,0]),('vertical_up',[0,0,.3])):
                position=np.array([8-distance,0,1.])
                current=raw_state(position,velocity)
                previous=raw_state(position-np.array(velocity)/120,velocity)
                value,terms,_=reward_on(model,env,current,previous)
                sweep.append(dict(distance=distance,behavior=label,velocity=velocity,reward_rate_approx=value*120,
                                  progress_rate=terms['progress_reward']*120,tracking_cost_rate=terms['near_goal_velocity_tracking_penalty']*120))
        write(root/'M0C_reward_rate_sweep.json',sweep)
        feasibility=[]
        for distance in (.3,.5,.8,1.,1.1,1.19):
            for degrees in (0,30,60,90):
                angle=np.deg2rad(degrees)
                desired_speed=.1+np.clip((distance-.2)/1.0,0,1)*.7
                desired=np.array([desired_speed*np.cos(angle),0,desired_speed*np.sin(angle)])
                feasible=desired.copy();feasible[2]=np.clip(feasible[2],-.3,.3)
                error=desired-feasible
                feasibility.append(dict(distance=distance,goal_direction_elevation_deg=degrees,desired_speed=desired_speed,
                                        desired_velocity=desired.tolist(),best_speed_box_velocity=feasible.tolist(),
                                        unavoidable_tracking_cost_rate=160*np.clip((1.2-distance)/.2,0,1)*float(error@error)))
        desired_speed=.1+np.clip((np.linalg.norm(delta,axis=1)-.2),0,1)*.7
        desired=delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-9)*desired_speed[:,None]
        inside=np.linalg.norm(delta,axis=1)<1.2
        write(root/'M0C_guidance_feasibility.json',dict(sweep=feasibility,guidance_replay_count=int(inside.sum()),
             vertical_desired_exceeds_cap_fraction=float(np.mean(np.abs(desired[inside,2])>.3)),
             global_progress_active=True,absolute_distance_reward=False,route_progress_coefficient=0.,goal_progress_coefficient=8.))
    finally:
        env.close()
    # Offline thresholds on already recorded L0 trajectories, without altering termination.
    l0=json.loads((VALUE/'l0_validation.json').read_text());arrivals={}
    for step,eps in l0.items():
        arrivals[step]=[]
        for e in eps:
            distances=np.linalg.norm(np.asarray([s['position'] for s in e['samples']])-np.array([8,0,1]),axis=1)
            crossings={}
            for radius in (.2,.3,.5):
                ids=np.flatnonzero(distances<radius)
                crossings[str(radius)]=dict(reached=bool(len(ids)),first_upper_step=int(ids[0]+1) if len(ids) else None,
                                            first_speed=float(np.linalg.norm(e['samples'][ids[0]]['velocity'])) if len(ids) else None)
            arrivals[step].append(dict(map_id=e['map_id'],minimum_distance=e['minimum_goal_distance'],strict_outcome=e['done_reason'],radii=crossings))
    # Main replay and whole production episodes have only upper-sampled minima;
    # distinguish them from 120Hz lower traces rather than overstate precision.
    episodes=json_rows(LOGS/'p0_audit/episodes.jsonl')
    write(root/'M0D_offline_arrivals.json',dict(validation16m=[r for r in results if r['group'].startswith('validation16m')],
         L0_upper_sampled=arrivals,training_episode_minima=[dict(step=e['step'],minimum=e['minimum_goal_distance'],done_reason=e['done_reason']) for e in episodes],
         formal_success_radius=.2,formal_guidance_radius=1.2,formal_success_comparison='strict 3D distance < radius; no dwell/speed requirement; success precedes collision',
         note='0.3/0.5 are offline arrivals on trajectories terminated by the original task; not retrained success rates. L0 first-crossing timing is upper-sampled.'))
    print('[M0 A/C/D] observation/reward/arrival analysis complete',flush=True)


def horizontal_visibility(positions, obstacles, radius):
    """Exact analytic 32-beam horizontal cylinder intersections; unknowns censored."""
    pos=np.asarray(positions)[:,:2]
    angles=np.arange(32)*2*np.pi/32;directions=np.c_[np.cos(angles),np.sin(angles)]
    centers=np.array([o.position[:2] for o in obstacles]);radii=np.array([o.radius for o in obstacles])
    relative=centers[None,:,:]-pos[:,None,:]
    projection=np.einsum('ntd,bd->ntb',relative,directions)
    norm2=np.sum(relative**2,axis=2)[:,:,None]
    perpendicular2=norm2-projection**2
    discriminant=radii[None,:,None]**2-perpendicular2
    hits=projection-np.sqrt(np.maximum(discriminant,0))
    valid=(discriminant>=0)&(hits>=.08)&(hits<radius)
    hits=np.where(valid,hits,np.inf)
    nearest=np.min(hits,axis=1);bodies=np.argmin(hits,axis=1)
    seen=[]
    for i in range(len(pos)):
        seen.append(sorted(set(bodies[i,np.isfinite(nearest[i])].tolist())))
    return np.minimum(nearest,radius),seen


def segment_clear(a,b,obstacles,margin=.26):
    direction=b[:2]-a[:2];denom=float(direction@direction)
    for o in obstacles:
        center=np.array(o.position[:2])
        t=0 if denom<1e-12 else float(np.clip((center-a[:2])@direction/denom,0,1))
        if np.linalg.norm(a[:2]+t*direction-center)<o.radius+margin:return False
    return True


def first_visible_waypoint(path, obstacles):
    for index in range(len(path)-1,0,-1):
        if segment_clear(path[0],path[index],obstacles):return path[index]
    return path[1]


def smooth_route(path, obstacles):
    nodes=[path[0]];index=0
    while index<len(path)-1:
        candidate=len(path)-1
        while candidate>index+1 and not segment_clear(path[index],path[candidate],obstacles):
            candidate-=1
        nodes.append(path[candidate]);index=candidate
    return np.asarray(nodes)


def physical_horizontal_visibility(env, position, radius):
    """Independent ray casts; production sensor range is never changed."""
    import pybullet as p
    directions=env.base_env._range_directions[32:64]
    origin=np.asarray(position)+[0,0,.02]
    offset=env.base_env.RANGE_RAY_ORIGIN_OFFSET
    hits=p.rayTestBatch((origin+directions*offset).tolist(),(origin+directions*radius).tolist(),physicsClientId=env.CLIENT)
    lookup={int(body):i for i,body in enumerate(env.base_env._tree_ids)}
    seen=sorted({lookup[int(hit[0])] for hit in hits if int(hit[0]) in lookup})
    distances=np.array([offset+np.clip(float(hit[2]),0,1)*(radius-offset) for hit in hits])
    return distances,seen


def sensing_time(root):
    saved=load(6000);env=env_for(saved,training_course=False,pool_split='validation')
    pool=env.pool
    paths=[];perception=[]
    try:
        # Restrict all analysis to train + validation geometry, never test.
        for split in ('train','validation'):
            for index,entry in enumerate(pool.entries(split)):
                for stage in range(4):
                    path=pool.path_for(split,index,stage)
                    length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
                    spec=pool.spec(split,index,stage)
                    smooth=smooth_route(path,spec.obstacles)
                    smooth_length=float(np.linalg.norm(np.diff(smooth,axis=0),axis=1).sum())
                    paths.append(dict(split=split,index=index,map_seed=entry['seed'],stage=stage,trees=(10,20,30,40)[stage],
                                      qa_path_length=length,optimistic_seconds=length/.8,
                                      smoothed_path_length=smooth_length,smoothed_constant_speed_seconds=smooth_length/.8,
                                      minimum_mean_speed_for_40s=length/40,
                                      fastest_terminal_discount=.99**(length/.8*15),
                                      terminal_discounted_raw=250*.99**(length/.8*15)))
        # Best-case observation geometry diagnostic, not an Actor/planner benchmark.
        for stage in (0,3):
            env.set_curriculum_stage_override(stage)
            for index in range(20):
                env.reset(seed=index)
                spec=pool.spec('validation',index,stage)
                path=pool.path_for('validation',index,stage)
                for fraction in (.15,.45,.75):
                    node=int(round((len(path)-1)*fraction));position=path[node]
                    env._place_kinematic_state(position,np.zeros(3))
                    analytic,_=horizontal_visibility(position.reshape(1,3),spec.obstacles,1.75)
                    physical,_=physical_horizontal_visibility(env,position,1.75)
                    actual=env.base_env._computeRangeObservation(position)[32:64]*1.75
                    # PyBullet cylinders have a mesh approximation/margin; quantify discrepancy.
                    sensor_error=float(np.max(np.abs(actual-analytic[0])))
                    physical_sensor_error=float(np.max(np.abs(actual-physical)))
                    full=clearance_route(replace(spec,start_pos=tuple(float(v) for v in position)))
                    if full is None:continue
                    full_way=first_visible_waypoint(full,spec.obstacles)
                    full_dir=full_way[:2]-position[:2];full_dir/=max(np.linalg.norm(full_dir),1e-9)
                    for radius in (1.75,3.,5.):
                        beams,seen=physical_horizontal_visibility(env,position,radius)
                        known=tuple(spec.obstacles[i] for i in seen)
                        optimistic=clearance_route(replace(spec,start_pos=tuple(float(v) for v in position),obstacles=known))
                        if optimistic is None:continue
                        next_way=first_visible_waypoint(optimistic,known)
                        local_dir=next_way[:2]-position[:2];local_dir/=max(np.linalg.norm(local_dir),1e-9)
                        angle=float(np.degrees(np.arccos(np.clip(full_dir@local_dir,-1,1))))
                        unsafe=any(not segment_clear(a,b,spec.obstacles) for a,b in zip(optimistic,optimistic[1:]))
                        lp=float(np.linalg.norm(np.diff(optimistic,axis=0),axis=1).sum())
                        fp=float(np.linalg.norm(np.diff(full,axis=0),axis=1).sum())
                        perception.append(dict(stage=stage,map_index=index,fraction=fraction,position=position.tolist(),range=radius,
                                               seen_tree_count=len(known),total_tree_count=len(spec.obstacles),
                                               heading_discrepancy_degrees=angle,optimistic_path_length=lp,full_path_length=fp,
                                               missing_path_length=max(0,fp-lp),optimistic_path_hits_unknown_inflated_geometry=unsafe,
                                               actual_vs_analytic_ray_max_error=sensor_error,
                                               actual_vs_physical_ray_max_error=physical_sensor_error))
                print(f'[M0B] stage={stage} map={index}',flush=True)
        write(root/'M0B_perception_routes.json',perception)
        write(root/'M0E_path_horizon.json',paths)
        write(root/'M0E_discount.json',dict(gamma_per_upper=.99,upper_hz=15,episode_seconds=40,
             effective_geometric_horizon_seconds=(1/(1-.99))/15,
             half_life_seconds=np.log(.5)/np.log(.99)/15,
             timings=[dict(seconds=t,upper_steps=t*15,discount=.99**(t*15),success_bonus_raw=250*.99**(t*15),success_bonus_scaled=25*.99**(t*15)) for t in (1,5,10,20,25,30,40)],
             min_start_band_path=.35,max_L0_path=1.2,original_mix_probabilities=[.2,.2,.3,.5,1.],
             theory='QA/smoothed length divided by speed is an optimistic traversal time for that particular feasible geometric route, not the globally fastest-route lower bound. Startup, turns, terminal slowing, replanning and tracking add time.'))
    finally:
        env.close()


def free_space_env(saved):
    """Isolated empty rectangle: original Full plant/action/reward, no Actor."""
    from types import SimpleNamespace
    from envs.learned_upper_dslpid import KinematicUpperEnv
    class ProbeFullEnv(LearnedUpperDSLPIDForestEnv):
        _place_kinematic_state=KinematicUpperEnv._place_kinematic_state
    args=SimpleNamespace(**saved['protocol']);args.map_id='ego_forest_20x10_seed43_v1'
    def factory(**kwargs):
        for k in tuple(kwargs):
            if k.startswith('kinematic_'):kwargs.pop(k)
        spec=kwargs['map_spec']
        kwargs['map_spec']=replace(spec,obstacles=(),environment_overrides={**spec.environment_overrides,'num_trees':0})
        kwargs['reward_model'].integration_dt=1/120
        return ProbeFullEnv(**kwargs)
    env=make_stage0_env(args,env_class=factory)
    assert len(env.base_env._tree_ids)==0
    return env


def physics(root):
    saved=load(6000);env=free_space_env(saved);traces=[]
    try:
        for label,after in (('stop',[0.,0.,0.]),('turn_left',[0.,1.,0.]),('turn_right',[0.,-1.,0.])):
            env.reset(seed=43);relocate_reset(env,[-4,0,1])
            for _ in range(45):env.step(np.array([[1.,0.,0.]],dtype=np.float32))
            initial=env.pos[0].copy();velocity=env.vel[0].copy();trajectory=[]
            for tick in range(45):
                _,reward,term,trunc,info=env.step(np.asarray(after,dtype=np.float32).reshape(1,3))
                trajectory.append(dict(seconds=(tick+1)/15,position=env.pos[0].tolist(),velocity=env.vel[0].tolist(),
                                       reference_velocity=env.reference_velocity.tolist(),tracking=info['reference_tracking_error_mean']))
                assert not term and not trunc
            stop_ticks=[i for i,v in enumerate(trajectory) if np.linalg.norm(v['velocity'][:2])<.05]
            traces.append(dict(probe=label,initial_position=initial.tolist(),initial_velocity=velocity.tolist(),
                               max_forward_displacement=max(v['position'][0]-initial[0] for v in trajectory),
                               stop_time_s=(stop_ticks[0]+1)/15 if stop_ticks else None,
                               final_lateral_displacement=trajectory[-1]['position'][1]-initial[1],samples=trajectory))
        # Public-observation aliases for omitted timer and PID state; synthetic probes only.
        aliases=[]
        for mode in ('timer','pid_integral'):
            outcomes=[];observations=[]
            for variant in (0,1):
                observation,_=env.reset(seed=43);observation=relocate_reset(env,[0,0,1])
                if mode=='timer' and variant:env.base_env.step_counter=int(39.995*env.PYB_FREQ)
                if mode=='pid_integral' and variant:env.teacher.integral_pos_e[0]=.05
                observations.append(preprocess_state(env._upper_observation(env.base_env._computeObs())))
                next_obs,reward,term,trunc,info=env.step(np.zeros((1,3),dtype=np.float32))
                outcomes.append(dict(next_state=preprocess_state(next_obs).tolist(),reward=reward,terminated=term,truncated=trunc,
                                     reason=info['done_reason'],motor_rpm=env.base_env._last_motor_rpm.tolist()))
            np.testing.assert_array_equal(observations[0],observations[1])
            aliases.append(dict(omitted=mode,public_state_identical=True,next_state_max_difference=float(np.max(np.abs(np.array(outcomes[0]['next_state'])-np.array(outcomes[1]['next_state'])))),outcomes=outcomes,
                                caveat='Injected hidden-state diagnostic; not evidence this exact alias occurs frequently in the visited distribution.'))
        write(root/'M0B_full_braking_turning.json',traces)
        write(root/'M0A_hidden_state_aliases.json',aliases)
        # Fixed reward definition in a bounded 40s hover versus safe navigation probe.
        budgets=[]
        for behavior in ('hover_40s','scheduled_capture','max_speed_flyby'):
            env.reset(seed=43)
            model=env.base_env._reward_model
            components={'progress_reward':0.,'terminal_outcome_reward':0.,**{k:0. for k in COSTS}}
            total=discounted=0.;upper=0;history=[]
            while True:
                if behavior=='hover_40s':command=np.zeros(3)
                elif behavior=='max_speed_flyby':command=np.array([.8,0,0])
                else:
                    d=env.TARGET_POS-env.pos[0];norm=np.linalg.norm(d)
                    speed=.8 if norm>=1.2 else .1+np.clip((norm-.2)/1.0,0,1)*.7
                    command=d/max(norm,1e-9)*speed
                action=command/np.array([.8,.8,.3]);action=np.clip(action,-1,1)
                _,reward,term,trunc,info=env.step(action.astype(np.float32).reshape(1,3))
                total+=reward;discounted+=.99**upper*reward;upper+=1
                for k in components:components[k]+=info.get('reward/'+k,0.)
                history.append(dict(time=env.runtime_lower_step/120,position=env.pos[0].tolist(),velocity=env.vel[0].tolist(),reward=reward))
                if term or trunc:break
                assert upper<610
            budgets.append(dict(behavior=behavior,outcome=info['done_reason'],duration_s=env.runtime_lower_step/120,upper_steps=upper,
                                raw_return=total,discounted_raw_return=discounted,components=components,samples=history))
            print(f'[M0C free-space] {behavior} {info["done_reason"]} return={total:.3f}',flush=True)
        write(root/'M0C_full_40s_reference_budgets.json',budgets)
    finally:env.close()


def report(root):
    guarded=json.loads((root/'protected_files.json').read_text())
    assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in guarded.items())
    write(root/'protection_result.json',dict(unchanged_files=len(guarded),unchanged=True,training_updates=0,test_policy_executions=0,D3_executed=False))
    # Aggregates retain episode identity; avoid treating duplicated 2k/5k actors as independent.
    reward=json.loads((root/'M0C_real_reward_components.json').read_text());aggregates={}
    for group in sorted({v['group'] for v in reward}):
        eps=[v for v in reward if v['group']==group]
        aggregates[group]=dict(episodes=len(eps),mean_raw_return=float(np.mean([e['raw_return'] for e in eps])),
                              mean_discounted_raw_return=float(np.mean([e['discounted_raw_return'] for e in eps])),
                              mean_duration=float(np.mean([e['duration_seconds'] for e in eps])),
                              mean_components={k:float(np.mean([e['reward_components'][k] for e in eps])) for k in eps[0]['reward_components']},
                              mean_regions={region:{k:float(np.mean([e['regions'][region][k] for e in eps])) for k in eps[0]['reward_components']} for region in ('inside_guidance','outside_guidance')},
                              maximum_reconstruction_error=max(e['maximum_lower_reward_error'] for e in eps))
    write(root/'M0C_reward_summary.json',aggregates)
    episodes=json_rows(LOGS/'p0_audit/episodes.jsonl')
    training_components={k:describe([e['reward_terms'][k] for e in episodes]) for k in ('progress_reward','terminal_outcome_reward',*COSTS)}
    for episode in episodes:
        terms=episode['reward_terms']
        total=terms['progress_reward']+terms['terminal_outcome_reward']-sum(terms[k] for k in COSTS)
        np.testing.assert_allclose(total,episode['return'],atol=1e-5,rtol=1e-7)
    for trace in reward:
        if trace['group']!='training_saved_traces' or trace['done_reason']=='p0_budget_boundary':continue
        candidate=min(episodes,key=lambda e:abs(e['return']-trace['raw_return']))
        np.testing.assert_allclose(candidate['return'],trace['raw_return'],atol=1e-5,rtol=1e-7)
        for key in trace['reward_components']:
            np.testing.assert_allclose(candidate['reward_terms'][key],trace['reward_components'][key],atol=1e-5,rtol=1e-7)
    write(root/'M0C_training_episode_summary.json',dict(episodes=len(episodes),
          mean_raw_return=float(np.mean([e['return'] for e in episodes])),components=training_components,
          outcomes={reason:sum(e['done_reason']==reason for e in episodes) for reason in sorted({e['done_reason'] for e in episodes})},
          note='All completed production episodes; 7 saved traces include one incomplete budget cut and are a selected subset.'))
    perception=json.loads((root/'M0B_perception_routes.json').read_text());sensing={}
    for stage in (0,3):
        sensing[str(stage)]={}
        for radius in (1.75,3.,5.):
            subset=[v for v in perception if v['stage']==stage and v['range']==radius]
            sensing[str(stage)][str(radius)]=dict(samples=len(subset),mean_seen_trees=float(np.mean([v['seen_tree_count'] for v in subset])),
                 heading_discrepancy_over_30deg_fraction=float(np.mean([v['heading_discrepancy_degrees']>30 for v in subset])),
                 heading_discrepancy_over_15deg_fraction=float(np.mean([v['heading_discrepancy_degrees']>15 for v in subset])),
                 missing_path_over_1m_fraction=float(np.mean([v['missing_path_length']>1 for v in subset])),
                 optimistic_path_hits_unknown_fraction=float(np.mean([v['optimistic_path_hits_unknown_inflated_geometry'] for v in subset])),
                 maximum_actual_physical_ray_error=max(v['actual_vs_physical_ray_max_error'] for v in subset),
                 maximum_actual_analytic_ray_error=max(v['actual_vs_analytic_ray_max_error'] for v in subset))
    write(root/'M0B_perception_summary.json',sensing)
    arrivals=json.loads((root/'M0D_offline_arrivals.json').read_text());summary={}
    for label in ('validation16m','L0_upper_sampled','training_episode_minima'):
        summary[label]={}
    for step in (2000,5000,6000):
        eps=[v for v in arrivals['validation16m'] if v['group']==f'validation16m_{step}']
        summary['validation16m'][str(step)]={str(radius):sum(v['radius_arrivals'][str(radius)]['reached'] for v in eps) for radius in (.2,.3,.5,1.2)}
    for step,eps in arrivals['L0_upper_sampled'].items():
        summary['L0_upper_sampled'][step]={str(radius):sum(v['radii'][str(radius)]['reached'] for v in eps) for radius in (.2,.3,.5)}
    eps=arrivals['training_episode_minima']
    summary['training_episode_minima']={str(radius):sum(v['minimum']<radius for v in eps) for radius in (.2,.3,.5,1.2)}
    summary['note']='21 full production episodes, 20 validation maps per policy. Training minima are lower-sampled episode aggregates; L0 arrays only upper-sampled. No success radius was changed.'
    write(root/'M0D_arrival_summary.json',summary)
    paths=json.loads((root/'M0E_path_horizon.json').read_text());path_summary={}
    for split in ('train','validation'):
        path_summary[split]={}
        for stage in range(4):
            subset=[v for v in paths if v['split']==split and v['stage']==stage]
            path_summary[split][str(stage)]={k:describe([v[k] for v in subset]) for k in ('qa_path_length','smoothed_path_length','smoothed_constant_speed_seconds','optimistic_seconds','minimum_mean_speed_for_40s','fastest_terminal_discount')}
    write(root/'M0E_path_summary.json',path_summary)
    print(json.dumps(dict(reward=aggregates,perception=sensing,arrivals=summary),indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('part',choices=('state_reward','sensing_time','physics','report'))
    parser.add_argument('--output',type=Path,default=ROOT);args=parser.parse_args()
    if args.part=='state_reward':
        args.output.mkdir(parents=True,exist_ok=False)
        files=list(SOURCE.iterdir())+[Path(p) for p in ('envs/forest/core.py','envs/forest/rewards.py','envs/learned_upper_dslpid.py','envs/preprocess.py','algos/td3/td3_upper_recurrent.py','scripts/change_td3.py','scripts/run_change_td3.py','envs/forest/ego_curriculum_env.py')]
        write(args.output/'protected_files.json',{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file()})
    globals()[args.part](args.output)


if __name__=='__main__':main()
