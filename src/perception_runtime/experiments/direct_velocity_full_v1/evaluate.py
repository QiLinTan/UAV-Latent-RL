"""Deterministic fixed validation maps; original and course starts are separate."""
import copy
import numpy as np
import torch
from .task_factory import make_env


def evaluate(agent,config,pool,step,course_level,course_stage,*,complete=False):
    saved=torch.get_rng_state();npstate=np.random.get_state()
    indices=config['evaluation']['complete_indices'] if complete else config['evaluation']['small_indices']
    rows=[]
    try:
        for task,stage,level in [('empty_original',0,4),('stage0_original',0,4),('stage3_original',3,4),
                                  ('current_course_start',course_stage,course_level)]:
            env=make_env(config,training=False,pool=pool,empty=task=='empty_original')
            try:
                for index in ([0] if task=='empty_original' else indices):
                    if task!='empty_original':
                        env.obstacle_curriculum.stage=stage;env.start_curriculum.level=level
                    obs,info=env.reset(seed=index);initial=info['goal_distance'];return_sum=0.;min_distance=initial
                    actions=[];speed_errors=[];zmin=zmax=float(env.base_env.pos[0,2]);ticks=0
                    while True:
                        action=agent.action(obs);actions.append(action.tolist())
                        obs,r,t,u,info=env.step(action);return_sum+=r;min_distance=min(min_distance,info['goal_distance'])
                        for tick in env.last_ticks:
                            speed_errors.append(np.linalg.norm(np.asarray(tick['velocity'])-tick['command_world']))
                            zmin=min(zmin,tick['position'][2]);zmax=max(zmax,tick['position'][2]);ticks+=1
                        if t or u:break
                    rows.append(dict(task=task,map_index=index,stage=stage,start_level=level,
                        start_source=info.get('start_curriculum_source','original'),map_id=info['map_id'],
                        geometry_sha256=info.get('geometry_sha256'),initial_distance=initial,minimum_distance=min_distance,
                        final_distance=info['goal_distance'],success=info['success'],collision=info['collision'],
                        done_reason=info['done_reason'],return_raw=return_sum,lower_ticks=ticks,
                        action_mean=np.mean(actions,axis=0).tolist(),action_std=np.std(actions,axis=0).tolist(),
                        velocity_error_rmse=float(np.sqrt(np.mean(np.square(speed_errors)))),height_min=zmin,height_max=zmax))
            finally:env.close()
    finally:torch.set_rng_state(saved);np.random.set_state(npstate)
    summary={task:dict(episodes=len(group),success_rate=float(np.mean([r['success'] for r in group])),
        collision_rate=float(np.mean([r['collision'] for r in group])),mean_final_distance=float(np.mean([r['final_distance'] for r in group])))
        for task in sorted(set(r['task'] for r in rows)) if (group:=[r for r in rows if r['task']==task])}
    return dict(step=step,complete=complete,deterministic=True,split='validation',episodes=rows,summary=summary)
