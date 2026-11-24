"""Frozen dv0 evaluation: predeclared validation maps, original starts, no updates."""
import os,sys,json,hashlib,contextlib,multiprocessing,concurrent.futures,time
from pathlib import Path
import numpy as np
import torch
import pybullet as p
from algos.td3.networks import Actor
from envs.map_atlas.forest_pool import ForestPool
from envs.map_atlas.navigation3d import signed_clearance
from experiments.direct_velocity_full_v1.task_factory import load_config,data_contract,ROOT,EXP
from experiments.direct_velocity_full_v1.rl.checkpoint import validate
from experiments.direct_velocity_full_v1.rl.td3_adapter import project
from experiments.direct_velocity_full_v1.envs.direct_velocity_full import DirectVelocityFullEnv
from experiments.direct_velocity_full_v1.contracts import ControlSpec
from .environment import plant_kwargs
from .benchmark import spec_from,digest,geometric_fingerprint
from .lidar import Lidar,LAYOUTS

CKPT=EXP/'outputs/dv0_velocity_full/dv0_seed43_20261026_night_v1/checkpoints/training_state_final_step240000.pt'
EXPECTED_SHA='866d718a67fe41c88b1a718e5b643856aac49f6f54706a1a3a4d4a32acb28767'
_ACTOR=None
def get_actor():
    global _ACTOR
    if _ACTOR is None:
        torch.set_num_threads(1);c=load_config();pool=ForestPool(ROOT/c['forest_pool_path'])
        if hashlib.sha256(CKPT.read_bytes()).hexdigest()!=EXPECTED_SHA:raise RuntimeError('frozen checkpoint changed')
        ckpt=torch.load(CKPT,map_location='cpu',weights_only=False);validate(ckpt,data_contract(c,pool))
        _ACTOR=Actor(114,3,1.);_ACTOR.load_state_dict(ckpt['agent']['actor']);_ACTOR.eval()
        for param in _ACTOR.parameters():param.requires_grad_(False)
    return _ACTOR

def episode(job):
    data,out=job;out=Path(out);name=data['map_id'];spec=spec_from(data);rows=[];ticks=[];comparison=[]
    with (out/'challenge_episodes'/f'{name}.log').open('w') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
        actor=get_actor();c=load_config();env=DirectVelocityFullEnv(control_spec=ControlSpec(**c['control_spec']),plant_kwargs=plant_kwargs(spec))
        try:
            obs,_=env.reset(seed=data['benchmark']['seed']);b=env.base_env;initial=b.pos[0].copy()
            if not np.allclose(initial,spec.start_pos):raise RuntimeError('not original task start')
            sensors={name:Lidar(cfg) for name,cfg in LAYOUTS.items()};total=0;wall=time.monotonic()
            for decision in range(2000):
                physical=b._getDroneStateVector(0).copy()
                with torch.inference_mode():raw=actor(torch.from_numpy(obs.reshape(1,114))).numpy().reshape(3)
                canonical=project(torch.from_numpy(raw)).numpy()
                if decision%15==0:
                    snapshot={}
                    for lname,sensor in sensors.items():
                        scan=sensor.scan(client_id=b.CLIENT,position=b.pos[0],quaternion_xyzw=b.quat[0],timestamp_s=b.physics_ticks/240,
                            self_ids=tuple(map(int,b.DRONE_IDS)),ground_ids=(int(b.PLANE_ID),),obstacle_ids=tuple(b._tree_ids))
                        snapshot[lname]=scan
                        comparison.append(dict(map_id=name,decision=decision,layout=lname,hit_rays=int(scan.hit_mask.sum()),
                            obstacle_rays=int((scan.hit_type==3).sum()),ground_rays=int((scan.hit_type==2).sum()),
                            detected_obstacle_ids=sorted(set(map(int,scan.hit_body[scan.hit_type==3]))),
                            minimum_hit_range_m=float(scan.range_m[scan.hit_mask].min()) if scan.hit_mask.any() else None))
                    # Verify Actor against its OWN frozen float32 ray geometry.
                    # The new double-precision angular grid is mathematically
                    # equivalent, but grazing casts can change discontinuously.
                    legacy=b._computeRangeObservation(physical[:3]).reshape(96)
                    parity=float(np.max(abs(legacy-obs.reshape(-1)[18:])))
                    if parity>2e-5:raise RuntimeError('frozen legacy recast disagrees with Actor input')
                    starts,ends,offset,length=b._rangeRayGeometry(physical[:3])
                    hits=p.rayTestBatch(starts.tolist(),ends.tolist(),physicsClientId=b.CLIENT)
                    body_ids=np.array([h[0] for h in hits]);obstacle=np.isin(body_ids,b._tree_ids)
                    comparison.append(dict(map_id=name,decision=decision,layout='legacy96_actor',hit_rays=int((body_ids>=0).sum()),
                        obstacle_rays=int(obstacle.sum()),ground_rays=int((body_ids==b.PLANE_ID).sum()),
                        detected_obstacle_ids=sorted(set(map(int,body_ids[obstacle]))),
                        minimum_hit_range_m=float(legacy[body_ids>=0].min()*8) if (body_ids>=0).any() else None,
                        legacy_actor_recast_error=parity,new_continuous96_vs_legacy_absmax=float(np.max(abs(snapshot['96_3x32'].range_m.reshape(-1)/8-legacy)))))
                new,r,t,u,info=env.step(canonical);total+=r
                rows.append(dict(observation=obs.reshape(114),raw_actor=raw,projected_action=canonical,
                    state_before=physical,accepted_world=info['accepted_world_command'],reward=r,lower_steps=info['lower_steps_executed']))
                ticks.extend(env.last_ticks);obs=new
                if t or u:break
            else:raise RuntimeError('episode exceeded strict task timeout')
            def arrays(records,prefix):return {prefix+k:np.asarray([x[k] for x in records]) for k in records[0]}
            trace={**arrays(rows,'upper_'),**arrays(ticks,'lower_'), 'start':initial,'goal':b.TARGET_POS.copy()}
            path=np.vstack((initial,trace['lower_position']));clear=signed_clearance(spec,trace['lower_position'])-float(np.hypot(b.COLLISION_R,b.COLLISION_H/2))
            duration=len(ticks)/120.;err=trace['lower_velocity']-trace['lower_command_world'];done=info['done_reason']
            result=dict(map_id=name,group=spec.scenario,variant=data['benchmark']['variant'],split=spec.split,geometry_sha256=data['geometry_sha256'],
                checkpoint_sha256=EXPECTED_SHA,original_start=True,deterministic=True,success=done=='success',done_reason=done,
                final_distance_m=float(info['goal_distance']),duration_s=duration,completion_time_s=duration if done=='success' else None,
                path_length_m=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()),minimum_geometric_clearance_m=float(clear.min()) if np.isfinite(clear.min()) else None,
                geometric_clearance_definition='exact primitive signed distance minus enclosing drone sphere, excludes ground/bounds; QA only',
                velocity_rmse_mps=float(np.sqrt(np.mean(np.sum(err*err,axis=1)))),
                planar_action_boundary_fraction=float(np.mean(np.linalg.norm(trace['upper_projected_action'][:,:2],axis=1)>=.99)),
                roll_pitch_peak_rad=float(abs(trace['lower_rpy'][:,:2]).max()),height_min_m=float(path[:,2].min()),height_max_m=float(path[:,2].max()),
                upper_steps=len(rows),lower_ticks=len(ticks),total_reward_raw=float(total),wall_seconds=time.monotonic()-wall,
                drone_radius_m=float(b.COLLISION_R),drone_height_m=float(b.COLLISION_H),physical_max_rpm=float(b.MAX_RPM))
            np.savez_compressed(out/'challenge_episodes'/f'{name}.npz',**trace)
            (out/'challenge_episodes'/f'{name}.json').write_text(json.dumps(result,indent=2))
            (out/'challenge_episodes'/f'{name}_sensors.json').write_text(json.dumps(comparison,indent=2))
            return result
        finally:env.close()

def main(out):
    parent=Path(out);gate=json.loads((parent/'geometry_gate.json').read_text())
    if gate['status']!='PASS_GEOMETRY_COST':raise RuntimeError('geometry gate failed; frozen evaluation held')
    out=parent/'challenge_eval_v2';out.mkdir(exist_ok=False);(out/'challenge_episodes').mkdir()
    catalog=json.loads((parent/'challenge_catalog.json').read_text());maps=[x for x in catalog['maps'] if x['split']=='validation']
    empty=dict(maps[0]);empty.update(map_id='rv1_empty_matched_diagnostic',scenario='empty',split='diagnostic',obstacles=[],
        benchmark=dict(group='empty',variant='empty_matched',seed=43));empty['geometry_sha256']=geometric_fingerprint(empty)
    protocol=dict(schema='rv1_frozen_dv0_challenge_eval_v1',checkpoint=str(CKPT),checkpoint_sha256=EXPECTED_SHA,
        maps=[x['map_id'] for x in maps],map_catalog_fingerprint=catalog['fingerprint'],deterministic=True,original_start=True,
        test_not_evaluated=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='Control/reward/state/action unchanged; new geometry and 40s scenario timeout. No CNN policy/navigation comparison.')
    (out/'challenge_evaluation_protocol.json').write_text(json.dumps(protocol,indent=2))
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(episode,[(data,str(out)) for data in [*maps,empty]]):
            if result['split']=='diagnostic':(out/'empty_matched_result.json').write_text(json.dumps(result,indent=2));continue
            results.append(result);(out/'challenge_results.json').write_text(json.dumps(results,indent=2))
            print(json.dumps(dict(completed=len(results),expected=len(maps),last=result['map_id'],done=result['done_reason'])),flush=True)
    groups=[]
    for group in sorted(set(x['group'] for x in results)):
        rows=[x for x in results if x['group']==group];reasons={d:sum(r['done_reason']==d for r in rows) for d in sorted(set(r['done_reason'] for r in rows))}
        groups.append(dict(group=group,episodes=len(rows),success=sum(x['success'] for x in rows),done_reasons=reasons,
            mean_final_distance_m=float(np.mean([x['final_distance_m'] for x in rows])),mean_path_m=float(np.mean([x['path_length_m'] for x in rows])),
            mean_duration_s=float(np.mean([x['duration_s'] for x in rows])),minimum_clearance_m=min(x['minimum_geometric_clearance_m'] for x in rows)))
    summary=dict(status='COMPLETE',episodes=len(results),success=sum(x['success'] for x in results),groups=groups,
        checkpoint_sha256=EXPECTED_SHA,checkpoint_after_sha256=hashlib.sha256(CKPT.read_bytes()).hexdigest(),no_training=True,
        catalog_fingerprint=catalog['fingerprint'],all_prespecified_validation_maps=True)
    (out/'challenge_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main(sys.argv[1])
