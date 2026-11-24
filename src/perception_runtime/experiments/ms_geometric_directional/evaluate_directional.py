"""Fixed Stage0–3 comparison plus physical left/right aperture policy probes."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from envs.map_atlas.forest_pool import ForestPool
from envs.map_atlas.spec import MapSpec,StaticObstacleSpec
from experiments.direct_velocity_full_v1.contracts import normalized_action_to_world,yaw_from_quaternion
from experiments.direct_velocity_full_v1.task_factory import ROOT,load_config
from experiments.rv1_range_cnn.environment import RangeImageEnv,make_course_env
from directional_agent import Agent
from directional_features import DIMS,ObservationEncoder
from train_directional import sha

MODES=("original","directional")
APERTURES=[
    (distance,offset,width) for distance in (1.65,2.45,3.35)
    for offset,width in ((.42,.44),(.42,.68),(.58,.44),(.58,.68))
]


def load_agent(run,mode,step,config):
    path=run/mode/f"checkpoint_{step:06d}.pt"
    relevant=[json.loads(s) for s in (run/mode/"events.jsonl").read_text().splitlines()]
    expected=[event["sha256"] for event in relevant if event["event"]=="checkpoint" and event["step"]==step]
    actual=sha(path)
    assert expected==[actual]
    pack=torch.load(path,map_location="cpu",weights_only=False)
    assert pack["contract"]["mode"]==mode and pack["contract"]["dimension"]==DIMS[mode]
    assert pack["trainer"]["step"]==step and not pack["contract"]["short_check"]
    agent=Agent(config["td3"],DIMS[mode],"cpu")
    agent.actor.load_state_dict(pack["agent"]["actor"])
    agent.actor.eval()
    return agent,actual,pack["contract"]["pool_fingerprint"]


def initial_info(env,info):
    b=env.base_env
    return dict(physical_state20=np.asarray(b._getDroneStateVector(0)).tolist(),target=b.TARGET_POS.tolist(),
                geometry_sha256=info.get("geometry_sha256"),map_pool_index=info.get("map_pool_index"))


def follow(env,agent,encoder,obs,info,wall=None):
    b=env.base_env
    position=b.pos[0].copy()
    collision=bool(info["collision"])
    path=duration=0.
    min_clear=float(info["min_tree_clearance"])
    traj=[]
    goal_in=float(np.linalg.norm(b.TARGET_POS-b.pos[0]))
    reason="running"
    for idx in range(2000):
        features=encoder(obs)
        action=agent.action(features)
        command=normalized_action_to_world(action,yaw_from_quaternion(b.quat[0]))
        p0=b.pos[0].copy()
        obs,reward,terminated,truncated,info=env.step(action)
        reason=info["done_reason"]
        collision|=bool(info["collision"])
        min_clear=min(min_clear,float(info["min_tree_clearance"]))
        for tick in env.last_ticks:
            new=np.asarray(tick["position"])
            path+=float(np.linalg.norm(new-position))
            position=new
            duration+=1/120
        if wall is not None:
            traj.append(dict(step=idx,position=p0.tolist(),next_position=b.pos[0].tolist(),velocity=b.vel[0].tolist(),
                             command=command.tolist(),actual_command=info["accepted_world_command"].tolist(),
                             goal_distance=float(info["goal_distance"]),front_distance=float(wall-.09-p0[0]),
                             clearance=float(info["min_tree_clearance"]),collision=bool(info["collision"]),reason=reason))
        if terminated or truncated:
            if reason=="running":
                raise RuntimeError("terminal has no reason")
            break
    if reason=="running":
        raise RuntimeError("evaluation exceeded harness bound")
    row=dict(reason=reason,success=reason=="success",collision_observed=collision,upper_steps=idx+1,
             goal_progress_m=goal_in-float(info["goal_distance"]),final_goal_distance_m=float(info["goal_distance"]),
             min_clearance_m=min_clear if np.isfinite(min_clear) else None,duration_s=duration,path_length_m=path)
    return row,traj


def paired_map(case,side):
    wall,offset,width=APERTURES[case]
    center=side*offset
    intervals=((-1.45,center-width/2),(center+width/2,1.45))
    panels=tuple(StaticObstacleSpec(kind="box",name=f"wall{index}",
                                   position=(wall,(a+b)/2,1.4),half_extents=(.09,(b-a)/2,1.4))
                 for index,(a,b) in enumerate(intervals))
    return MapSpec(map_id=f"direction_probe_{case}_{side}",source="paired diagnostic",
                   scenario="left/right aperture",split="probe",start_pos=(0.,0.,1.),
                   target_pos=(4.35,0.,1.),xy_half_extent=5.,obstacle_sensor_range=8.,
                   obstacles=panels,episode_len_sec=30.)


def first_reference_turn(traj,side):
    # Three consecutive 15 Hz commands toward the aperture. Record first
    # command of that window, as well as whether the wall was still outside ±1 m.
    for i in range(len(traj)-2):
        if all(side*t["command"][1]>=.08 for t in traj[i:i+3]):
            return traj[i]["front_distance"]
    return None


def through(traj,wall,center,width):
    for t in traj:
        if t["position"][0]<wall+.09<=t["next_position"][0]:
            y=t["next_position"][1]
            z=t["next_position"][2]
            return bool(abs(y-center)<=width/2-.06 and .08<z<2.75)
    return False


def aperture_job(run_mode,agent,encoder,indices,trace_dir):
    out=[]
    for case in indices:
        wall,offset,width=APERTURES[case]
        starts={}
        for side in (-1,1):
            env=RangeImageEnv(map_spec=paired_map(case,side))
            try:
                obs,info=env.reset(seed=1000+case)
                init=initial_info(env,info)
                pair=np.asarray(encoder(obs))
                starts[side]=pair
                act=agent.action(pair)
                command=normalized_action_to_world(act,yaw_from_quaternion(env.base_env.quat[0]))
                row,traj=follow(env,agent,encoder,obs,info,wall)
                lead=first_reference_turn(traj,side)
                row.update(mode=run_mode,case=case,side=side,wall_center_m=wall,aperture_offset_m=offset,
                           aperture_width_m=width,init=init,initial_command=command.tolist(),
                           first_sustained_turn_lead_m=lead,early_turn=lead is not None and lead>1,
                           through_opening=through(traj,wall,side*offset,width))
                (trace_dir/f"{run_mode}_{case:02d}_{side:+d}.json").write_text(json.dumps(dict(row=row,trajectory=traj)))
                out.append(row)
            finally:
                env.close()
        near_equal=np.array_equal(starts[-1][:268],starts[1][:268])
        far_norm=float(np.linalg.norm(starts[-1][268:]-starts[1][268:]))
        for r in out[-2:]:
            r.update(pair_near_equal=bool(near_equal),pair_far_l2=far_norm,
                     matching_state20=out[-2]["init"]["physical_state20"]==out[-1]["init"]["physical_state20"])
    return out


def execute(kind,mode,stage,indices,run,results,step):
    torch.set_num_threads(1)
    config=load_config()
    agent,checksum,fingerprint=load_agent(run,mode,step,config)
    encoder=ObservationEncoder(mode)
    if kind=="aperture":
        rows=aperture_job(mode,agent,encoder,indices,results/"aperture_traces")
    else:
        pool=ForestPool(ROOT/config["forest_pool_path"])
        assert pool.fingerprint==fingerprint
        env=make_course_env(pool,training=False)
        rows=[]
        try:
            env.obstacle_curriculum.stage=stage
            for case in indices:
                obs,info=env.reset(seed=case)
                if info["density_stage"]!=stage or info["start_curriculum_source"]!="original":
                    raise RuntimeError("fixed Stage0–3 contract changed")
                init=initial_info(env,info)
                row,_=follow(env,agent,encoder,obs,info)
                row.update(mode=mode,stage=stage,case=case,init=init)
                rows.append(row)
        finally:
            env.close()
    return dict(kind=kind,mode=mode,stage=stage,checkpoint_sha256=checksum,rows=rows)


def consolidate(batches):
    forest=[r for batch in batches if batch["kind"]=="forest" for r in batch["rows"]]
    probe=[r for batch in batches if batch["kind"]=="aperture" for r in batch["rows"]]
    forest.sort(key=lambda r:(r["stage"],r["case"],r["mode"]))
    probe.sort(key=lambda r:(r["mode"],r["case"],r["side"]))
    for a,b in zip(forest[::2],forest[1::2]):
        assert (a["stage"],a["case"],a["init"])==(b["stage"],b["case"],b["init"])
    paired=[]
    for left,right in zip(probe[::2],probe[1::2]):
        assert left["case"]==right["case"] and left["mode"]==right["mode"]
        assert left["side"]==-1 and right["side"]==1
        assert left["matching_state20"] and right["matching_state20"]
        negative_y=left["initial_command"][1]
        positive_y=right["initial_command"][1]
        paired.append(dict(mode=left["mode"],case=left["case"],near_equal=left["pair_near_equal"],
                           far_l2=left["pair_far_l2"],delta_y_mps=positive_y-negative_y,
                           correct_signed_pair=negative_y<=-.04 and positive_y>=.04,
                           correct_contrast=positive_y-negative_y>=.08))
    summary={}
    for mode in MODES:
        group=[r for r in forest if r["mode"]==mode]
        apertures=[r for r in probe if r["mode"]==mode]
        pairs=[r for r in paired if r["mode"]==mode]
        eligible=[r for r in pairs if r["near_equal"] and r["far_l2"]>1e-7]
        summary[mode]=dict(
            forest={str(i):dict(n=sum(r["stage"]==i for r in group),
                                reason=dict(Counter(r["reason"] for r in group if r["stage"]==i)),
                                goal_distance_median=float(np.median([r["final_goal_distance_m"] for r in group if r["stage"]==i])))
                    for i in sorted({r["stage"] for r in group})},
            probe=dict(pairs=len(pairs),eligible_only_far_varies=len(eligible),
                       median_delta_y_mps=float(np.median([r["delta_y_mps"] for r in eligible])) if eligible else None,
                       contrasting_pairs=sum(r["correct_contrast"] for r in eligible),
                       correct_signed_pairs=sum(r["correct_signed_pair"] for r in eligible),
                       closed_loop_cases=len(apertures),
                       reasons=dict(Counter(r["reason"] for r in apertures)),
                       passed_gate=sum(r["through_opening"] for r in apertures),
                       early_turn=sum(r["early_turn"] for r in apertures)))
    return dict(summary=summary,forest=forest,probe=probe,actor_paired=paired)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--step",type=int,choices=[20000,40000],required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=12)
    args=parser.parse_args()
    run=args.run.resolve()
    results=args.output.resolve()
    results.mkdir(exist_ok=False,parents=True)
    (results/"aperture_traces").mkdir()
    (results/"protocol.json").write_text(json.dumps(dict(
        step=args.step,forest_stages=list(range(4)),validation_case_indices=list(range(20)),original_starts=True,
        aperture_cases=APERTURES,action_contrast_threshold_mps=.08,per_side_threshold_mps=.04,
        sustained_turn_threshold_mps=.08,sustained_turn_ticks=3,cpu_threads_per_process=1),indent=2))
    jobs=[("forest",m,s,list(range(i,i+5))) for m in MODES for s in range(4) for i in range(0,20,5)]
    jobs += [("aperture",m,-1,list(range(i,i+3))) for m in MODES for i in range(0,len(APERTURES),3)]
    started=time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(execute,*j,run,results,args.step):j for j in jobs}
        reports=[]
        for f in as_completed(futures):
            case=futures[f]
            try:
                reports.append(f.result())
            except BaseException:
                (results/"error.json").write_text(json.dumps(dict(job=case,traceback=traceback.format_exc()),indent=2))
                raise
            status=dict(completed=len(reports),total=len(jobs),last=case,elapsed_s=time.time()-started)
            (results/"status.json").write_text(json.dumps(status,indent=2))
            print(json.dumps(status),flush=True)
    result=consolidate(reports)
    result["checkpoint_sha256"]={m:next(b["checkpoint_sha256"] for b in reports if b["mode"]==m) for m in MODES}
    assert len(result["forest"])==160 and len(result["probe"])==48
    (results/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result["summary"],indent=2),flush=True)


if __name__=="__main__":
    main()
