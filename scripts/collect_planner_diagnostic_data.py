"""Collect privileged trajectories ONLY for A-D observability diagnosis.

This dataset is explicitly ineligible for the nonprivileged Planner BC stage.
Only benchmark train/validation layouts are accessible; failures are retained.
"""

from __future__ import annotations

import argparse
from collections import deque
import contextlib
import io
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import pybullet as p

from benchmarks.forest_benchmark import FAMILIES, load_split, grid_route, install_map
from high_level_planner.coordinate_contract import future_to_local, local_goal, rotate_z, ranges_to_heading_bins
from scripts.evaluate_native_latent_actor import load_networks, write_json, terminal_reason
from envs.ForestAviary import CustomForestAviary
from envs.preprocess import preprocess_state
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from scripts.evaluate_latent_async_pid import make_motor_codec
from scripts.freeze_logical_async import file_record


class DiagnosticGridTeacher:
    def __init__(self, env, record):
        self.env = env
        points = grid_route(record)
        if points is None:
            raise ValueError("Diagnostic map is grid-unreachable")
        # Greedy shortcut only if every segment sample clears the inflated map.
        def safe(first,second):
            length=np.linalg.norm(second-first)
            samples=first+np.linspace(0,1,max(2,int(np.ceil(length/.025))+1))[:,None]*(second-first)
            return all(np.linalg.norm(samples-np.asarray(t["xy"]),axis=1).min()>t["radius"]+.16 for t in record["trees"])
        simplified=[points[0]]
        index=0
        while index<len(points)-1:
            following=len(points)-1
            while following>index+1 and not safe(points[index],points[following]):
                following-=1
            simplified.append(points[following])
            index=following
        self.points=np.column_stack([np.asarray(simplified),np.full(len(simplified),env.TARGET_POS[2])])
        self.lengths=np.linalg.norm(np.diff(self.points,axis=0),axis=1)
        self.cumulative=np.r_[0,np.cumsum(self.lengths)]
        self.distance=0.
        self.speed=0.
        self.codec=make_motor_codec(env)
        self.pid=DSLPIDControl(drone_model=env.DRONE_MODEL)
        self.pid.reset()
        self.yaw=float(record["initial_yaw"])

    def action(self):
        dt=self.env.CTRL_TIMESTEP
        remaining=max(0.,self.cumulative[-1]-self.distance)
        desired=min(.6,np.sqrt(2*1.2*remaining))
        self.speed=max(0.,self.speed+float(np.clip(desired-self.speed,-1.2*dt,1.2*dt)))
        self.distance=min(self.cumulative[-1],self.distance+self.speed*dt)
        j=int(np.clip(np.searchsorted(self.cumulative,self.distance,side="right"),1,len(self.points)-1))
        alpha=(self.distance-self.cumulative[j-1])/max(self.lengths[j-1],1e-9)
        pos=(1-alpha)*self.points[j-1]+alpha*self.points[j]
        vel=(self.points[j]-self.points[j-1])/max(self.lengths[j-1],1e-9)*self.speed
        rpm,_,_=self.pid.computeControl(control_timestep=dt,cur_pos=self.env.pos[0],cur_quat=p.getQuaternionFromEuler(self.env.rpy[0]),cur_vel=self.env.vel[0],cur_ang_vel=self.env.ang_v[0],target_pos=pos,target_vel=vel,target_rpy=np.array([0.,0.,self.yaw]))
        return self.codec.rpm_to_normalized_action(rpm).astype(np.float32),pos,vel


def collect_map(record,encoder,split,episode_id):
    with contextlib.redirect_stdout(io.StringIO()):
        env=CustomForestAviary(obs=ObservationType.KIN,act=ActionType.RPM,pyb_freq=240,ctrl_freq=120,gui=False,curriculum=False,num_trees=0,route_blocking_tree=False,
                              initial_xyzs=np.array([record["start"]]),target_pos=record["goal"],initial_rpys=np.array([[0.,0.,record["initial_yaw"]]]),episode_len_sec=20.)
    try:
        env.reset(seed=record["seed"])
        obs,_=install_map(env,record)
        teacher=DiagnosticGridTeacher(env,record)
        history=deque(maxlen=8)
        previous_velocity=np.zeros(3)
        decisions=[]
        positions=[env.pos[0].copy()]
        velocities=[env.vel[0].copy()]
        clearances=[env._computeNearestTreeClearance(env.pos[0])]
        step=0
        done=False
        with torch.inference_mode():
            while not done:
                state=preprocess_state(obs.reshape(-1))
                if step%8==0:
                    yaw=float(env.rpy[0][2])
                    raw_ranges=state[-8:].copy()
                    if not history:
                        history.extend([raw_ranges.copy() for _ in range(8)])
                    else:
                        history.append(raw_ranges.copy())
                    latent=encoder(torch.from_numpy(state).unsqueeze(0))[0].numpy().copy()
                    goal=local_goal(env.TARGET_POS,env.pos[0],yaw)
                    ranges=ranges_to_heading_bins(raw_ranges,yaw)
                    local_history=ranges_to_heading_bins(np.asarray(history),yaw).reshape(-1)
                    if record["trees"]:
                        closest=min(record["trees"],key=lambda t:np.linalg.norm(env.pos[0][:2]-t["xy"])-t["radius"])
                        vector=np.r_[np.asarray(closest["xy"])-env.pos[0][:2],0.]
                        obstacle=rotate_z(vector,-yaw)[:2]
                    else:
                        obstacle=np.zeros(2)
                    decisions.append({"step":step,"planner_input":np.r_[latent,goal,ranges],"raw_state":state.copy(),
                                      "history":local_history,"position":env.pos[0].copy(),"velocity":rotate_z(env.vel[0],-yaw),
                                      "previous_reference_velocity":rotate_z(previous_velocity,-yaw),"yaw":yaw,"obstacle_local":obstacle})
                action,reference_position,previous_velocity=teacher.action()
                obs,_,terminated,truncated,info=env.step(action.reshape(1,4))
                done=terminated or truncated
                positions.append(env.pos[0].copy())
                velocities.append(env.vel[0].copy())
                clearances.append(env._computeNearestTreeClearance(env.pos[0]))
                step+=1
        positions,velocities,clearances=np.asarray(positions),np.asarray(velocities),np.asarray(clearances)
        samples=[]
        for d in decisions:
            indices=d["step"]+np.array([12,24,36,48,60])
            if indices[-1]>=len(positions):
                continue
            samples.append({"map_id":record["map_id"],"split":split,"episode_id":episode_id,"family":record["family"],
                            "step":d["step"],"planner_input":d["planner_input"],"raw_state":d["raw_state"],"history":d["history"],
                            "position":d["position"],"yaw":d["yaw"],"velocity":d["velocity"],"previous_reference_velocity":d["previous_reference_velocity"],
                            "obstacle_local":d["obstacle_local"],"reference_target":future_to_local(positions[indices],velocities[indices],d["position"],d["yaw"]),
                            "future_minimum_clearance":float(clearances[d["step"]:indices[-1]+1].min()),"success":bool(info["success"]),
                            "termination_reason":terminal_reason(env,info,terminated,truncated)})
        summary={"map_id":record["map_id"],"split":split,"family":record["family"],"samples":len(samples),"steps":step,
                 "success":bool(info["success"]),"collision":bool(info["collision"]),"termination_reason":terminal_reason(env,info,terminated,truncated)}
        print(f"diagnostic {split} {record['family']} #{episode_id}: {summary['termination_reason']}, samples={len(samples)}",flush=True)
        return samples,summary
    finally:
        env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark",default="data/benchmarks/forest_map_isolation_v1")
    parser.add_argument("--output",default="data/planner_diagnostics/map_isolated_v1")
    parser.add_argument("--train-per-family",type=int,default=4)
    parser.add_argument("--validation-per-family",type=int,default=2)
    args=parser.parse_args()
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    encoder,_,weights=load_networks("checkpoints/latent_only_affordance_dim16/model_final")
    samples,summaries,maps=[],[],[]
    for split,count in (("train",args.train_per_family),("validation",args.validation_per_family)):
        candidates=load_split(args.benchmark,split)
        for family in FAMILIES:
            selected=[r for r in candidates if r["family"]==family][:count]
            for record in selected:
                rows,summary=collect_map(record,encoder,split,len(summaries))
                samples.extend(rows)
                summaries.append(summary)
                maps.append(record)
    arrays={key:np.asarray([row[key] for row in samples]) for key in samples[0]}
    np.savez_compressed(root/"samples.npz",**arrays)
    write_json(root/"maps.json",maps)
    write_json(root/"metadata.json",{"purpose":"observability_diagnostic_only", "eligible_for_main_bc":False,
               "teacher":"privileged_global_grid_BFS_shortcuts_plus_DSLPID", "teacher_reason":"Native Actor success gate failed; global teacher is used ONLY for label-conflict/probe diagnosis, as allowed in A-D.",
               "sample_count":len(samples),"episodes":summaries,"encoder":weights["encoder"],"benchmark_manifest":file_record(Path(args.benchmark)/"manifest.json"),
               "coordinate_contract":"planner_local_coordinates_v1", "input_layout":"latent16+unclipped_local_goal3+heading_resampled_ranges8", "failures_retained":True,
               "test_maps_read":False, "samples":file_record(root/"samples.npz")})


if __name__=="__main__":
    main()
