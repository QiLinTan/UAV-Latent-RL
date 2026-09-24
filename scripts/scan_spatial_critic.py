"""No-training 3-D Q scan and one-action-then-actor return calibration."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from envs.preprocess import preprocess_state
from scripts import change_td3
from scripts.diagnose_terminal_reward_rollouts import PRESET, preset_args
from scripts.run_change_td3 import build_mode_cmd
from trainers.spatial_q_diagnostics import spatial_candidates, calibration_indices, q_summary


def dump(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")


def scan(callback, trainer, args, cli):
    prior = (json.loads((cli.reference_dir/"summary.json").read_text())
             if cli.reference_dir is not None else None)
    eval_references = None
    if cli.reference_eval is not None:
        rows = [json.loads(line) for line in cli.reference_eval.read_text().splitlines()
                if line.strip()]
        eval_references = rows[-1]["episodes"]
    actor_hash = hashlib.sha256(Path(cli.checkpoint+"_actor").read_bytes()).hexdigest()
    if prior is not None:
        assert actor_hash == prior["metadata"]["actor_sha256"]
    output=cli.output_dir
    output.mkdir(parents=True,exist_ok=False)
    env=callback.env_cls(**callback.env_kwargs,gui=False)
    if bool(getattr(env,"CURRICULUM_ENABLED",False)):
        env.set_curriculum_stage_override(trainer.env.get_curriculum_stage())
    records, calibrations, samples, observations = [],[],[],{}
    synthetic=[]
    for norm in (0.,.1,.3,.5,.75,1.):
        action=np.array([norm,0,0],dtype=np.float32)
        command=env._velocity_command(action)
        assert abs(float(np.linalg.norm(command[:2]))-norm*env.max_reference_speed)<1e-6
        synthetic.append(dict(norm=norm,command=command.tolist()))
    try:
        for seed in range(cli.seeds):
            obs,_=env.reset(seed=seed)
            actions, states=[],[]
            done=False
            while not done:
                position,velocity=env.pos[0].copy(),env.vel[0].copy()
                states.append(dict(obs=obs.copy(),position=position,velocity=velocity))
                action=trainer.agent.select_action(preprocess_state(obs.reshape(-1)))
                actions.append(action.copy())
                command=env._velocity_command(action)
                expected_velocity=env._next_kinematic_velocity(velocity,command)
                obs,reward,terminated,truncated,info=env.step(action)
                done=terminated or truncated
                np.testing.assert_allclose(env.vel[0],expected_velocity,atol=2e-6)
                error=env.TARGET_POS-position
                distance=float(np.linalg.norm(error))
                mapped_error=abs(float(np.linalg.norm(command[:2]))-
                                 min(float(np.linalg.norm(action[:2])),1.)*env.max_reference_speed)
                assert mapped_error < 2e-6
                samples.append(dict(seed=seed,step=len(actions)-1,position=position.tolist(),
                    goal_distance=distance,xy_distance=float(np.linalg.norm(error[:2])),
                    z_error=float(error[2]),actor_action=action.tolist(),command=command.tolist(),
                    actual_velocity=env.vel[0].tolist(),mapping_error=mapped_error,
                    command_closing_speed=float(command@error/max(distance,1e-9))))
            if prior is not None:
                ref=next(r for r in prior["policy_episodes"] if r["seed"]==seed)
            elif eval_references is not None:
                eval_ref=next(r for r in eval_references if r["episode"]==seed)
                ref=dict(outcome=eval_ref["done_reason"],steps=eval_ref["length"],
                         final_distance=eval_ref["final_goal_distance"])
            else:
                ref=dict(outcome=info["done_reason"],steps=len(actions),
                         final_distance=float(info["goal_distance"]))
            assert info["done_reason"]==ref["outcome"] and len(actions)==ref["steps"]
            assert abs(float(info["goal_distance"])-ref["final_distance"])<1e-4, (
                seed, float(info["goal_distance"]), ref["final_distance"])
            if prior is not None:
                episode_cases=[c for c in prior["cases"] if c["seed"]==seed]
            else:
                distances=[float(np.linalg.norm(env.TARGET_POS-s["position"])) for s in states]
                xy_distances=[float(np.linalg.norm(env.TARGET_POS[:2]-s["position"][:2]))
                              for s in states]
                indices={0:["reset"]}
                for index,label in ((45,"3_seconds"),(90,"6_seconds"),
                                    (int(np.argmin(distances)),"closest_preterminal")):
                    if index<len(states):
                        indices.setdefault(index,[]).append(label)
                for threshold,label,values in (
                    (1.2,"first_xy_within_1.2m",xy_distances),
                    (1.2,"first_guidance_entry",distances),
                    (.8,"first_3d_within_0.8m",distances),
                    (.5,"first_3d_within_0.5m",distances),
                ):
                    index=next((i for i,value in enumerate(values) if value<threshold),None)
                    if index is not None:
                        indices.setdefault(index,[]).append(label)
                episode_cases=[]
                for index,labels in sorted(indices.items()):
                    state=states[index]
                    episode_cases.append(dict(id=f"seed{seed:02d}_step{index:03d}",seed=seed,
                        step=index,labels=labels,position=state["position"].tolist(),
                        velocity=state["velocity"].tolist(),elapsed=index*env.CTRL_TIMESTEP,
                        remaining_seconds=max(0.,env.EPISODE_LEN_SEC-index*env.CTRL_TIMESTEP),
                        d_xy=xy_distances[index],
                        z_error=abs(float(env.TARGET_POS[2]-state["position"][2])),
                        d_3d=distances[index]))
            # Calibrate the approach, guidance entry, and final capture region.
            six_second=next((c for c in episode_cases if "6_seconds" in c["labels"]),None)
            guidance=next((c for c in episode_cases if "first_guidance_entry" in c["labels"]),None)
            capture=next((c for c in episode_cases if "first_3d_within_0.5m" in c["labels"]),None)
            closest=next(c for c in episode_cases if "closest_preterminal" in c["labels"])
            calibration_steps={c["step"] for c in (six_second,guidance,capture or closest)
                               if c is not None}
            for case in episode_cases:
                index=case["step"]
                state=states[index]
                np.testing.assert_allclose(state["position"],case["position"],atol=1e-5)
                normalized=preprocess_state(state["obs"].reshape(-1))
                observations[case["id"]]=normalized
                candidates,metadata=spatial_candidates(actions[index],state["position"],env.TARGET_POS,
                    xy_limit=env.max_reference_speed,z_limit=env.max_reference_vertical_speed)
                q1,q2=trainer.agent.evaluate_critic_actions(normalized,candidates)
                assert np.all(np.isfinite(q1)) and np.all(np.isfinite(q2))
                clearance=float(env.base_env._computeNearestTreeClearance(state["position"]))
                candidate_rows=[dict(**m,action=a.tolist(),q1=float(v1),q2=float(v2),q_min=float(min(v1,v2)))
                                for m,a,v1,v2 in zip(metadata,candidates,q1,q2)]
                record=dict(case=case,clearance=clearance if np.isfinite(clearance) else None,
                    clearance_above_0_35=bool(clearance>.35),
                    category=("xy_near_height_mismatch" if case["d_xy"]<1.2 and case["d_3d"]>=1.2
                              else "guidance" if case["d_3d"]<1.2 else "outside"),
                    candidates=candidate_rows)
                records.append(record)
                if index not in calibration_steps:
                    continue
                for candidate_index in calibration_indices(q1,q2,metadata):
                    obs,_=env.reset(seed=seed)
                    for action in actions[:index]:
                        obs,_,terminated,truncated,_=env.step(action)
                        assert not (terminated or truncated)
                    np.testing.assert_allclose(obs,state["obs"],atol=1e-5)
                    np.testing.assert_allclose(env.vel[0],state["velocity"],atol=1e-5)
                    assert env.runtime_lower_step==index*args.high_level_interval
                    trace=[]
                    for step in range(200):
                        action=(candidates[candidate_index] if step==0 else
                                trainer.agent.select_action(preprocess_state(obs.reshape(-1))))
                        obs,reward,terminated,truncated,info=env.step(action)
                        trace.append(dict(reward=float(reward),distance=float(info["goal_distance"]),
                                          position=env.pos[0].tolist(),action=np.asarray(action).tolist()))
                        if terminated or truncated:
                            break
                    else:
                        raise AssertionError("No natural termination")
                    actual_return=sum(args.reward_scale*args.gamma**k*t["reward"] for k,t in enumerate(trace))
                    row=dict(case_id=case["id"],seed=seed,step=index,category=record["category"],
                        clearance=record["clearance"],**candidate_rows[candidate_index],
                        actual_return=float(actual_return),outcome=info["done_reason"],steps=len(trace),
                        final_distance=float(info["goal_distance"]),minimum_distance=min(t["distance"] for t in trace),
                        immediate_scaled_reward=args.reward_scale*trace[0]["reward"],trace=trace)
                    if candidate_index==0:
                        assert row["outcome"]==ref["outcome"] and row["steps"]==len(actions)-index
                        assert abs(row["final_distance"]-ref["final_distance"])<1e-4
                    calibrations.append(row)
            # Flush per-seed results so a partial run remains inspectable.
            dump(output/"q_scan.json",records)
            with (output/"calibration.jsonl").open("w") as stream:
                for row in calibrations:
                    stream.write(json.dumps(row,allow_nan=False)+"\n")
            print(f"Q scan seed {seed}: {len(records)} states, {len(calibrations)} calibrated continuations",flush=True)
    finally:
        env.close()
    np.savez_compressed(output/"observations.npz",**observations)
    with (output/"actor_actions.jsonl").open("w") as stream:
        for row in samples:
            stream.write(json.dumps(row,allow_nan=False)+"\n")
    summary=dict(metadata=dict(checkpoint=cli.checkpoint,actor_sha256=actor_hash,
        critic_sha256=hashlib.sha256(Path(cli.checkpoint+"_critic").read_bytes()).hexdigest(),
        preset=cli.preset,reference_dir=(str(cli.reference_dir) if cli.reference_dir else None),
        reference_eval=(str(cli.reference_eval) if cli.reference_eval else None),
        seeds=list(range(cli.seeds)),gamma=args.gamma,
        reward_scale=args.reward_scale,actor_objective="Q1",scan_count_per_state=len(candidates),
        calibration="One candidate action, then current deterministic actor until original natural termination.",
        caveat="Finite remaining horizon and current actor; target smoothing, target-network lag and hidden clock prevent interpreting every Q-minus-return residual as a pure implementation error."),
        all_states=q_summary(records),
        by_category={key:q_summary([r for r in records if r["category"]==key])
                     for key in ("outside","xy_near_height_mismatch","guidance")},
        clear_states=q_summary([r for r in records if r["clearance_above_0_35"]]),
        calibration_state_count=len(set(r["case_id"] for r in calibrations)),
        calibration_rollout_count=len(calibrations),calibration_outcomes=dict(Counter(r["outcome"] for r in calibrations)),
        actor_transition_count=len(samples),synthetic_mapping=synthetic,
        maximum_mapping_error=max(r["mapping_error"] for r in samples))
    dump(output/"summary.json",summary)
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--seeds",type=int,default=20)
    parser.add_argument("--preset",default=PRESET)
    parser.add_argument("--reference-dir",type=Path)
    parser.add_argument("--reference-eval",type=Path)
    parser.add_argument("--checkpoint")
    cli=parser.parse_args()
    if cli.checkpoint is None:
        cli.checkpoint=f"checkpoints/{cli.preset}/model_final"
    if cli.reference_dir is not None and cli.reference_eval is not None:
        parser.error("Use only one of --reference-dir and --reference-eval")
    if cli.output_dir.exists():
        raise FileExistsError(cli.output_dir)
    if not 1<=cli.seeds<=20:
        parser.error("--seeds must be 1..20")
    args=preset_args(cli.preset)
    class ScanCallback(change_td3.EvalCallback):
        def on_step(self,trainer,*,force=False):
            scan(self,trainer,args,cli)
    command=build_mode_cmd(cli.preset)[3:]+["--evaluate_only","true","--load_checkpoint",cli.checkpoint,
        "--log_dir",str(cli.output_dir.parent/(cli.output_dir.name+"_loader"))]
    with patch.object(change_td3,"EvalCallback",ScanCallback),patch.object(sys,"argv",[__file__]+command):
        change_td3.main()


if __name__=="__main__":
    main()
