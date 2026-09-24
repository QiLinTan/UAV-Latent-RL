"""Branch real policy trajectories in the original forest; compare reward returns.

Uses the production CLI to construct the environment and load the actor. Only
its evaluation callback is replaced. No replay collection or training occurs.
Every branch replays the original action prefix, preserving geometry, velocity,
observation history and remaining episode time. Scripted branches are candidate
trajectories, not an optimal planner or an estimate of the actor's success rate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from envs.preprocess import preprocess_state
from benchmarks.forest_benchmark import grid_route
from scripts import change_td3
from scripts.diagnose_terminal_reward_rollouts import (
    PRESET, ROOT, command_for, feasible, preset_args, reward_model,
)
from scripts.run_change_td3 import build_mode_cmd


BEHAVIORS = (
    "actor", "fast_to_goal", "scheduled_capture", "height_then_capture",
    "xy_only", "hold_outside", "brake_to_stop", "planned_capture",
)


def planned_path(env):
    """Privileged 2-D diagnostic path, never supplied to the learned policy.

    Treat trees as infinite cylinders, so the route also permits descending.
    Production swept-body collision tests still determine each rollout outcome.
    """
    trees = env.base_env._tree_specs
    assert trees, "This diagnostic planner expects the procedural tree map."
    record = dict(start=env.pos[0].copy(), goal=env.TARGET_POS.copy(), trees=trees)
    inflation = float(env.COLLISION_R)+0.22
    path = grid_route(record, inflation=inflation,
                      bound=max(env.FOREST_HALF_EXTENT, 4.0))
    if path is None:
        return None
    centers = np.asarray([tree["xy"] for tree in trees])
    radii = np.asarray([tree["radius"]+inflation for tree in trees])
    def visible(start, end):
        delta = end-start
        u = np.clip(((centers-start)@delta)/max(float(delta@delta), 1e-9), 0, 1)
        distances = np.linalg.norm(centers-(start+u[:,None]*delta), axis=1)
        return bool(np.all(distances > radii))
    smooth, index = [path[0]], 0
    while index < len(path)-1:
        candidate = len(path)-1
        while candidate > index+1 and not visible(path[index], path[candidate]):
            candidate -= 1
        smooth.append(path[candidate])
        index = candidate
    return np.asarray(smooth[1:])
COLUMNS = (
    "time", "x", "y", "z", "vx", "vy", "vz", "distance",
    "baseline_reward", "no_route_reward", "discounted_baseline", "discounted_no_route",
    "route_progress_reward", "goal_progress_reward", "tracking_penalty",
    "height_penalty", "proximity_penalty", "terminal_reward",
)


def score_forest_step(env, alternative, command):
    base = env.base_env
    previous_position, previous_velocity = base.pos[0].copy(), base.vel[0].copy()
    previous_distance = float(np.linalg.norm(base.TARGET_POS-previous_position))
    action = np.asarray(command)/[env.max_reference_speed, env.max_reference_speed,
                                  env.max_reference_vertical_speed]
    assert np.linalg.norm(action[:2]) <= 1+1e-6 and abs(action[2]) <= 1+1e-6
    expected_velocity = env._next_kinematic_velocity(previous_velocity, env._velocity_command(action))
    obs, reward, terminated, truncated, info = env.step(action)
    state = base._getDroneStateVector(0)
    alt_reward, _, _ = alternative.compute(
        state=state, prev_goal_dist=previous_distance, prev_pos=previous_position,
        start_pos=base.START_POS, target_pos=base.TARGET_POS,
        goal_tolerance=base.GOAL_TOLERANCE, safe_distance=base.SAFE_DISTANCE,
        clearance=base._computeNearestTreeClearance(state[:3]),
        collision=bool(info["collision"]), terminal_reason=info["done_reason"])
    # The production K1 clips components, then projects onto the speed disk.
    # Verify that exact plant contract rather than imposing a different model.
    np.testing.assert_allclose(base.vel[0], expected_velocity, atol=2e-6)
    terms = base._last_reward_terms.copy()
    expected = (terms["terminal_outcome_reward"] if terminated or truncated else
                terms["progress_reward"]-terms["continuous_penalty_total"])
    assert np.isclose(reward, expected, atol=3e-5)
    return obs, float(alt_reward), float(reward), terms, terminated or truncated


def comparison_summary(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["case_id"], []).append(row)
    result = {}
    for version in ("baseline", "no_route"):
        margins, all_margins, counterexamples, no_success = [], [], [], []
        for case, group in groups.items():
            success = [r for r in group if r["outcome"] == "success"]
            failure = [r for r in group if r["outcome"] != "success"]
            if not success:
                no_success.append(case)
            if not success or not failure:
                continue
            best_success = max(success, key=lambda r: r[f"G_{version}"])
            best_failure = max(failure, key=lambda r: r[f"G_{version}"])
            margin = best_success[f"G_{version}"] - best_failure[f"G_{version}"]
            margins.append(margin)
            all_margins.extend(s[f"G_{version}"] - f[f"G_{version}"]
                               for s in success for f in failure)
            if margin <= 0:
                counterexamples.append(dict(case_id=case, margin=margin,
                    success_behavior=best_success["behavior"],
                    failure_behavior=best_failure["behavior"]))
        result[version] = dict(
            comparable_cases=len(margins), best_success_better=sum(m > 0 for m in margins),
            minimum_best_success_margin=min(margins, default=None),
            success_failure_pair_count=len(all_margins),
            success_failure_pair_wins=sum(m > 0 for m in all_margins),
            minimum_any_success_margin=min(all_margins, default=None),
            counterexamples=counterexamples, cases_without_sampled_success=no_success,
        )
    return result


def audit(callback, trainer, args, cli):
    output = cli.output_dir
    output.mkdir(parents=True, exist_ok=False)
    alt_args = copy.copy(args)
    alt_args.route_progress_reward_weight = 0.0
    no_route = reward_model(alt_args)
    env = callback.env_cls(**callback.env_kwargs, gui=False)
    # Match EvalCallback: validation maps explicitly disable curriculum.
    if bool(getattr(env, "CURRICULUM_ENABLED", False)):
        env.set_curriculum_stage_override(trainer.env.get_curriculum_stage())
    references = None
    if cli.reference_eval is not None:
        references = json.loads(cli.reference_eval.read_text().splitlines()[-1])["episodes"]
    records, cases, traces, policy_episodes = [], [], {}, []
    try:
        for seed in range(cli.seeds):
            obs, _ = env.reset(seed=seed)
            actions, states = [], []
            done = False
            while not done:
                states.append(dict(position=env.pos[0].copy(), velocity=env.vel[0].copy(),
                                   observation=np.asarray(obs).copy()))
                action = trainer.agent.select_action(preprocess_state(obs.reshape(-1)))
                actions.append(action.copy())
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            target = env.TARGET_POS.copy()
            distances = [float(np.linalg.norm(target - s["position"])) for s in states]
            policy_episodes.append(dict(seed=seed, steps=len(actions),
                outcome=info["done_reason"], final_distance=float(info["goal_distance"])))
            if references is not None:
                reference = next(r for r in references if r["episode"] == seed)
                assert info["done_reason"] == reference["done_reason"]
                assert len(actions) == reference["length"]
                assert abs(float(info["goal_distance"])-reference["final_goal_distance"]) < 1e-4
            indices = {0: ["reset"]}
            for index, label in ((45, "3_seconds"), (90, "6_seconds"),
                                 (int(np.argmin(distances)), "closest_preterminal")):
                if index < len(states):
                    indices.setdefault(index, []).append(label)
            xy_entries = [i for i, s in enumerate(states)
                          if np.linalg.norm(target[:2] - s["position"][:2]) < 1.2]
            if xy_entries:
                indices.setdefault(xy_entries[0], []).append("first_xy_within_1.2m")
            for index, labels in sorted(indices.items()):
                state = states[index]
                case_id = f"seed{seed:02d}_step{index:03d}"
                elapsed = index * env.CTRL_TIMESTEP
                case = dict(id=case_id, seed=seed, step=index, labels=labels,
                    position=state["position"].tolist(), velocity=state["velocity"].tolist(),
                    elapsed=elapsed, remaining_seconds=max(0.0, env.EPISODE_LEN_SEC-elapsed),
                    d_xy=float(np.linalg.norm(target[:2]-state["position"][:2])),
                    z_error=abs(float(target[2]-state["position"][2])), d_3d=distances[index])
                cases.append(case)
                for behavior in BEHAVIORS:
                    obs, _ = env.reset(seed=seed)
                    for action in actions[:index]:
                        obs, _, terminated, truncated, _ = env.step(action)
                        assert not (terminated or truncated)
                    np.testing.assert_allclose(env.pos[0], state["position"], atol=1e-6)
                    np.testing.assert_allclose(env.vel[0], state["velocity"], atol=1e-6)
                    np.testing.assert_allclose(obs, state["observation"], atol=1e-6)
                    assert env.runtime_lower_step == index * args.high_level_interval
                    path = planned_path(env) if behavior == "planned_capture" else None
                    waypoint_index = 0
                    rows = []
                    for step in range(200):
                        if behavior == "actor":
                            action = trainer.agent.select_action(preprocess_state(obs.reshape(-1)))
                            command = env._velocity_command(action)
                        elif behavior == "xy_only":
                            command = feasible(np.r_[2*(target[:2]-env.pos[0][:2]),
                                1.5*(case["position"][2]-env.pos[0][2])], env)
                        elif behavior == "planned_capture" and path is not None:
                            while (waypoint_index < len(path)-1 and
                                   np.linalg.norm(path[waypoint_index]-env.pos[0][:2]) < 0.12):
                                waypoint_index += 1
                            offset = path[waypoint_index]-env.pos[0][:2]
                            command = feasible(np.r_[2.5*offset, 1.5*(target[2]-env.pos[0][2])], env)
                        elif behavior == "planned_capture":
                            command = command_for("fast_to_goal", env, case, step*env.CTRL_TIMESTEP)
                        else:
                            command = command_for(behavior, env, case, step*env.CTRL_TIMESTEP)
                        obs, r0, rb, terms, done = score_forest_step(env, no_route, command)
                        multiplier = args.reward_scale * args.gamma**step
                        terminal = env.base_env._last_done_reason != "running"
                        route = 0.0 if terminal else args.route_progress_reward_weight*terms["route_progress"]
                        goal = 0.0 if terminal else args.goal_progress_reward_weight*terms["goal_progress"]
                        assert np.isclose(rb-r0, route, atol=3e-5)
                        rows.append([(step+1)*env.CTRL_TIMESTEP, *env.pos[0], *env.vel[0],
                            np.linalg.norm(target-env.pos[0]), rb, r0,
                            multiplier*rb, multiplier*r0, route, goal,
                            terms["near_goal_velocity_tracking_penalty"], terms["height_penalty"],
                            terms["proximity_penalty"], terms["terminal_outcome_reward"]])
                        if done:
                            break
                    else:
                        raise AssertionError("Natural termination not reached")
                    trace = np.asarray(rows)
                    key = f"{case_id}_{behavior}"
                    traces[key] = trace
                    row = dict(case_id=case_id, seed=seed, branch_step=index,
                        behavior=behavior, outcome=env.base_env._last_done_reason,
                        steps=len(trace), min_distance=float(trace[:,7].min()),
                        final_distance=float(trace[-1,7]),
                        guidance_steps=int(np.sum((trace[:,7]>=env.GOAL_TOLERANCE)&(trace[:,7]<1.2))),
                        G_baseline=float(trace[:,10].sum()), G_no_route=float(trace[:,11].sum()),
                        raw_baseline=float(trace[:,8].sum()), raw_no_route=float(trace[:,9].sum()),
                        raw_terms={name:float(trace[:,col].sum()) for name,col in
                                   (("route",12),("goal",13),("tracking",14),("height",15),
                                    ("proximity",16),("terminal",17))}, trace_key=key)
                    if behavior == "planned_capture":
                        row["planner_path_available"] = path is not None
                        row["planner_waypoints"] = path.tolist() if path is not None else []
                    for horizon in (1, 2):
                        mask = trace[:,0] <= horizon+1e-8
                        row[f"G_baseline_first_{horizon}s"] = float(trace[mask,10].sum())
                        row[f"G_no_route_first_{horizon}s"] = float(trace[mask,11].sum())
                    if behavior == "actor":
                        assert row["outcome"] == policy_episodes[-1]["outcome"]
                        assert row["steps"] == len(actions)-index
                        assert abs(row["final_distance"]-policy_episodes[-1]["final_distance"]) < 1e-5
                    records.append(row)
            print(f"Audit seed {seed}: {len(cases)} states / {len(records)} branches", flush=True)
    finally:
        env.close()
    summary = dict(
        schema_version="forest_spatial_terminal_return_audit_v1",
        metadata=dict(preset=PRESET, checkpoint=cli.checkpoint, seeds=list(range(cli.seeds)),
            reference_evaluation=str(cli.reference_eval) if cli.reference_eval else None,
            actor_sha256=hashlib.sha256(Path(cli.checkpoint+"_actor").read_bytes()).hexdigest(),
            gamma=args.gamma, reward_scale=args.reward_scale, trace_columns=COLUMNS,
            baseline_route_weight=args.route_progress_reward_weight, alternative_route_weight=0.0,
            goal_weight=args.goal_progress_reward_weight, counterfactual_reward_only=True,
            remaining_time_preserved=True, source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                for name in ("envs/forest/rewards.py", "envs/learned_upper_dslpid.py", "scripts/audit_spatial_terminal_consistency.py")}),
        cases=cases, case_count=len(cases), rollout_count=len(records),
        policy_episodes=policy_episodes, rankings=comparison_summary(records),
        by_behavior={b:dict(Counter(r["outcome"] for r in records if r["behavior"]==b)) for b in BEHAVIORS},
    )
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    (output/"rollouts.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in records))
    np.savez_compressed(output/"traces.npz", **traces)
    print(json.dumps({k:summary[k] for k in ("case_count","rollout_count","rankings","by_behavior")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--checkpoint", default=f"checkpoints/{PRESET}/model_final")
    parser.add_argument("--reference-eval", type=Path, help="Assert exact seed outcomes against prior formal evaluation JSONL.")
    cli = parser.parse_args()
    if cli.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {cli.output_dir}")
    if cli.seeds <= 0:
        parser.error("--seeds must be positive")
    args = preset_args()
    class AuditCallback(change_td3.EvalCallback):
        def on_step(self, trainer, *, force=False):
            audit(self, trainer, args, cli)
    command = build_mode_cmd(PRESET)[3:] + [
        "--evaluate_only", "true", "--load_checkpoint", cli.checkpoint,
        "--log_dir", str(cli.output_dir.parent/(cli.output_dir.name+"_loader")),
    ]
    with patch.object(change_td3, "EvalCallback", AuditCallback), patch.object(sys, "argv", [__file__]+command):
        change_td3.main()


if __name__ == "__main__":
    main()
