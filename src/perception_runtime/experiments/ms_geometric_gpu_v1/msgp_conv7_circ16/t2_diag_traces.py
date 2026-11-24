"""Replay paired T2 scenes and cross-score archived critics on executed actions."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import multiprocessing
from pathlib import Path

import numpy as np
import torch

from agent import Actor, Critic, project
from evaluate import make_validation_env, realtime_decisions
from perception import CELLS, ObservationBuilder, unpack
from t0_eval import initial_info
from t2_eval import read_json, write_json
from train import sha

NODES = (80000, 120000)
VELOCITY_SCALE = np.array([.8, .8, .3])


def select_cases(docs):
    lookup = [{(r["stage"], r["case"]): r for r in d["rows"]} for d in docs]
    chosen = []
    for reason, quota in (("xy_bound", 10), ("collision", 5)):
        by_stage = defaultdict(list)
        for key, old in sorted(lookup[0].items()):
            new = lookup[1][key]
            if old["success"] and new["reason"] == reason:
                by_stage[key[0]].append(dict(stage=key[0], case=key[1], reason=reason, expected=[old, new]))
        group = []
        while len(group) < quota:
            for stage in range(4):
                if by_stage[stage] and len(group) < quota:
                    group.append(by_stage[stage].pop(0))
            if not any(by_stage.values()) and len(group) < quota:
                raise RuntimeError(f"insufficient paired cases for {reason}")
        chosen.extend(group)
    return chosen


def initialize(paths):
    global actors, critics, contract, builder, gamma, reward_scale
    torch.set_num_threads(1)
    actors, critics = {}, {}
    for path in paths:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        node = ck["trainer"]["step"]
        contract = ck["contract"]
        if contract["use_far_branch"] or contract["course"] != "original":
            raise RuntimeError("requested traces need T2 near-only original-course checkpoints")
        actor, critic = Actor(False), Critic(False)
        actor.load_state_dict(ck["agent"]["actor"])
        critic.load_state_dict(ck["agent"]["critic"])
        actors[node], critics[node] = actor.eval(), critic.eval()
        gamma = contract["td3"]["gamma"]
        del ck
    from experiments.direct_velocity_full_v1.task_factory import load_config
    reward_scale = load_config()["reward_scale"]
    from experiments.rv1_range_cnn.lidar import LidarConfig
    builder = ObservationBuilder("cpu", LidarConfig(**contract["sensor"]), False)


def xy_bounds(plant):
    if plant.MAP_SPEC is not None and plant.MAP_SPEC.xy_half_extents is not None:
        return np.asarray(plant.MAP_SPEC.bounds_xy, dtype=np.float64)
    return np.full(2, max(plant.FOREST_HALF_EXTENT + .75, max(abs(plant.START_POS[:2])) + .5,
                          max(abs(plant.TARGET_POS[:2])) + .5))


@torch.inference_mode()
def run_episode(node, stage, case, expected):
    env = make_validation_env(contract)
    try:
        env.obstacle_curriculum.stage = stage
        obs, info = env.reset(seed=case)
        initial = initial_info(env, info)
        if initial != expected["init"]:
            raise RuntimeError("pair initialization does not reproduce fixed evaluation")
        phys = env.base_env
        bounds = xy_bounds(phys)
        states, records = [], []
        collision = bool(info["collision"])
        with realtime_decisions():
            for tick in range(2000):
                features = builder(obs)
                network_action = actors[node](unpack(features[None], False))
                action = project(network_action).numpy()[0]
                pos = np.asarray(phys.pos[0]).copy()
                vel = np.asarray(phys.vel[0]).copy()
                clear = float(info["min_tree_clearance"])
                goal = float(info["goal_distance"])
                states.append(features.numpy().copy())
                obs, reward, term, trunc, info = env.step(action)
                executed = info["projected_action"]
                collision |= bool(info["collision"])
                records.append(dict(
                    tick=tick, time_s=tick / 15, position_m=pos.tolist(), next_position_m=phys.pos[0].tolist(),
                    velocity_world_mps=vel.tolist(), goal_distance_m=goal, next_goal_distance_m=float(info["goal_distance"]),
                    min_tree_clearance_m=clear if np.isfinite(clear) else None,
                    xy_margin_m=float(np.min(bounds - np.abs(pos[:2]))),
                    occupied_voxels=int(np.count_nonzero(states[-1][:CELLS])),
                    actor_raw=network_action.numpy()[0].tolist(), executed_action=executed.tolist(),
                    command_yaw_frame_mps=(executed * VELOCITY_SCALE).tolist(),
                    command_world_mps=info["accepted_world_command"].tolist(), reward_scaled=float(reward * reward_scale),
                    reason=info["done_reason"],
                ))
                if term or trunc:
                    break
        if ((len(records), info["done_reason"], collision) !=
                (expected["upper_steps"], expected["reason"], expected["collision_observed"])
                or abs(info["goal_distance"] - expected["final_goal_distance_m"]) > 1e-5):
            raise RuntimeError(f"rollout does not reproduce frozen reference {node}/{stage}/{case}")
        features = torch.from_numpy(np.stack(states))
        actions = torch.tensor([row["executed_action"] for row in records], dtype=torch.float32)
        q_values = {}
        counterfactual = {}
        for scored_node in NODES:
            heads = [torch.cat(critics[scored_node](unpack(x, False), a), dim=1).numpy()
                     for x, a in zip(features.split(128), actions.split(128))]
            q_values[scored_node] = np.concatenate(heads)
            counterfactual[scored_node] = np.concatenate(
                [project(actors[scored_node](unpack(b, False))).numpy() * VELOCITY_SCALE for b in features.split(128)])
        realized = 0.
        for i in range(len(records) - 1, -1, -1):
            row = records[i]
            realized = row["reward_scaled"] + gamma * realized
            row["realized_discounted_return"] = realized
            row["critics"] = {str(n): {"q1": float(q_values[n][i, 0]), "q2": float(q_values[n][i, 1]),
                                       "minimum": float(q_values[n][i].min())} for n in NODES}
            row["actors_on_same_input_yaw_frame_mps"] = {str(n): counterfactual[n][i].tolist() for n in NODES}
        return dict(initial=initial, bounds_xy_m=bounds.tolist(), goal_m=phys.TARGET_POS.tolist(), rows=records)
    finally:
        env.close()


def pair_summary(old, new):
    a, b = old["rows"], new["rows"]
    origin = np.asarray(a[0]["position_m"][:2])
    tangent = np.asarray(old["goal_m"][:2]) - origin
    tangent /= np.linalg.norm(tangent)
    lateral = np.array([-tangent[1], tangent[0]])
    target_toward = lambda row: float(np.asarray(row["command_world_mps"][:2]) @ tangent)
    target_lateral = lambda row: float(np.asarray(row["command_world_mps"][:2]) @ lateral)
    divergence = None
    for u, v in zip(a, b):
        gap = float((np.asarray(v["position_m"][:2]) - u["position_m"][:2]) @ lateral)
        if abs(gap) >= .5:
            divergence = dict(tick=v["tick"], time_s=v["time_s"], signed_lateral_gap_m=gap,
                              goal_distance_m=v["goal_distance_m"],
                              clearance_m=v["min_tree_clearance_m"])
            break
    first_near = next((r["time_s"] for r in b if r["min_tree_clearance_m"] is not None
                       and r["min_tree_clearance_m"] <= .5), None)
    early = {}
    for node, rows in zip(NODES, (a, b)):
        early[str(node)] = dict(first_command_world=rows[0]["command_world_mps"],
                                forward_world_first_15_mps=float(np.mean([target_toward(r) for r in rows[:15]])),
                                lateral_world_first_15_mps=float(np.mean([target_lateral(r) for r in rows[:15]])),
                                lateral_positive_fraction_first_15=float(np.mean([target_lateral(r) > 0 for r in rows[:15]])))
    final_xy = np.array(b[-1]["next_position_m"][:2])
    bounds = np.array(new["bounds_xy_m"])
    crossed = [axis + ("+" if pos >= 0 else "-") for axis, pos, limit in zip("xy", final_xy, bounds) if abs(pos) > limit]
    tail = b[-15:]
    critic_diff = np.array([r["critics"]["120000"]["minimum"] - r["critics"]["80000"]["minimum"] for r in tail])
    return dict(first_half_meter_lateral_divergence=divergence, first_clearance_at_most_half_m_s=first_near,
                initial_commands=early, crossed_world_bounds=crossed,
                terminal_goal_distance_m=b[-1]["next_goal_distance_m"],
                tail_n=len(tail), q120_minus_q80_last15_mean=float(critic_diff.mean()),
                q120_minus_q80_terminal=float(critic_diff[-1]),
                terminal_reward_scaled=b[-1]["reward_scaled"],
                q80_terminal=b[-1]["critics"]["80000"]["minimum"],
                q120_terminal=b[-1]["critics"]["120000"]["minimum"],
                t120_largest_abs_lateral_command_first_15_mps=float(max(abs(target_lateral(r)) for r in b[:15])))


def run_pair(case):
    old, new = [run_episode(node, case["stage"], case["case"], expected)
                for node, expected in zip(NODES, case["expected"])]
    return dict(stage=case["stage"], case=case["case"], failure=case["reason"], old=old, new=new,
                summary=pair_summary(old, new))


def paired_inference(group):
    values = [r["summary"] for r in group]
    collected = defaultdict(list)
    for r in group:
        collected[r["case"]].append(r["summary"]["q120_minus_q80_last15_mean"])
    by_seed = np.array([np.mean(v) for v in collected.values()])
    observed = float(by_seed.sum())
    null = sum(float(np.dot(signs, by_seed)) >= observed - 1e-10
               for signs in product((-1, 1), repeat=len(by_seed))) / 2 ** len(by_seed)
    return dict(n=len(values), unique_map_seeds=len(by_seed), mean_tail_diff=float(np.mean(
                    [r["q120_minus_q80_last15_mean"] for r in values])),
                higher_q120_cases=sum(r["q120_minus_q80_last15_mean"] > 0 for r in values),
                p_one_sided_map_sign_permutation=null,
                q80_minus_observed_terminal_reward_mean=float(np.mean(
                    [r["q80_terminal"] - r["terminal_reward_scaled"] for r in values])),
                q120_minus_observed_terminal_reward_mean=float(np.mean(
                    [r["q120_terminal"] - r["terminal_reward_scaled"] for r in values])),
                mean_terminal_q_difference=float(np.mean([r["q120_minus_q80_terminal"] for r in values])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    fixed = [read_json(args.run / f"evaluation_{node:06d}.json") for node in NODES]
    paths = [args.run / "train" / f"checkpoint_{node:06d}.pt" for node in NODES]
    if any(sha(path) != report["checkpoint_sha256"] for path, report in zip(paths, fixed)):
        raise RuntimeError("trajectory checkpoint digest mismatch")
    chosen = select_cases(fixed)
    patients = []
    context = multiprocessing.get_context("spawn")
    folder = args.output / "paired_trajectories"
    folder.mkdir(exist_ok=False)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context, initializer=initialize,
                             initargs=(paths,)) as pool:
        for future in as_completed([pool.submit(run_pair, case) for case in chosen]):
            r = future.result()
            name = f"{r['failure']}_s{r['stage']}_{r['case']:02d}.json"
            write_json(folder / name, r)
            patients.append({k: v for k, v in r.items() if k not in ("old", "new")} | {"trajectory": name})
            print(f"Confirmed paired rollout {r['stage']}/{r['case']} {len(patients)}/15", flush=True)
    patients.sort(key=lambda r: (r["failure"] != "xy_bound", r["stage"], r["case"]))
    groups = {reason: paired_inference([r for r in patients if r["failure"] == reason])
              for reason in ("xy_bound", "collision")}
    groups["all"] = paired_inference(patients)
    doc = dict(cases=patients, critic_groups=groups, source_checkpoint_sha256={s: d["checkpoint_sha256"]
               for s, d in zip(NODES, fixed)}, clean_policy_rollouts=True, evaluation_device="cpu",
               selection="Round robin across stages, ascending seed within stage; fixed failures",
               q_definition="min of critic twin heads on each executed normalized state-action pair",
               critic_inference_unit="Case tail means, grouped by map seed for the one-sided sign permutation",
               terminal_comparator="Observed terminal reward multiplied by frozen reward scale")
    write_json(args.output / "paired_trace_summary.json", doc)


if __name__ == "__main__":
    main()
