"""Apply the frozen E11 evaluation protocol to an RV1 CNN checkpoint."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import numpy as np
import torch

from envs.map_atlas.forest_pool import ForestPool
from experiments.direct_velocity_full_v1.contracts import normalized_action_to_world, yaw_from_quaternion
from experiments.direct_velocity_full_v1.task_factory import ROOT, load_config
from experiments.ms_geometric_mixed.mixed_course import assigned_stage
from experiments.rv1_range_cnn.environment import RangeImageEnv, make_course_env
from experiments.rv1_range_cnn.td3 import RangeTD3
from experiments.rv1_range_cnn.train_formal import FORMAT, sha

for source in ("ms_geometric_gpu_v1", "ms_geometric_directional"):
    sys.path.insert(0, str(ROOT / "experiments" / source))
from evaluate_directional import APERTURES, first_reference_turn, follow, initial_info, paired_map, through

BASE = Path("/opt/uav-training-20261103/runs")
E11 = BASE / "directional_mixed_seed43_40k_20261119a"
STEP = 40000


def identity(x):
    return x


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def init_worker(checkpoint):
    global policy, forest_pool
    torch.set_num_threads(1)
    config = load_config()
    forest_pool = ForestPool(ROOT / config["forest_pool_path"])
    pack = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy = RangeTD3(config["td3"], pack["contract"], device="cpu")
    policy.actor.load_state_dict(pack["agent"]["actor"])
    policy.actor.eval()


def forest(stage, cases):
    result = []
    env = make_course_env(forest_pool, training=False)
    try:
        env.obstacle_curriculum.stage = stage
        for case in cases:
            obs, info = env.reset(seed=case)
            if (info["density_stage"], info["start_curriculum_source"]) != (stage, "original"):
                raise RuntimeError("E11 forest initialization changed")
            init = initial_info(env, info)
            row, _ = follow(env, policy, identity, obs, info)
            result.append(dict(stage=stage, case=case, init=init, **row))
    finally:
        env.close()
    return result


def aperture(case, trace_dir):
    result = []
    wall, offset, width = APERTURES[case]
    for side in (-1, 1):
        env = RangeImageEnv(map_spec=paired_map(case, side))
        try:
            obs, info = env.reset(seed=1000 + case)
            init = initial_info(env, info)
            command = normalized_action_to_world(policy.action(obs), yaw_from_quaternion(env.base_env.quat[0]))
            row, trajectory = follow(env, policy, identity, obs, info, wall)
            lead = first_reference_turn(trajectory, side)
            row.update(case=case, side=side, init=init, initial_command=command.tolist(),
                       first_sustained_turn_lead_m=lead, early_turn=lead is not None and lead > 1,
                       through_opening=through(trajectory, wall, side * offset, width))
            (trace_dir / f"{case:02d}_{side:+d}.json").write_text(
                json.dumps(dict(row=row, trajectory=trajectory), allow_nan=False))
            result.append(row)
        finally:
            env.close()
    return result


def summary(rows):
    return dict(
        n=len(rows), reasons=dict(Counter(row["reason"] for row in rows)),
        collisions=sum(row["collision_observed"] for row in rows),
        median_final_goal_distance_m=float(np.median([row["final_goal_distance_m"] for row in rows])),
    )


def verify_training(run, checkpoint, reference):
    pack = torch.load(checkpoint, map_location="cpu", weights_only=False)
    agent = pack["agent"]
    metadata = read_json(run / "metadata.json")
    seed = metadata["config"]["seed"]
    replay = run / pack["replay_file"]
    if (pack["format"], pack["trainer"]["step"], seed, pack["device"],
            agent["updates"], agent["actor_updates"]) != (FORMAT, STEP, 43, "cuda", 35001, 17500):
        raise RuntimeError("expected the original RV1 Seed 43 checkpoint at exactly 40k")
    if sha(replay) != pack["replay_sha256"]:
        raise RuntimeError("RV1 replay checksum mismatch")
    episodes = [r for r in read_jsonl(run / "episodes.jsonl") if r["step"] <= STEP]
    updates = [r for r in read_jsonl(run / "updates.jsonl") if r["step"] <= STEP]
    if [r["step"] for r in updates] != list(range(5000, STEP + 1)):
        raise RuntimeError("missing or repeated training update")
    provenance = dict(replay_sha256=pack["replay_sha256"], checkpoint_sha256=sha(checkpoint),
                      training_reasons=dict(Counter(r["done_reason"] for r in episodes)),
                      last_1000_mean_critic_loss=float(np.mean([r["critic_loss"] for r in updates[-1000:]])),
                      last_1000_mean_q=float(np.mean([(r["q1_mean"] + r["q2_mean"]) / 2 for r in updates[-1000:]])))
    if reference:
        return provenance
    starts = read_jsonl(run / "episode_starts.jsonl")
    if not all(r["episode"] == i and r["stage"] == assigned_stage(seed, i) for i, r in enumerate(starts)):
        raise RuntimeError("E11 episode schedule mismatch")
    if len(starts) != len(episodes) + 1:
        raise RuntimeError("missing episode start or conclusion")
    stage_steps = Counter()
    past = 0
    for episode, start in zip(episodes, starts):
        if episode["episode"] != start["episode"] or not past < episode["step"] <= STEP:
            raise RuntimeError("invalid episode index or step")
        stage_steps[start["stage"]] += episode["step"] - past
        past = episode["step"]
    stage_steps[starts[-1]["stage"]] += STEP - past
    common = {}
    for mode in ("original", "directional"):
        reference_starts = read_jsonl(E11 / mode / "episode_starts.jsonl")
        keys = ("episode", "stage", "map_pool_index", "geometry_sha256")
        for actual, expected in zip(starts, reference_starts):
            if any(actual[k] != expected[k] for k in keys):
                raise RuntimeError(f"E11 map sequence mismatch for {mode}")
        common[mode] = min(len(starts), len(reference_starts))
    return dict(episode_starts_by_stage=dict(Counter(r["stage"] for r in starts)),
                training_stage_steps=dict(stage_steps), common_e11_map_prefix=common, **provenance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    checkpoint = args.run / f"checkpoint_{STEP:06d}.pt"
    training = verify_training(args.run, checkpoint, args.reference)
    history = read_json(E11 / "evaluation_40000/results.json")
    protocol = read_json(E11 / "evaluation_40000/protocol.json")
    if protocol["aperture_cases"] != [list(x) for x in APERTURES]:
        raise RuntimeError("E11 aperture definitions changed")
    output = args.output
    trace_dir = output / "aperture_traces"
    output.mkdir(exist_ok=False)
    trace_dir.mkdir()
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    forest_rows, aperture_rows = [], []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(checkpoint,)) as pool:
        cases = protocol["validation_case_indices"]
        forest_jobs = [pool.submit(forest, stage, cases[i:i + 5])
                       for stage in protocol["forest_stages"] for i in range(0, len(cases), 5)]
        aperture_jobs = [pool.submit(aperture, case, trace_dir) for case in range(len(APERTURES))]
        for i, job in enumerate(as_completed(forest_jobs + aperture_jobs), 1):
            (forest_rows if job in forest_jobs else aperture_rows).extend(job.result())
            print(f"Completed evaluation batch {i}/{len(forest_jobs) + len(aperture_jobs)}", flush=True)
    forest_rows.sort(key=lambda row: (row["stage"], row["case"]))
    aperture_rows.sort(key=lambda row: (row["case"], row["side"]))
    forest_inits = {(r["stage"], r["case"]): r["init"] for r in history["forest"]}
    aperture_inits = {(r["case"], r["side"]): r["init"] for r in history["probe"]}
    if len(forest_rows) != 80 or any(r["init"] != forest_inits[r["stage"], r["case"]] for r in forest_rows):
        raise RuntimeError("forest cases did not match E11")
    if len(aperture_rows) != 24 or any(r["init"] != aperture_inits[r["case"], r["side"]] for r in aperture_rows):
        raise RuntimeError("aperture cases did not match E11")
    outcome = dict(
        training=training, forest=summary(forest_rows),
        stage={str(s): summary([r for r in forest_rows if r["stage"] == s]) for s in range(4)},
        aperture=dict(**summary(aperture_rows), through_opening=sum(r["through_opening"] for r in aperture_rows)),
    )
    result = dict(summary=outcome, forest=forest_rows, aperture=aperture_rows)
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(outcome, indent=2), flush=True)


if __name__ == "__main__":
    main()
