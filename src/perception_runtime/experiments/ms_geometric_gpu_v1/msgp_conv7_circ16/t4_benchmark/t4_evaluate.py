"""Run the paired standard and sealed T4 evaluations after training completes."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path

import numpy as np
import pybullet as bullet
import torch

from envs.map_atlas.spec import MapSpec
from evaluate import load_policy, realtime_decisions
from experiments.rv1_range_cnn.environment import RangeImageEnv
from sensor import use_parallel_lidar
from t0_eval import follow, initial_info, through
from t2_diag_traces import xy_bounds
from t2_eval import read_json, read_lines, write_json
import t3_eval as base
from t4_suite import load_suite
from t4_train import NODES, RUN_NAMES
from train import sha

SEEDS = (43, 44, 45)
MODELS = ("t3", "a1")
FINAL_NODES = (200000, 240000)
CHECKED_CONFIG = ("td3", "reward_scale", "reward_kwargs", "control_spec", "curriculum")


def run_path(runs, seed, model):
    if seed == 43:
        return (runs / "msgp_conv7_circ16_dual_seed43_240k/train" if model == "t3"
                else runs / "formal_gpu_a1_seed43_240k_20261118a")
    return runs / RUN_NAMES[model].format(seed=seed) / "train"


def classify(reason, position):
    if reason != "height_bound":
        return reason
    z = np.float32(position[2])
    if z <= .08:
        return "low_height_bound"
    if z >= 2.75:
        return "high_height_bound"
    raise RuntimeError("height-bound result did not identify the bound")


def load_worker(model, seed, node, path, fingerprint):
    torch.set_num_threads(1)
    if model == "t3":
        base.agent, base.builder, base.contract, base.step = load_policy(path, "cpu")
        if (base.contract["course"], base.contract["seed"], base.contract["short"]) != ("original", seed, False):
            raise RuntimeError("not a formal T4 dual checkpoint")
    else:
        base.init_worker("a1", path, fingerprint)
    if base.step != node or base.contract["pool_fingerprint"] != fingerprint:
        raise RuntimeError("checkpoint node or validation pool mismatch")


def annotate_standard(rows):
    for row in rows:
        row["terminal_class"] = classify(row["reason"], row["final_position_m"])
    return rows


def fixed_forest(stage, ids):
    return annotate_standard(base.forest(stage, ids)[1])


def heldout_case(case):
    spec = MapSpec.from_dict(case["map"], root=Path("."))
    env = use_parallel_lidar(RangeImageEnv(map_spec=spec))
    try:
        obs, info = env.reset(seed=case["reset_seed"])
        initial = initial_info(env, info)
        initial["geometry_sha256"] = case["geometry_sha256"]
        aperture = case["aperture"]
        physics = env.base_env
        window = deque([(env.base_env.physics_ticks, float(info["goal_distance"]))], maxlen=61)
        zone_clearance = [[], [], []]
        contacts = []
        stalls = windows = 0
        max_x = float(physics.pos[0, 0])
        closest_goal = float(info["goal_distance"])
        original = env.step

        def track(action):
            nonlocal stalls, windows, max_x, closest_goal
            result = original(action)
            meta = result[-1]
            distance = float(meta["goal_distance"])
            closest_goal = min(closest_goal, distance)
            window.append((physics.physics_ticks, distance))
            if window[-1][0] - window[0][0] >= 4 * 240:
                windows += 1
                stalls += window[0][1] - window[-1][1] < .12
            if aperture or case["suite"] in ("B", "C"):
                for tick in env.last_ticks:
                    max_x = max(max_x, float(tick["position"][0]))
                if result[2] or result[3]:
                    contacts.extend(dict(body=int(row[2]), point_on_body_m=list(row[6]))
                                    for row in bullet.getContactPoints(bodyA=int(physics.DRONE_IDS[0]),
                                                                       physicsClientId=physics.CLIENT))
            if case["suite"] == "C":
                x = env.base_env.pos[0, 0]
                zone = 0 if x < -4 else 2 if x >= 4 else 1
                clearance = float(meta["min_tree_clearance"])
                if np.isfinite(clearance):
                    zone_clearance[zone].append(clearance)
            return result

        env.step = track
        wall = aperture["wall_center_x_m"] if aperture else None
        with realtime_decisions():
            score, trajectory = follow(env, base.agent, base.builder, obs, info, wall)
        final = env.base_env.pos[0].tolist()
        passed = through(trajectory, wall, aperture["opening_center_y_m"], aperture["opening_width_m"]) if aperture else None
        bounds = xy_bounds(physics)
        exits = [axis + ("+" if pos >= 0 else "-") for axis, pos, limit in zip("xy", final, bounds)
                 if abs(np.float32(pos)) > limit] if score["reason"] == "xy_bound" else []
        obstacle_contacts = [row for row in contacts if row["body"] in physics._tree_ids]
        opening = None
        if aperture:
            center, width = aperture["opening_center_y_m"], aperture["opening_width_m"]
            distance = min((abs(row["point_on_body_m"][1] - (center + side * width / 2))
                            for row in obstacle_contacts for side in (-1, 1)), default=None)
            contact_class = ("jamb" if distance <= .06 else "wall_face") if distance is not None else None
            opening = dict(contact_class=contact_class, contact_to_jamb_m=distance, reached_rear_plane=max_x >= wall + .09)
        corridor = None
        if case["suite"] == "C":
            x = final[0]
            corridor = dict(entered_dense=max_x >= -4, exited_dense=max_x >= 4, max_x_m=max_x,
                            terminal_phase="entry" if x < -4 else "exit" if x >= 4 else "dense",
                            terminal_boundary="entry" if abs(x + 4) <= 1 else "exit" if abs(x - 4) <= 1 else None,
                            phase_min_clearance_m=[min(a) if a else None for a in zone_clearance])
        return dict(id=case["id"], suite=case["suite"], stage=case["stage"], map_seed=case["seed"],
                    init=initial, final_position_m=final, terminal_class=classify(score["reason"], final),
                    xy_exits=exits, closest_goal_control_m=closest_goal,
                    x_overshoot_at_control_rate="x+" in exits and final[0] > initial["target"][0]
                    and closest_goal >= float(physics.GOAL_TOLERANCE),
                    terminal_contacts=contacts, opening=opening,
                    stall_windows=stalls, windows_at_least_4s=windows, passed=passed, corridor=corridor, **score)
    finally:
        env.close()


def heldout_group(cases):
    return [heldout_case(case) for case in cases]


def eval_jobs(model, seed, node, path, fingerprint, workers, manifest=None):
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=load_worker,
                             initargs=(model, seed, node, path, fingerprint)) as executor:
        if manifest is None:
            tasks = [executor.submit(fixed_forest, stage, range(i, i+5)) for stage in range(4)
                     for i in range(0, 20, 5)]
        else:
            maps = manifest["maps"]
            tasks = [executor.submit(heldout_group, maps[i:i+4]) for i in range(0, len(maps), 4)]
        rows = [row for job in as_completed(tasks) for row in job.result()]
    if manifest is None:
        return sorted(rows, key=lambda r: (r["stage"], r["case"]))
    return sorted(rows, key=lambda r: r["id"])


def archived_fixed(out, model, node):
    local = out.parent
    if node <= 160000:
        source = local / "t3_dual_seed43_160k"
        if model == "a1":
            source /= "history"
        return read_json(source / f"{model}_{node:06d}.json")
    source = local / "t3_dual_seed43_240k"
    if model == "a1":
        return read_json(source / f"rv1_comparison/a1_{node:06d}.json")
    return read_json(source / f"t3_{node:06d}.json")


def identify(runs, output):
    frozen = read_json(output / "frozen_seed43.json")
    standard_config = read_json(run_path(runs, 43, "t3") / "metadata.json")
    fingerprint = standard_config["contract"]["pool_fingerprint"]
    registry = {}
    for seed in SEEDS:
        for model in MODELS:
            run = run_path(runs, seed, model)
            metadata = read_json(run / "metadata.json")
            if metadata["config"]["seed"] != seed or any(
                    metadata["config"][key] != standard_config["config"][key] for key in CHECKED_CONFIG):
                raise RuntimeError("T4 frozen training contract mismatch")
            actual_pool = metadata.get("pool_fingerprint", metadata["contract"].get("pool_fingerprint"))
            if actual_pool != fingerprint:
                raise RuntimeError("T4 training map pool mismatch")
            events = {r["step"]: r for r in read_lines(run / "events.jsonl") if r["event"] == "checkpoint"}
            rows = {}
            for node in NODES:
                path = run / f"checkpoint_{node:06d}.pt"
                checksum = sha(path)
                recorded = (frozen["models"][model]["checkpoints"][str(node)]["sha256"]
                            if seed == 43 else events[node].get("sha256"))
                if recorded and checksum != recorded:
                    raise RuntimeError(f"T4 checkpoint integrity mismatch: {path}")
                rows[node] = dict(path=str(path), sha256=checksum)
            registry[seed, model] = rows
    return fingerprint, registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    out = args.output
    manifest = load_suite(out / "heldout_suite/manifest.json")
    if args.pilot:
        fingerprint = manifest["frozen_training_pool_fingerprint"]
        # Use only previously seen fixed validation maps for pipeline preflight.
        for model in MODELS:
            path = run_path(args.runs, 43, model) / "checkpoint_240000.pt"
            load_worker(model, 43, 240000, path, fingerprint)
            row, = fixed_forest(0, [0])
            if row["stage"] != 0 or row["case"] != 0:
                raise RuntimeError("standard pilot failed")
        print("T4 evaluator passed both fixed-validation pilots. Held-out policies were not run.", flush=True)
        return
    fingerprint, registry = identify(args.runs, out)
    if manifest["frozen_training_pool_fingerprint"] != fingerprint:
        raise RuntimeError("held-out manifest linked to incorrect training pool")
    fixed_dir, heldout_dir = out / "standard_results", out / "heldout_results"
    for directory in (fixed_dir, heldout_dir):
        directory.mkdir(exist_ok=True)
    standard_reference = [(r["stage"], r["case"], r["init"])
                          for r in archived_fixed(out, "t3", 240000)["rows"]]
    heldout_reference = None
    specs = {r["id"]: r for r in manifest["maps"]}
    for seed in SEEDS:
        for model in MODELS:
            for node, checkpoint in registry[seed, model].items():
                path = Path(checkpoint["path"])
                file = fixed_dir / f"{model}_{seed}_{node}.json"
                if file.exists():
                    fixed = read_json(file)
                elif seed == 43:
                    archived = archived_fixed(out, model, node)
                    fixed = dict(rows=annotate_standard(archived.get("rows", archived.get("forest"))),
                                 checkpoint_sha256=checkpoint["sha256"])
                    if "checkpoint_sha256" in archived and archived["checkpoint_sha256"] != checkpoint["sha256"]:
                        raise RuntimeError("Seed43 fixed archive digest mismatch")
                else:
                    rows = eval_jobs(model, seed, node, path, fingerprint, args.workers)
                    fixed = dict(rows=rows, checkpoint_sha256=checkpoint["sha256"])
                base.verify_initializations(fixed["rows"], standard_reference)
                if fixed["checkpoint_sha256"] != checkpoint["sha256"]:
                    raise RuntimeError("cached standard evaluation checksum mismatch")
                write_json(file, dict(fixed, seed=seed, model=model, step=node))
                print(f"{model} Seed{seed} step {node}: 80 standard scenes verified", flush=True)
                if node not in FINAL_NODES:
                    continue
                file = heldout_dir / f"{model}_{seed}_{node}.json"
                if file.exists():
                    heldout = read_json(file)
                else:
                    rows = eval_jobs(model, seed, node, path, fingerprint, args.workers, manifest)
                    heldout = dict(rows=rows, checkpoint_sha256=checkpoint["sha256"],
                                   suite_fingerprint=manifest["fingerprint"])
                rows = heldout["rows"]
                if (len(rows) != 144 or {r["id"] for r in rows} != specs.keys()
                        or heldout["checkpoint_sha256"] != checkpoint["sha256"]
                        or heldout["suite_fingerprint"] != manifest["fingerprint"]
                        or any(r["init"]["geometry_sha256"] != specs[r["id"]]["geometry_sha256"] for r in rows)):
                    raise RuntimeError("sealed evaluation cohort mismatch")
                initials = [(r["id"], r["init"]) for r in rows]
                if heldout_reference is None:
                    heldout_reference = initials
                if initials != heldout_reference:
                    raise RuntimeError("held-out initial state mismatch across models/steps")
                write_json(file, dict(heldout, seed=seed, model=model, step=node))
                print(f"{model} Seed{seed} step {node}: 144 sealed scenes verified", flush=True)


if __name__ == "__main__":
    main()
