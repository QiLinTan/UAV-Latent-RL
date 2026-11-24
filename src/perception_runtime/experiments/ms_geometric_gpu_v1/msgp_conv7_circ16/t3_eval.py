"""Evaluate T3 checkpoints and archived comparators on common fixed forest cases."""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import multiprocessing
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from evaluate import load_policy, make_validation_env, realtime_decisions
from t0_eval import follow, identity, initial_info
from t2_diag_traces import xy_bounds
from t2_eval import group_stats, read_json, read_lines, write_json
from t3_e12 import measure, summarize
from train import sha

from experiments.direct_velocity_full_v1.task_factory import load_config
from experiments.rv1_range_cnn.lidar import LidarConfig
from experiments.rv1_range_cnn.td3 import RangeTD3
from ms_td3 import GeometricTD3, ObservationEncoder

NODES = (40000, 80000, 100000, 120000, 140000, 160000)
E12_NODES = (80000, 120000, 160000)
T2_NODES = (40000, 80000, 100000, 120000)


def init_worker(model, checkpoint, fingerprint):
    global agent, builder, contract, step
    torch.set_num_threads(1)
    if model == "t3":
        agent, builder, contract, step = load_policy(checkpoint, "cpu")
        if (contract["course"], contract["seed"], contract["use_far_branch"], contract["short"]) != (
                "original", 43, True, False):
            raise RuntimeError("wrong checkpoint for formal T3 evaluation")
        return
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    c = state["contract"]
    cfg = load_config()
    step = state["trainer"]["step"]
    if model == "a1":
        agent = RangeTD3(cfg["td3"], c, device="cpu")
        builder = identity
    else:
        if c["mode"] != "near_far" or c["seed"] != 43 or c["pool_fingerprint"] != fingerprint:
            raise RuntimeError("wrong original A4 checkpoint")
        agent = GeometricTD3(cfg["td3"], device="cpu")
        builder = ObservationEncoder(LidarConfig())
    agent.actor.load_state_dict(state["agent"]["actor"])
    agent.actor.eval()
    contract = dict(sensor=asdict(LidarConfig()), pool_fingerprint=fingerprint,
                    lidar_extraction="parallel_pybullet_v1")


def forest(stage, seeds):
    env = make_validation_env(contract)
    env.obstacle_curriculum.stage = stage
    original_step = env.step
    rows = []
    try:
        for seed in seeds:
            obs, info = env.reset(seed=seed)
            if (info["density_stage"], info["start_curriculum_source"]) != (stage, "original"):
                raise RuntimeError("fixed forest start contract mismatch")
            init = initial_info(env, info)
            window = deque([(env.base_env.physics_ticks, float(info["goal_distance"]))], maxlen=61)
            last_world_y = deque(maxlen=15)
            stalls = windows = 0

            def track(action):
                nonlocal stalls, windows
                outcome = original_step(action)
                meta = outcome[-1]
                window.append((env.base_env.physics_ticks, float(meta["goal_distance"])))
                last_world_y.append(float(meta["accepted_world_command"][1]))
                if window[-1][0] - window[0][0] >= 4 * 240:
                    windows += 1
                    stalls += window[0][1] - window[-1][1] < .12
                return outcome

            env.step = track
            with realtime_decisions():
                row, _ = follow(env, agent, builder, obs, info)
            position = env.base_env.pos[0].tolist()
            bounds = xy_bounds(env.base_env)
            exits = [axis + ("+" if pos >= 0 else "-")
                     for axis, pos, limit in zip("xy", position, bounds)
                     if abs(np.float32(pos)) > limit] if row["reason"] == "xy_bound" else []
            rows.append(dict(stage=stage, case=seed, init=init, stall_windows=stalls,
                             windows_at_least_4s=windows, final_position_m=position,
                             xy_exits=exits, last15_world_vy_mean_mps=float(np.mean(last_world_y)), **row))
    finally:
        env.close()
    return step, rows


def aperture(case):
    return step, measure(case, agent, builder)


def count_outcomes(rows):
    counts = group_stats(rows)
    counts["counts"].update(height_bound=counts["reasons"].get("height_bound", 0),
                            xy_bound=counts["reasons"].get("xy_bound", 0))
    counts["xy_exits_by_side"] = dict(Counter(edge for row in rows for edge in row.get("xy_exits", [])))
    return counts


def verify_initializations(rows, reference):
    if len(rows) != 80 or [(r["stage"], r["case"], r["init"]) for r in rows] != reference:
        raise RuntimeError("forest cases did not match the fixed T2 scenes")


def evaluate(model, node, checkpoint, checksum, fingerprint, reference, workers, output,
             complete_forest=True, e12=False):
    if sha(checkpoint) != checksum:
        raise RuntimeError(f"checkpoint digest mismatch for {model} {node}")
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=init_worker,
                             initargs=(model, checkpoint, fingerprint)) as pool:
        futures = {}
        if complete_forest:
            for stage in range(4):
                for seed in range(0, 20, 5):
                    futures[pool.submit(forest, stage, range(seed, seed + 5))] = "forest"
        if e12:
            for case in range(12):
                futures[pool.submit(aperture, case)] = "E12"
        rows, apertures = [], []
        for future in as_completed(futures):
            loaded, result = future.result()
            if loaded != node:
                raise RuntimeError(f"evaluated wrong node, got {loaded} expected {node}")
            kind = futures[future]
            if kind == "forest":
                rows.extend(result)
                print(f"{model} {node} forest {len(rows)}/80", flush=True)
            else:
                apertures.append(result)
                print(f"{model} {node} E12 {len(apertures)}/12", flush=True)
    if complete_forest:
        rows.sort(key=lambda r: (r["stage"], r["case"]))
        verify_initializations(rows, reference)
        write_json(output / f"{model}_{node:06d}.json", dict(
            model=model, step=node, device="cpu", checkpoint_sha256=checksum,
            stages={str(s): count_outcomes([r for r in rows if r["stage"] == s]) for s in range(4)},
            totals=count_outcomes(rows), rows=rows))
    if e12:
        apertures.sort(key=lambda r: r["case"])
        write_json(output / f"e12_{node:06d}.json", dict(step=node, device="cpu",
                   checkpoint_sha256=checksum, summary=summarize(apertures), cases=apertures))


def checkpoints(run):
    return {r["step"]: r.get("sha256") for r in read_lines(run / "events.jsonl") if r["event"] == "checkpoint"}


def history(runs, output, t2, reference, fingerprint, workers):
    older = runs / "formal_gpu_seed43_seven_nodes_eval_20261118a"
    t2_source = t2 / "train"
    for node in T2_NODES:
        path = (output.parent.parent / "t2_near_seed43_120k/t2_diag/fixed_100000.json" if node == 100000
                else t2 / f"evaluation_{node:06d}.json")
        doc = read_json(path)
        if checkpoints(t2_source)[node] != doc["checkpoint_sha256"]:
            raise RuntimeError("T2 historical fixed evaluation digest mismatch")
        verify_initializations(doc["rows"], reference)
        rows = doc["rows"]
        write_json(output / f"t2_{node:06d}.json", dict(
            model="t2", step=node, device="cpu", checkpoint_sha256=doc["checkpoint_sha256"],
            stages={str(stage): count_outcomes([r for r in rows if r["stage"] == stage]) for stage in range(4)},
            totals=count_outcomes(rows), rows=rows))
    for model in ("a1", "a4"):
        source = runs / f"formal_gpu_{model}_seed43_240k_20261118a"
        stored = checkpoints(source)
        for node in NODES:
            filename = output / f"{model}_{node:06d}.json"
            if filename.exists():
                verify_initializations(read_json(filename)["rows"], reference)
                continue
            checkpoint = source / f"checkpoint_{node:06d}.pt"
            protocol = older / f"{model}_{node:06d}/protocol.json"
            checksum = stored.get(node)
            if protocol.is_file():
                archived = read_json(protocol)
                if checksum is not None and checksum != archived["checkpoint_sha256"]:
                    raise RuntimeError("inconsistent historical hash")
                checksum = archived["checkpoint_sha256"]
            # The RV1 trainer recorded hashes for separate Replay and saved no model
            # digest in its events at 100k/140k. Record the observed archive SHA.
            checksum = checksum or sha(checkpoint)
            evaluate(model, node, checkpoint, checksum, fingerprint, reference, workers, output)


def watch(root, output, unit, reference, fingerprint, workers):
    train = root / "train"
    for node in NODES:
        checkpoint = train / f"checkpoint_{node:06d}.pt"
        while True:
            ready = checkpoints(train).get(node)
            if ready and checkpoint.is_file():
                break
            if subprocess.run(["systemctl", "is-active", "--quiet", unit], check=False).returncode:
                raise RuntimeError(f"T3 training stopped before {node}")
            time.sleep(15)
        forest_done = (output / f"t3_{node:06d}.json").is_file()
        e12_done = node not in E12_NODES or (output / f"e12_{node:06d}.json").is_file()
        if not (forest_done and e12_done):
            evaluate("t3", node, checkpoint, ready, fingerprint, reference, workers, output,
                     complete_forest=not forest_done, e12=not e12_done)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("watch", "history"), required=True)
    parser.add_argument("--training-unit", default="msgp-conv7-t3-train.service")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    runs = args.root.parent
    t2 = runs / "msgp_conv7_t2_near_seed43_120k_20261120a"
    base = read_json(t2 / "evaluation_080000.json")
    reference = [(r["stage"], r["case"], r["init"]) for r in base["rows"]]
    fingerprint = read_json(t2 / "train/metadata.json")["contract"]["pool_fingerprint"]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "watch":
        write_json(args.output / "evaluation_protocol.json", dict(
            nodes=NODES, e12_nodes=E12_NODES, seeds=list(range(20)), stages=list(range(4)),
            fixed_original_starts=True, seed=43, device="cpu", pool_fingerprint=fingerprint,
            checkpoint_steps=list(range(40000, 160001, 10000)), hard_stop=160000))
        watch(args.root, args.output, args.training_unit, reference, fingerprint, args.workers)
    else:
        history(runs, args.output, t2, reference, fingerprint, args.workers)


if __name__ == "__main__":
    main()
