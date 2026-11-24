"""Evaluate the T2 nodes on paired fixed forest maps and write the T2 decision."""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
from statistics import median
import subprocess
import time

import torch

from evaluate import load_policy, make_validation_env, realtime_decisions
from t0_eval import follow, initial_info
from train import sha

NODES = (40000, 80000, 120000)
INIT_FIELDS = ("physical_state20", "target", "geometry_sha256", "map_pool_index")


def read_json(path):
    return json.loads(path.read_text())


def read_lines(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.endswith("\n")]


def write_json(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def historical_a2(runs):
    formal = runs / "formal_gpu_seed43_seven_nodes_eval_20261118a/a2_240000/results.json"
    selected = runs / "msgp_e02_e03_seed43_20261119_selected"
    paths = [formal] + [selected / f"a2_s{stage}_{case:02d}/results.json"
                        for stage in (1, 2) for case in range(0, 20, 5)]
    baseline = {}
    for path in paths:
        for row in read_json(path):
            key = row["stage"], row["seed"]
            if key in baseline:
                raise RuntimeError(f"duplicate historical A2 case: {key}")
            baseline[key] = row
    expected = {(s, c) for s in range(4) for c in range(20)}
    totals = [sum(baseline[s, c]["success"] for c in range(20)) for s in range(4)]
    if baseline.keys() != expected or totals != [16, 11, 5, 1]:
        raise RuntimeError(f"historical A2 pairing totals changed: {totals}")
    return baseline, {str(p.relative_to(runs)): sha(p) for p in paths}


def initialize_worker(checkpoint):
    global agent, builder, contract, checkpoint_step
    torch.set_num_threads(1)
    agent, builder, contract, checkpoint_step = load_policy(checkpoint, "cpu")
    if (contract["course"], contract["seed"], contract["use_far_branch"], contract["short"]) != (
            "original", 43, False, False):
        raise RuntimeError("checkpoint is not the T2 formal near-only variant")


def evaluate_cases(stage, cases):
    env = make_validation_env(contract)
    env.obstacle_curriculum.stage = stage
    env_step = env.step
    rows = []
    try:
        for case in cases:
            obs, info = env.reset(seed=case)
            if (info["density_stage"], info["start_curriculum_source"]) != (stage, "original"):
                raise RuntimeError("fixed evaluation start mismatch")
            initial = initial_info(env, info)
            iv = deque([(env.base_env.physics_ticks, float(info["goal_distance"]))], maxlen=61)
            windows = stalls = 0

            def track_progress(action):
                nonlocal windows, stalls
                outcome = env_step(action)
                iv.append((env.base_env.physics_ticks, float(outcome[-1]["goal_distance"])))
                if iv[-1][0] - iv[0][0] >= 4 * 240:
                    windows += 1
                    stalls += iv[0][1] - iv[-1][1] < .12
                return outcome

            env.step = track_progress
            with realtime_decisions():
                row, _ = follow(env, agent, builder, obs, info)
            rows.append(dict(stage=stage, case=case, init=initial, stall_windows=stalls,
                             windows_at_least_4s=windows, **row))
    finally:
        env.close()
    return checkpoint_step, rows


def paired_test(rows, baseline):
    trial = {(r["stage"], r["case"]): r for r in rows}
    changes = [sum(int(trial[s, c]["success"]) - int(baseline[s, c]["success"])
                   for s in range(3)) for c in range(20)]
    # Group the three nested densities for the same map seed into one paired unit.
    observed = sum(changes)
    distribution = Counter({0: 1})
    for value in changes:
        new = Counter()
        for prior, count in distribution.items():
            new[prior + value] += count
            new[prior - value] += count
        distribution = new
    p = sum(count for gain, count in distribution.items() if gain >= observed) / 2 ** len(changes)
    return dict(a2_success=32, t2_success=32 + observed, net=observed,
                map_level_net_success=changes, map_seeds=len(changes),
                test="one-sided exact paired sign permutation over map-seed aggregates",
                p=p, alpha=.05, exceeds=observed > 0 and p < .05)


def group_stats(rows):
    n = len(rows)
    reasons = Counter(r["reason"] for r in rows)
    collision = sum(r["collision_observed"] for r in rows)
    bounds = reasons["xy_bound"] + reasons["height_bound"]
    timeouts = [r for r in rows if r["reason"] == "timeout"]
    stall_share = [r["stall_windows"] / r["windows_at_least_4s"] for r in timeouts if r["windows_at_least_4s"]]
    counts = dict(success=reasons["success"], collision=collision, timeout=reasons["timeout"], bounds=bounds)
    return dict(n=n, counts=counts, rates={name: count / n for name, count in counts.items()},
                reasons=dict(reasons), median_goal_progress_m=median(r["goal_progress_m"] for r in rows),
                timeout_median_final_goal_m=median(r["final_goal_distance_m"] for r in timeouts) if timeouts else None,
                timeout_median_stall_window_share=median(stall_share) if stall_share else None)


def training_stats(run, step):
    metric, = [r for r in read_lines(run / "metrics.jsonl") if r["step"] == step]
    transitions = []
    prior = (0, 0)
    for row in read_lines(run / "episodes.jsonl"):
        if row["step"] > step:
            break
        course = row["course"]
        current = course["start_curriculum_level"], course["obstacle_curriculum_stage"]
        if current != prior:
            transitions.append(dict(step=row["step"], start_level=current[0], obstacle_stage=current[1]))
            prior = current
    metric["course_transitions"] = transitions
    return metric


def evaluate(node, checkpoint, checksum, train, output, baseline, workers):
    actual_sha = sha(checkpoint)
    if actual_sha != checksum:
        raise RuntimeError("checkpoint digest does not match training event")
    rows = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=initialize_worker,
                             initargs=(checkpoint,)) as pool:
        tasks = [pool.submit(evaluate_cases, stage, range(i, i + 5))
                 for stage in range(4) for i in range(0, 20, 5)]
        for future in as_completed(tasks):
            step, batch = future.result()
            if step != node:
                raise RuntimeError("loaded checkpoint at wrong step")
            rows.extend(batch)
            print(f"{node}: evaluated {len(rows)}/80", flush=True)
    rows.sort(key=lambda r: (r["stage"], r["case"]))
    keys = {(r["stage"], r["case"]) for r in rows}
    if len(rows) != 80 or keys != baseline.keys():
        raise RuntimeError("fixed Stage0–3 evaluation is incomplete")
    for r in rows:
        reference = baseline[r["stage"], r["case"]]["initial"]
        if any(r["init"][field] != reference[field] for field in INIT_FIELDS):
            raise RuntimeError(f"historical A2 case mismatch at {(r['stage'], r['case'])}")
    stages = {str(s): group_stats([r for r in rows if r["stage"] == s]) for s in range(4)}
    doc = dict(step=node, checkpoint_sha256=actual_sha, paired_a2_cases=len(rows), stages=stages,
               device="cpu", training=training_stats(train, node), paired_a2=paired_test(rows, baseline), rows=rows)
    write_json(output, doc)
    return doc


def update_scoreboard(nodes, output):
    last = nodes.get(120000)
    gate = None
    if last:
        stage0 = last["stages"]["0"]["counts"]["success"]
        gate = dict(stage0_success=stage0, stage0_gate=stage0 >= 18,
                    significantly_exceeds_a2=last["paired_a2"]["exceeds"],
                    a2_test=last["paired_a2"], stage0_stop_threshold=stage0 < 14,
                    stage_counts={stage: value["counts"] for stage, value in last["stages"].items()})
        gate["numerical_release_conditions_met"] = gate["stage0_gate"] and gate["significantly_exceeds_a2"]
    write_json(output, dict(nodes={step: dict(stages=doc["stages"], training=doc["training"],
                                           paired_a2=doc["paired_a2"]) for step, doc in nodes.items()},
                            final_gate=gate))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--training-unit", required=True)
    parser.add_argument("--export", type=Path, required=True)
    args = parser.parse_args()
    train = args.root / "train"
    baseline, sources = historical_a2(args.root.parent)
    args.export.mkdir(parents=True, exist_ok=True)
    protocol = dict(nodes=NODES, stages=list(range(4)), seeds=list(range(20)), split="validation",
                    original_start=True, device="cpu", workers=args.workers, historical_a2_source_sha256=sources,
                    stage0_required_success=18, stage0_stop_below=14,
                    paired_test="one-sided exact map-seed sign permutation", alpha=.05)
    write_json(args.root / "evaluation_protocol.json", protocol)
    write_json(args.export / "evaluation_protocol.json", protocol)
    docs = {}
    for node in NODES:
        checkpoint = train / f"checkpoint_{node:06d}.pt"
        output = args.root / f"evaluation_{node:06d}.json"
        existing = output.is_file()
        while True:
            checks = [r["sha256"] for r in read_lines(train / "events.jsonl")
                      if r["event"] == "checkpoint" and r["step"] == node]
            if len(checks) == 1 and checkpoint.is_file():
                break
            status = subprocess.run(["systemctl", "is-active", "--quiet", args.training_unit], check=False)
            if status.returncode != 0:
                raise RuntimeError(f"training terminated before checkpoint {node}")
            time.sleep(15)
        if existing:
            doc = read_json(output)
            if doc["checkpoint_sha256"] != checks[0] or doc["step"] != node or len(doc["rows"]) != 80:
                raise RuntimeError(f"existing evaluation {node} does not match this experiment")
        else:
            doc = evaluate(node, checkpoint, checks[0], train, output, baseline, args.workers)
        docs[node] = doc
        write_json(args.export / output.name, doc)
        for dest in (args.root, args.export):
            update_scoreboard(docs, dest / "t2_nodes.json")


if __name__ == "__main__":
    main()
