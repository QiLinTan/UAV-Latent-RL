"""CPU-only frozen A1/T3 Suite D evaluation with restartable per-map results."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import gzip
import itertools
import json
import multiprocessing
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pybullet
import torch

from contract import BASE, CONTROL_HZ, HERE, capture_initial, digest, load_manifest, make_env, read
from geometry import solve
from metrics import clearance, obstacle_match, stats, terminal_failure
from evaluate import realtime_decisions
from experiments.rv1_range_cnn.lidar import LidarConfig
import t3_eval as policy
from t4_evaluate import CHECKED_CONFIG, FINAL_NODES, MODELS, SEEDS, load_worker, run_path
from train import sha

EXPERIMENT = tuple(itertools.product(SEEDS, MODELS, FINAL_NODES))


def registry(runs, fingerprint):
    archived = read(HERE.parent / "frozen_seed43.json")["models"]
    standard = read(run_path(runs, 43, "t3") / "metadata.json")["config"]
    ready = []
    missing = []
    for seed, model, checkpoint in EXPERIMENT:
        folder = run_path(runs, seed, model)
        path = folder / f"checkpoint_{checkpoint:06d}.pt"
        identifier = f"{model}_{seed}_{checkpoint}"
        if not path.is_file():
            missing.append(dict(id=identifier, path=str(path), reason="checkpoint missing"))
            continue
        metadata = read(folder / "metadata.json")
        config = metadata["config"]
        contract = metadata["contract"]
        pool = metadata.get("pool_fingerprint", contract.get("pool_fingerprint"))
        if (config["seed"] != seed or pool != fingerprint
                or any(config[key] != standard[key] for key in CHECKED_CONFIG)):
            raise RuntimeError(f"training contract drift: {identifier}")
        if seed == 43:
            expected = archived[model]["checkpoints"][str(checkpoint)]["sha256"]
            provenance = "frozen_seed43"
        else:
            events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines() if line.strip()]
            event = next((record for record in reversed(events) if record.get("event") == "checkpoint"
                          and record.get("step") == checkpoint), None)
            expected = event.get("sha256") if event else None
            provenance = "training_event"
            standard_result = HERE.parent / "standard_results" / f"{identifier}.json"
            if not expected and standard_result.is_file():
                expected = read(standard_result).get("checkpoint_sha256")
                if expected:
                    provenance = "t4_standard_evaluation"
            if (not expected and model == "a1" and event and event.get("file") == path.name
                    and any(record.get("event") == "stopped" and record.get("complete")
                            and record.get("step") == 240000 for record in events)):
                provenance = "suite_d_preflight_digest_from_completed_a1_save_event"
        if not expected and provenance != "suite_d_preflight_digest_from_completed_a1_save_event":
            missing.append(dict(id=identifier, path=str(path), reason="no sealed checkpoint digest"))
            continue
        checksum = sha(path)
        if expected and checksum != expected:
            raise RuntimeError(f"checkpoint checksum mismatch: {path}")
        ready.append(dict(id=identifier, model=model, seed=seed, checkpoint=checkpoint,
                          path=str(path), sha256=checksum, digest_provenance=provenance))
    return ready, missing


def replace_json(path, result):
    temp = path.with_suffix(".tmp")
    compressed = path.suffix == ".gz"
    opener = gzip.open if compressed else open
    with opener(temp, "wt", encoding="utf-8", **({"compresslevel": 3} if compressed else {})) as stream:
        json.dump(result, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    temp.replace(path)


def identifiers(env, case):
    base = env.base_env
    mapping = dict(zip(base._map_runtime.body_ids, [obstacle["name"] for obstacle in case["map"]["obstacles"]]))
    mapping[int(base.PLANE_ID)] = "ground_plane"
    return mapping


def terminal_contacts(base, mapping):
    return [dict(id=int(contact[2]), name=mapping[int(contact[2])], point=list(contact[6]),
                 normal=list(contact[7]), separation_m=float(contact[8]))
            for contact in pybullet.getContactPoints(bodyA=int(base.DRONE_IDS[0]), physicsClientId=base.CLIENT)
            if int(contact[2]) in mapping]


def sample_scan(scan):
    hit = scan.hit_mask.ravel()
    return dict(body=scan.hit_body.ravel()[hit], xyz=scan.raw_hit_world.reshape(-1, 3)[hit],
                origin=scan.origin_world, directions=scan.directions_world.reshape(-1, 3)[hit])


def evaluate_case(case, folder, entry, fingerprint):
    path = folder / f"{case['id']}.json"
    if path.exists():
        completed = read(path)
        if (completed["run"] != entry or completed["suite_fingerprint"] != fingerprint
                or completed["initial_digest"] != digest(case["initial"])
                or not (folder / completed["trace"]).is_file()):
            raise RuntimeError(f"incompatible cached result: {path}")
        return completed
    env = make_env(case, LidarConfig(**policy.contract["sensor"]))
    try:
        obs, initial = capture_initial(env, case["reset_seed"])
        if digest(initial) != digest(case["initial"]):
            raise RuntimeError(f"initial contract mismatch: {case['id']}")
        base = env.base_env
        mapping = identifiers(env, case)
        positions, velocities = [base.pos[0].copy()], [base.vel[0].copy()]
        heading = [float(base.rpy[0, 2])]
        tail = deque(maxlen=30)
        with realtime_decisions():
            while True:
                scan = env.lidar.cache
                if scan.hit_mask.size != 4096 or not scan.scanned_mask.all():
                    raise RuntimeError("raw LiDAR coverage changed")
                action = policy.agent.action(policy.builder(obs))
                obs, _, terminated, truncated, info = env.step(action)
                tail.append((int(env.upper_tick) - 1, sample_scan(scan), np.asarray(action).ravel().tolist()))
                for tick in env.last_ticks:
                    positions.append(tick["position"])
                    velocities.append(tick["velocity"])
                    heading.append(tick["rpy"][2])
                if terminated or truncated:
                    break
                if base.physics_ticks > case["timeout_physics_ticks"]:
                    raise RuntimeError("timeout was not enforced")
        outcome = info["done_reason"]
        contacts = terminal_contacts(base, mapping) if outcome == "collision" else []
        if outcome == "collision" and not contacts:
            raise RuntimeError("collision terminated without terminal contact")
        if outcome == "height_bound":
            outcome = "low_height_bound" if positions[-1][2] < .08 else "high_height_bound"
        decisions = list(tail)
        evidence = [obstacle_match([scan for _, scan, _ in decisions], contact) for contact in contacts]
    finally:
        env.close()
    graph = solve(case["map"])
    if abs(graph.length - case["l_geo_m"]) > 1e-8:
        raise RuntimeError(f"frozen route mismatch: {case['id']}")
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)
    progress, usable = graph.progress(positions)
    clearances = clearance(case["map"], positions)
    measures = stats(case, positions, velocities, clearances, progress, usable, outcome, CONTROL_HZ)
    failure, flags = (None, []) if outcome == "success" else terminal_failure(
        case, outcome, positions, velocities, heading, progress, contacts, evidence, CONTROL_HZ)
    trace = path.with_suffix(".trace.json.gz")
    replace_json(trace, dict(position_m=positions.tolist(), velocity_mps=velocities.tolist(), yaw_rad=heading,
                             geodesic_progress_m=progress.tolist(), progress_valid=usable.tolist(),
                             clearance_m=clearances.tolist(), control_hz=CONTROL_HZ,
                             tail_decisions=[dict(step=step, action=action, origin=scan["origin"].tolist(),
                                                  hit_xyz=scan["xyz"].tolist(),
                                                  hit_object=[mapping.get(int(body), "other") for body in scan["body"]])
                                             for step, scan, action in decisions]))
    result = dict(id=case["id"], topology=case["topology"], geometry_sha256=case["geometry_sha256"],
                  suite_fingerprint=fingerprint, initial_digest=digest(initial), run=entry, terminal=outcome,
                  terminal_contacts=contacts, contact_match_last_three_frames=evidence, failure_bucket=failure,
                  matching_buckets=flags, goal_distance_m=float(info["goal_distance"]), trace=trace.name,
                  phase3=None, **measures)
    replace_json(path, result)
    return result


def initialize(entry, fingerprint):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    load_worker(entry["model"], entry["seed"], entry["checkpoint"], Path(entry["path"]), fingerprint)
    if entry["model"] == "t3" and not policy.contract["use_far_branch"]:
        raise RuntimeError("T3 dual-branch checkpoint required")
    if policy.contract["sensor"] != vars(LidarConfig()):
        raise RuntimeError("policy uses a different native LiDAR contract")


def quiet_case(*args):
    with open(os.devnull, "w", encoding="utf-8") as stream, contextlib.redirect_stdout(stream):
        return evaluate_case(*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--runs", type=Path, default=BASE / "runs")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    sources = {name: sha(HERE / name) for name in ("contract.py", "geometry.py", "metrics.py", "run.py")}
    ready, missing = registry(args.runs, manifest["training_pool_fingerprint"])
    print(json.dumps(dict(ready=[entry["id"] for entry in ready], missing=missing,
                         suite_fingerprint=manifest["fingerprint"]), ensure_ascii=False), flush=True)
    if args.preflight:
        return
    if missing and not args.pilot:
        raise RuntimeError("three-seed paired grid is incomplete; no formal runs started")
    if args.pilot:
        args.output = args.output / "pilot"
        ready = [entry for entry in ready if entry["seed"] == 43 and entry["checkpoint"] == 240000]
    if not ready:
        raise RuntimeError("no checkpoints selected")
    cases = [case for case in manifest["cases"] if not args.pilot or case["index"] in (0, 19)]
    args.output.mkdir(parents=True, exist_ok=True)
    run_meta = dict(manifest=manifest["fingerprint"], sources=sources, mode="pilot" if args.pilot else "formal",
                    checkpoints=ready, expected_rollouts=len(ready) * len(cases))
    run_meta_path = args.output / "run_manifest.json"
    if run_meta_path.exists() and read(run_meta_path) != run_meta:
        raise RuntimeError("existing output is bound to another manifest/runner; use a fresh output directory")
    replace_json(run_meta_path, run_meta)
    context = multiprocessing.get_context("spawn")
    for entry in ready:
        target = args.output / entry["id"]
        target.mkdir(exist_ok=True)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context, initializer=initialize,
                                 initargs=(entry, manifest["training_pool_fingerprint"])) as executor:
            jobs = [executor.submit(quiet_case, case, target, entry, manifest["fingerprint"]) for case in cases]
            for index, job in enumerate(as_completed(jobs), 1):
                row = job.result()
                if args.pilot or index % 10 == 0 or index == len(cases):
                    print(f"{entry['id']} {index}/{len(cases)} {row['id']} {row['terminal']}", flush=True)
    print(f"completed {run_meta['expected_rollouts']} rollouts", flush=True)


if __name__ == "__main__":
    main()
